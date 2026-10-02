#!/usr/bin/env python3
"""Natural OLMoE expert-accumulation probe without metadata side channels.

The candidate and repair share model weights, routing, expert outputs, inputs,
and loss.  The only changed operation is the repeated-destination combine:
native BF16 accumulation versus FP32 accumulation followed by one cast.
The runner records local loss, gradients, and a declared cold-start AdamW
parameter-write contrast for a real input bank.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/allenai/OLMoE-1B-7B-0125")
DEFAULT_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/olmoe_1b7b_text128.json"
DEFAULT_OUTPUT = ROOT / "results/property/olmoe_router_accum_natural_v1/result.json"


def _router_forward(self: Any, hidden_states: torch.Tensor, *, repair: bool, trace: dict[str, Any]) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    # Current Transformers returns logits from the gate module; routing is
    # performed by the surrounding sparse-MoE block.  Reproduce that contract
    # exactly before changing only the repeated-destination accumulation dtype.
    router_logits = self.gate(flat)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1
    )
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(flat.dtype)
    final_dtype = flat.dtype
    accum_dtype = torch.float32 if repair else final_dtype
    combined = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=accum_dtype,
        device=flat.device,
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.num_experts
    ).permute(2, 1, 0)
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current = flat[None, top_x].reshape(-1, hidden_dim)
        gate = expert_layer.gate_proj(current)
        up = expert_layer.up_proj(current)
        current = expert_layer.act_fn(gate) * up
        current = expert_layer.down_proj(current)
        current = current * routing_weights[top_x, idx, None]
        combined.index_add_(0, top_x, current.float() if repair else current.to(final_dtype))
    trace["selected_experts"] = selected_experts.detach().cpu()
    trace["routing_weights"] = routing_weights.detach().float().cpu()
    return combined.to(final_dtype).reshape(batch_size, sequence_length, hidden_dim), router_logits


def _patch(model: torch.nn.Module, layer: int, repair: bool, trace: dict[str, Any]) -> None:
    block = model.model.layers[layer].mlp
    block.forward = types.MethodType(
        lambda self, hidden_states: _router_forward(self, hidden_states, repair=repair, trace=trace),
        block,
    )


def _grads(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and any(name.startswith(prefix) for prefix in prefixes)
    }


def _vector(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([values[name].reshape(-1) for name in sorted(values)]) if values else torch.empty(0)


def _write(gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
    # Step one, zero moments, FP32 master, no weight decay.
    return -learning_rate * gradient / (gradient.abs() + 1e-8)


def _summary(effects: list[torch.Tensor], references: list[torch.Tensor], split: int) -> dict[str, Any]:
    matrix = torch.stack([x.double() for x in effects])
    refs = torch.stack([x.double() for x in references])
    cal = matrix[:split]
    conf = matrix[split:]
    direction = cal.mean(dim=0)
    norm = float(direction.norm())
    if norm > 0 and len(conf):
        direction = direction / norm
        projections = conf @ direction
        mean = float(projections.mean())
        sd = float(projections.std(unbiased=True)) if len(projections) > 1 else 0.0
        half = 1.96 * sd / math.sqrt(len(projections)) if len(projections) > 1 else None
        signs = {
            "positive": int((projections > 0).sum()),
            "negative": int((projections < 0).sum()),
            "zero": int((projections == 0).sum()),
        }
        interval = [mean - half, mean + half] if half is not None else None
    else:
        mean, sd, interval, signs = None, None, None, None
    effect_energy = float(matrix.square().sum(dim=1).mean())
    reference_energy = float(refs.square().sum(dim=1).mean())
    aligned = []
    for effect, reference in zip(matrix[split:], refs[split:]):
        denominator = float(torch.dot(reference, reference))
        aligned.append(float(torch.dot(effect, reference)) / max(denominator, 1e-30))
    if aligned:
        aligned_mean = sum(aligned) / len(aligned)
        aligned_sd = math.sqrt(sum((value - aligned_mean) ** 2 for value in aligned) / max(len(aligned) - 1, 1))
        aligned_interval = [
            aligned_mean - 1.96 * aligned_sd / math.sqrt(len(aligned)),
            aligned_mean + 1.96 * aligned_sd / math.sqrt(len(aligned)),
        ]
    else:
        aligned_mean, aligned_interval = None, None
    return {
        "state_count": int(matrix.shape[0]),
        "calibration_count": int(split),
        "confirmation_count": int(len(conf)),
        "effect_rms_over_reference": math.sqrt(effect_energy / reference_energy) if reference_energy > 0 else None,
        "mean_effect_over_reference_rms": float(matrix.mean(dim=0).norm()) / math.sqrt(reference_energy) if reference_energy > 0 else None,
        "calibration_direction_norm": norm,
        "confirmation_projection_mean": mean,
        "confirmation_projection_sd": sd,
        "confirmation_projection_interval_normal_95": interval,
        "confirmation_signs": signs,
        "confirmation_aligned_mean": aligned_mean,
        "confirmation_aligned_interval_normal_95": aligned_interval,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--states", type=int, default=26)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--scope", choices=("router_gate", "layer_attn", "layer_norm"), default="router_gate")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if args.start < 0 or args.start + args.states > len(states):
        raise ValueError("input bank is shorter than the requested range")
    if args.states < 4 or args.states % 2:
        raise ValueError("states must be an even number >= 4")
    if args.scope == "router_gate":
        prefixes = (f"model.layers.{args.layer}.mlp.gate.",)
    elif args.scope == "layer_attn":
        prefixes = (f"model.layers.{args.layer}.self_attn.",)
    else:
        prefixes = (f"model.layers.{args.layer}.input_layernorm.",)

    torch.manual_seed(24000)
    torch.cuda.manual_seed_all(24000)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    ).to(device).train()
    model.config.use_cache = False
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError("layer outside model")
    target = model.model.layers[args.layer].mlp
    effects = {"gradient": [], "write": []}
    references = {"gradient": [], "write": []}
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states[args.start : args.start + args.states]):
        ids = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
        arms: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]] = {}
        for name, repair in (("candidate", False), ("repair", True)):
            trace: dict[str, Any] = {}
            _patch(model, args.layer, repair, trace)
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids, use_cache=False, return_dict=False)[0]
            loss.backward()
            grads = _grads(model, prefixes)
            arms[name] = (loss.detach().float().cpu(), grads, trace)
        candidate_loss, candidate_grads, candidate_trace = arms["candidate"]
        repair_loss, repair_grads, repair_trace = arms["repair"]
        if set(candidate_grads) != set(repair_grads):
            raise RuntimeError("candidate/repair gradient reach differs")
        gradient_effect = _vector(candidate_grads) - _vector(repair_grads)
        write_effect = _write(_vector(candidate_grads), args.learning_rate) - _write(_vector(repair_grads), args.learning_rate)
        effects["gradient"].append(gradient_effect)
        references["gradient"].append(_vector(repair_grads))
        effects["write"].append(write_effect)
        references["write"].append(_write(_vector(repair_grads), args.learning_rate))
        rows.append({
            "state_id": state.get("state_id", state.get("sequence_id")),
            "loss_candidate": float(candidate_loss),
            "loss_repair": float(repair_loss),
            "loss_difference": float(candidate_loss - repair_loss),
            "routing_same": bool(torch.equal(candidate_trace["selected_experts"], repair_trace["selected_experts"])),
            "routing_weights_same": bool(torch.equal(candidate_trace["routing_weights"], repair_trace["routing_weights"])),
            "gradient_effect_l2": float(gradient_effect.norm()),
            "write_effect_l2": float(write_effect.norm()),
        })
        print(json.dumps({"event": "OLMOE_ROUTER_ACCUM_STATE", "index": index, "state_id": rows[-1]["state_id"]}), flush=True)
        del ids
        torch.cuda.empty_cache()

    split = args.states // 2
    payload = {
        "schema": "kernel-analyzer-olmoe-router-accum-natural-v1",
        "status": "COMPLETE",
        "claim_boundary": "Real OLMoE input states and parameter slice; cold-start AdamW write; not a natural pretraining population guarantee.",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "layer": args.layer,
        "scope": args.scope,
        "state_count": args.states,
        "candidate": "native BF16 repeated-destination expert accumulation",
        "repair": "FP32 accumulation of identical expert contributions followed by one BF16 cast",
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "zero_moments": True, "weight_decay": 0.0},
        "rows": rows,
        "summaries": {name: _summary(effects[name], references[name], split) for name in effects},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
