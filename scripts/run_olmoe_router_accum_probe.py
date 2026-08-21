#!/usr/bin/env python3
"""Two-state F+B probe for the OLMoE router accumulation semantic region.

This is an engineering probe, not a bias verdict.  It keeps the complete
OLMoE forward/backward step unchanged except for one selected
``OlmoeSparseMoeBlock``: the candidate performs the native BF16
``index_add_`` accumulation, while the repair accumulates the same BF16
summands in FP32 and casts once at the end.  Routing indices and top-k values
are captured for both arms so a routing flip cannot be mistaken for an
arithmetic effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def digest_tensor(t: torch.Tensor) -> str:
    x = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(repr(tuple(x.shape)).encode())
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def router_forward(self, hidden_states: torch.Tensor, *, repair: bool, trace: dict[str, Any]):
    # This mirrors transformers.models.olmoe.modeling_olmoe exactly through
    # the target accumulation boundary.  No routing or expert computation is
    # changed by the repair arm.
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(flat)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(flat.dtype)
    final_dtype = flat.dtype
    if repair:
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=torch.float32,
            device=flat.device,
        )
    else:
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=final_dtype,
            device=flat.device,
        )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.num_experts
    ).permute(2, 1, 0)
    for expert_idx, expert_layer in enumerate(self.experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = flat[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        if repair:
            final_hidden_states.index_add_(0, top_x, current_hidden_states.float())
        else:
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(final_dtype))
    trace.setdefault("selected_experts", []).append(selected_experts.detach().cpu().tolist())
    trace.setdefault("routing_weights_digest", []).append(digest_tensor(routing_weights))
    if repair:
        final_hidden_states = final_hidden_states.to(final_dtype)
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


def patch_target(model: torch.nn.Module, layer_idx: int, repair: bool, trace: dict[str, Any]) -> None:
    block = model.model.layers[layer_idx].mlp
    block.forward = types.MethodType(
        lambda self, hidden_states: router_forward(
            self, hidden_states, repair=repair, trace=trace
        ),
        block,
    )


def capture_gradients(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Move a declared parameter slice off GPU before running the repair arm."""
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and any(name.startswith(p) for p in prefixes)
    }


def gradient_stats(
    candidate_grads: dict[str, torch.Tensor],
    repair: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> tuple[dict[str, float | int], torch.Tensor]:
    sq = 0.0
    cand_sq = 0.0
    rep_sq = 0.0
    max_abs = 0.0
    tensors = 0
    repair_names = {
        name
        for name, parameter in repair.named_parameters()
        if parameter.grad is not None and any(name.startswith(p) for p in prefixes)
    }
    if set(candidate_grads) != repair_names:
        raise RuntimeError("candidate/repair gradient reach differs")
    deltas = []
    for name, parameter in repair.named_parameters():
        if parameter.grad is None or not any(name.startswith(p) for p in prefixes):
            continue
        c = candidate_grads[name]
        r = parameter.grad.detach().float().cpu()
        delta = c - r
        sq += float((delta * delta).sum())
        cand_sq += float((c * c).sum())
        rep_sq += float((r * r).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        tensors += 1
        deltas.append(delta.reshape(-1))
    vector = torch.cat(deltas) if deltas else torch.empty(0, dtype=torch.float32)
    return {
        "gradient_delta_l2": sq ** 0.5,
        "candidate_gradient_l2": cand_sq ** 0.5,
        "repair_gradient_l2": rep_sq ** 0.5,
        "gradient_delta_relative_to_repair": sq ** 0.5 / (rep_sq ** 0.5 + 1e-30),
        "gradient_delta_max_abs": max_abs,
        "gradient_tensor_count": tensors,
    }, vector


def run_arm(model: torch.nn.Module, ids: torch.Tensor, layer_idx: int, repair: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    trace: dict[str, Any] = {"arm": "repair" if repair else "candidate"}
    patch_target(model, layer_idx, repair, trace)
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids, use_cache=False, return_dict=False)
    loss = out[0]
    loss.backward()
    return loss.detach(), trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--input-bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument(
        "--scope",
        choices=("router_gate", "layer_attn", "layer_norm", "final_norm", "all"),
        default="router_gate",
        help="parameter slice for formation geometry; all is engineering-only and memory-heavy",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.states < 2:
        raise ValueError("engineering probe needs at least two states")
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if args.start < 0 or args.start + args.states > len(states):
        raise ValueError("input bank shorter than requested states")
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
    if args.scope == "router_gate":
        prefixes = (f"model.layers.{args.layer}.mlp.gate.",)
    elif args.scope == "layer_attn":
        prefixes = (f"model.layers.{args.layer}.self_attn.",)
    elif args.scope == "layer_norm":
        prefixes = (f"model.layers.{args.layer}.input_layernorm.",)
    elif args.scope == "final_norm":
        prefixes = ("model.norm.",)
    else:
        prefixes = ("",)
    rows = []
    delta_vectors = []
    for state in states[args.start : args.start + args.states]:
        ids = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
        # Candidate and repair start from exactly the same state and receive
        # the same input; no weights are updated in this formation probe.
        c_loss, c_trace = run_arm(model, ids, args.layer, False)
        candidate_grads = capture_gradients(model, prefixes)
        # The repair is applied to the same frozen weights only after all
        # candidate gradients have left GPU memory.
        r_loss, r_trace = run_arm(model, ids, args.layer, True)
        if c_trace["selected_experts"] != r_trace["selected_experts"]:
            routing_hamming = sum(
                a != b
                for aa, bb in zip(c_trace["selected_experts"], r_trace["selected_experts"])
                for arow, brow in zip(aa, bb)
                for a, b in zip(arow, brow)
            )
        else:
            routing_hamming = 0
        stats, delta_vector = gradient_stats(candidate_grads, model, prefixes)
        delta_vectors.append(delta_vector)
        rows.append({
            "state_id": state.get("state_id", state.get("sequence_id")),
            "loss_candidate": float(c_loss.cpu()),
            "loss_repair": float(r_loss.cpu()),
            "loss_delta": float((c_loss - r_loss).cpu()),
            "routing_hamming": routing_hamming,
            "routing_weights_equal": c_trace["routing_weights_digest"] == r_trace["routing_weights_digest"],
            **stats,
        })
        del ids
        torch.cuda.empty_cache()
    geometry = {}
    if delta_vectors and all(v.numel() == delta_vectors[0].numel() for v in delta_vectors):
        matrix = torch.stack(delta_vectors).double()
        gram = matrix @ matrix.T
        total = matrix.sum(dim=0)
        energy = float(torch.diagonal(gram).sum())
        geometry = {
            "scope": args.scope,
            "parameter_prefixes": list(prefixes),
            "coordinate_count": int(matrix.shape[1]),
            "state_count": int(matrix.shape[0]),
            "cross_state_gram": gram.tolist(),
            "coherence_amplification": float(total.norm() / (energy ** 0.5 + 1e-30)),
            "mean_pair_dot": float((gram.sum() - torch.diagonal(gram).sum()) / max(matrix.shape[0] * (matrix.shape[0] - 1), 1)),
            "mean_vector_norm": float(total.norm() / matrix.shape[0]),
            "per_state_delta_norm_mean": float(matrix.norm(dim=1).mean()),
        }
    payload = {
        "schema": "kernel-analyzer-olmoe-router-accum-probe-v1",
        "status": "ENGINEERING_ONLY",
        "claim_boundary": "Two-state full eager F+B implementation probe; no bias or trajectory verdict.",
        "model": str(args.model.resolve()),
        "input_bank": str(args.input_bank.resolve()),
        "layer": args.layer,
        "state_start": args.start,
        "scope": args.scope,
        "candidate": "native BF16 index_add_ accumulation",
        "repair": "FP32 accumulation of identical BF16 expert summands, one final cast",
        "rows": rows,
        "formation_geometry": geometry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
