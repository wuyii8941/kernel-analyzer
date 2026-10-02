#!/usr/bin/env python3
"""Natural OLMoE router-score materialization probe.

The expert set and expert computations are held fixed.  Only the router
linear score is evaluated in native dtype versus FP32, then the native top-k
indices are reused in the reference branch.  This separates router-score
materialization from selection and expert-contribution effects.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if len(values) < 2:
        return [float(x.mean()), float(x.mean())]
    m = float(x.mean())
    h = 1.96 * float(x.std(unbiased=True)) / math.sqrt(len(values))
    return [m - h, m + h]


def windows(tokenizer: Any, source: Path, sequence_length: int, count: int) -> list[torch.Tensor]:
    ids = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = max(1, sequence_length)
    need = (count - 1) * stride + sequence_length + 1
    if ids.numel() < need:
        raise RuntimeError("text source has {} tokens, need {}".format(ids.numel(), need))
    return [ids[i * stride : i * stride + sequence_length].clone() for i in range(count)]


def make_forward(block, fp32_router: bool, frozen_indices: torch.Tensor | None):
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        if fp32_router:
            router_logits = F.linear(flat.float(), self.gate.weight.float())
        else:
            router_logits = self.gate(flat)
        routing_scores = F.softmax(router_logits, dim=1, dtype=torch.float)
        if frozen_indices is None:
            routing_weights, selected_experts = torch.topk(routing_scores, self.top_k, dim=-1)
        else:
            selected_experts = frozen_indices.to(routing_scores.device)
            routing_weights = routing_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=flat.dtype, device=flat.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = flat[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(flat.dtype))
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

    return types.MethodType(forward, block)


def ratio(value: torch.Tensor, base: torch.Tensor) -> float:
    return float(value.norm()) / max(float(base.norm()), 1e-30)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/data1/tzh/models/allenai/OLMoE-1B-7B-0125"))
    p.add_argument("--text-source", type=Path, default=ROOT / "docs/root_cause_closure_current.md")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    block = model.model.layers[args.layer].mlp
    target = block.gate.weight
    native_forward = block.forward
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(device)
        labels = ids.clone()
        block.forward = native_forward
        # Capture the native selected set from the *actual* hidden state that
        # enters this MoE block.  Embedding output is not a valid substitute:
        # attention and the preceding norm already changed it.
        captured: dict[str, torch.Tensor] = {}

        def capture_selection(module, inputs):
            hidden = inputs[0]
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            native_logits = module.gate(flat_hidden)
            captured["selected"] = torch.topk(
                F.softmax(native_logits, dim=-1, dtype=torch.float), module.top_k, dim=-1
            ).indices.detach()

        hook = block.register_forward_pre_hook(capture_selection)
        model.zero_grad(set_to_none=True)
        native_output = model(input_ids=ids, labels=labels, use_cache=False)
        hook.remove()
        native_loss = native_output.loss
        native_loss.backward()
        native_grad = target.grad.detach().float().cpu().clone()
        native_selected = captured["selected"]

        block.forward = make_forward(block, True, native_selected)
        model.zero_grad(set_to_none=True)
        reference_output = model(input_ids=ids, labels=labels, use_cache=False)
        reference_loss = reference_output.loss
        reference_loss.backward()
        reference_grad = target.grad.detach().float().cpu().clone()
        block.forward = native_forward
        effect = reference_grad - native_grad
        native_write = -args.learning_rate * native_grad
        reference_write = -args.learning_rate * reference_grad
        write_effect = reference_write - native_write
        aligned = float(torch.dot(write_effect.reshape(-1), native_write.reshape(-1))) / max(
            float(torch.dot(native_write.reshape(-1), native_write.reshape(-1))), 1e-30
        )
        effects.append(effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        references.append(native_write.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "native_loss": float(native_loss.detach().cpu()),
            "fp32_router_loss": float(reference_loss.detach().cpu()),
            "loss_difference_fp32_minus_native": float((reference_loss - native_loss).detach().cpu()),
            "gradient_effect_rms_over_native": ratio(effect, native_grad),
            "write_effect_rms_over_native": ratio(write_effect, native_write),
            "write_aligned_fp32_minus_native": aligned,
            "selection_frozen": True,
        })
    split = args.states // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm())
    projections: list[float] = []
    if direction_norm:
        unit = direction / direction_norm
        projections = [float(torch.dot(value, unit)) for value in effects[split:]]
    aligned = [row["write_aligned_fp32_minus_native"] for row in rows[split:]]
    result = {
        "schema": "kernel-analyzer-olmoe-router-score-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "OLMoE router score projection materialization",
        "parameter": "model.layers.{}.mlp.gate.weight".format(args.layer),
        "candidate": "native OLMoE router score projection",
        "reference": "same router score projection in FP32 with native top-k indices frozen",
        "layer": args.layer,
        "input_source": "real document-derived token windows retokenized with the OLMoE tokenizer",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_expert_set": True,
            "same_expert_computation": True,
            "single_changed_boundary": "router score linear projection",
        },
        "claim_boundary": "One OLMoE checkpoint, one router layer and declared natural text bank; fixed-selection source and update evidence only.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "confirmation_aligned_write_mean": sum(aligned) / len(aligned),
            "confirmation_aligned_write_interval_normal_95": interval(aligned),
            "confirmation_projection_interval_normal_95": interval(projections),
            "confirmation_projection_positive": sum(x > 0 for x in projections),
            "confirmation_projection_negative": sum(x < 0 for x in projections),
            "loss_difference_interval_normal_95": interval([r["loss_difference_fp32_minus_native"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
