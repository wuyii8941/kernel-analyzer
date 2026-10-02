#!/usr/bin/env python3
"""Isolate Granite MoE top-k routing-weight materialization on real text.

The router logits, top-k indices, expert inputs, expert computations and
expert output accumulation are held fixed.  The only intervention is whether
the FP32 top-k softmax weights are cast to the model dtype before multiplying
expert outputs, or retained in FP32 for that multiplication and cast once
before the unchanged repeated-destination write.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _windows(tokenizer: Any, source: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    ids = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = 2 * seq_len
    need = count * stride + seq_len
    if ids.numel() < need:
        raise RuntimeError(f"text source has {ids.numel()} tokens, need {need}")
    return [ids[i * stride : i * stride + seq_len].clone() for i in range(count)]


def _interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if len(values) < 2:
        return [float(x.mean()), float(x.mean())]
    mean = float(x.mean())
    half = 1.96 * float(x.std(unbiased=True)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def _relative(delta: torch.Tensor, reference: torch.Tensor) -> float:
    return float(delta.norm()) / max(float(reference.norm()), 1e-30)


def _make_forward(moe: torch.nn.Module, retain_fp32_gate: bool):
    router = moe.router

    def forward(self: torch.nn.Module, layer_input: torch.Tensor):
        bsz, length, emb_size = layer_input.size()
        flat = layer_input.reshape(-1, emb_size)

        # This is the native Granite router score path.  Both arms use the
        # same logits and exact same top-k indices; only gate-weight storage
        # and the following multiplication differ.
        logits = router.layer(flat).float()
        top_k_logits, top_k_indices = logits.topk(router.top_k, dim=1)
        fp32_weights = torch.softmax(top_k_logits, dim=1)
        gate_weights = fp32_weights if retain_fp32_gate else fp32_weights.to(flat.dtype)

        zeros = torch.zeros(
            [gate_weights.size(0), router.num_experts],
            dtype=gate_weights.dtype,
            device=gate_weights.device,
        )
        gates = zeros.scatter(1, top_k_indices, 1)
        expert_size = gates.long().sum(0).tolist()
        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(router.top_k, rounding_mode="trunc")
        batch_gates = gate_weights.flatten()[index_sorted_experts]

        expert_inputs = flat[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        first, second = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(first) * second
        expert_outputs = self.output_linear(hidden_states, expert_size)
        if retain_fp32_gate:
            expert_outputs = (expert_outputs.float() * batch_gates[:, None]).to(expert_outputs.dtype)
        else:
            expert_outputs = expert_outputs * batch_gates[:, None]

        output = torch.zeros(
            (bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device
        )
        output.index_add_(0, batch_index, expert_outputs)
        return output.view(bsz, length, self.input_size), logits

    return types.MethodType(forward, moe)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = _windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False
    moe = model.model.layers[args.layer].block_sparse_moe
    target = next(moe.output_linear.parameters())
    router_target = moe.router.layer.weight
    native_forward = moe.forward
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    router_effects: list[torch.Tensor] = []
    router_writes: list[torch.Tensor] = []
    router_references: list[torch.Tensor] = []
    try:
        for state_id, tokens in enumerate(bank):
            ids = tokens.unsqueeze(0).to(device)
            moe.forward = native_forward
            model.zero_grad(set_to_none=True)
            native = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
            native_loss = native.loss
            native_loss.backward()
            native_grad = target.grad.detach().float().cpu().clone()
            native_router_grad = router_target.grad.detach().float().cpu().clone()

            moe.forward = _make_forward(moe, retain_fp32_gate=True)
            model.zero_grad(set_to_none=True)
            reference = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_grad = target.grad.detach().float().cpu().clone()
            reference_router_grad = router_target.grad.detach().float().cpu().clone()

            effect = reference_grad - native_grad
            native_write = -args.learning_rate * native_grad
            reference_write = -args.learning_rate * reference_grad
            write_effect = reference_write - native_write
            aligned = float(torch.dot(write_effect.flatten(), native_write.flatten())) / max(
                float(torch.dot(native_write.flatten(), native_write.flatten())), 1e-30
            )
            effects.append(effect.double().flatten())
            writes.append(write_effect.double().flatten())
            references.append(native_write.double().flatten())
            router_effect = reference_router_grad - native_router_grad
            native_router_write = -args.learning_rate * native_router_grad
            reference_router_write = -args.learning_rate * reference_router_grad
            router_write_effect = reference_router_write - native_router_write
            router_effects.append(router_effect.double().flatten())
            router_writes.append(router_write_effect.double().flatten())
            router_references.append(native_router_write.double().flatten())
            rows.append({
                "state_id": state_id,
                "native_loss": float(native_loss.detach().cpu()),
                "fp32_gate_loss": float(reference_loss.detach().cpu()),
                "loss_difference_fp32_gate_minus_native": float((reference_loss - native_loss).detach().cpu()),
                "gradient_effect_rms_over_native": _relative(effect, native_grad),
                "write_effect_rms_over_native": _relative(write_effect, native_write),
                "write_aligned_fp32_gate_minus_native": aligned,
                "router_gradient_effect_rms_over_native": _relative(router_effect, native_router_grad),
                "router_write_effect_rms_over_native": _relative(router_write_effect, native_router_write),
                "router_write_aligned_fp32_gate_minus_native": float(
                    torch.dot(router_write_effect.flatten(), native_router_write.flatten())
                ) / max(float(torch.dot(native_router_write.flatten(), native_router_write.flatten())), 1e-30),
            })
    finally:
        moe.forward = native_forward

    split = args.states // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm())
    projections: list[float] = []
    if direction_norm > 0:
        unit = direction / direction_norm
        projections = [float(torch.dot(value, unit)) for value in effects[split:]]
    aligned_values = [row["write_aligned_fp32_gate_minus_native"] for row in rows[split:]]
    router_direction = torch.stack(router_effects[:split]).mean(dim=0)
    router_direction_norm = float(router_direction.norm())
    router_projections: list[float] = []
    if router_direction_norm > 0:
        router_unit = router_direction / router_direction_norm
        router_projections = [float(torch.dot(value, router_unit)) for value in router_effects[split:]]
    router_aligned_values = [row["router_write_aligned_fp32_gate_minus_native"] for row in rows[split:]]
    return {
        "schema": "kernel-analyzer-granite-router-gate-materialization-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "parameter": f"model.layers.{args.layer}.block_sparse_moe.output_linear.weight",
        "operator": "Granite MoE top-k routing-weight materialization",
        "candidate": "FP32 top-k softmax weights cast to BF16 before expert-output multiplication",
        "reference": "same logits, top-k indices and expert computations with FP32 gate multiplication then BF16 write-back",
        "input_source": str(args.text_source),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_router_logits": True,
            "same_top_k_indices": True,
            "same_expert_computation": True,
            "same_repeated_destination_accumulation": True,
            "single_changed_boundary": "top-k routing-weight cast and multiplication",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "router_gradient_effect_rms_mean": sum(r["router_gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "router_write_effect_rms_mean": sum(r["router_write_effect_rms_over_native"] for r in rows) / len(rows),
            "confirmation_aligned_write_mean": sum(aligned_values) / len(aligned_values),
            "confirmation_aligned_write_interval_normal_95": _interval(aligned_values),
            "confirmation_projection_interval_normal_95": _interval(projections),
            "confirmation_projection_positive": sum(x > 0 for x in projections),
            "confirmation_projection_negative": sum(x < 0 for x in projections),
            "router_confirmation_aligned_write_mean": sum(router_aligned_values) / len(router_aligned_values),
            "router_confirmation_aligned_write_interval_normal_95": _interval(router_aligned_values),
            "router_confirmation_projection_interval_normal_95": _interval(router_projections),
            "router_confirmation_projection_positive": sum(x > 0 for x in router_projections),
            "router_confirmation_projection_negative": sum(x < 0 for x in router_projections),
            "router_calibration_direction_norm": router_direction_norm,
            "loss_difference_interval_normal_95": _interval([r["loss_difference_fp32_gate_minus_native"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
        "claim_boundary": "One Granite checkpoint, one layer and the declared natural text bank; source and fixed-suite update evidence only. This is not a population or long-run loss claim.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base"))
    p.add_argument("--text-source", type=Path, default=Path("docs/root_cause_closure_current.md"))
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
