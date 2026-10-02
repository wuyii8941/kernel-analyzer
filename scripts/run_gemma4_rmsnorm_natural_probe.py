#!/usr/bin/env python3
"""Natural Gemma-4 RMSNorm materialization probe.

The native Gemma-4 text RMSNorm keeps the normalized value and BF16 weight
multiplication in FP32 and casts once at the end.  The paired reference casts
the normalized value before multiplying by the BF16 weight.  The model, token
stream, and target parameter are otherwise unchanged.  This is a model-level
training probe, not a synthetic operator test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def first_step_write(grad: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = grad.float()
    return -learning_rate * g / (g.abs() + eps)


def install_cast_before_weight(norm: torch.nn.Module) -> None:
    def reference_forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        value = hidden_states.float()
        mean_squared = value.pow(2).mean(-1, keepdim=True) + self.eps
        normalized = value * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            normalized = normalized.to(input_dtype)
            normalized = normalized * self.weight
        return normalized.to(input_dtype)

    norm.forward = types.MethodType(reference_forward, norm)


def ratio(value: torch.Tensor, base: torch.Tensor) -> float:
    den = float(torch.linalg.vector_norm(base).item())
    return float(torch.linalg.vector_norm(value).item()) / den if den else 0.0


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if x.numel() <= 1:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [mean - half, mean + half]


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    if not states:
        raise ValueError("input bank contains no states")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    norm = model.model.language_model.layers[args.layer].input_layernorm
    target = norm.weight
    rows: list[dict[str, Any]] = []
    gradient_deltas: list[torch.Tensor] = []
    write_deltas: list[torch.Tensor] = []
    reference_gradients: list[torch.Tensor] = []
    reference_writes: list[torch.Tensor] = []
    for state in states:
        ids = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        labels = ids.clone()

        model.zero_grad(set_to_none=True)
        native = model(input_ids=ids, labels=labels, use_cache=False)
        native_loss = native.loss
        native_loss.backward()
        native_grad = target.grad.detach().float().clone()
        native_write = first_step_write(native_grad, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        original_forward = norm.forward
        install_cast_before_weight(norm)
        try:
            reference = model(input_ids=ids, labels=labels, use_cache=False)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_grad = target.grad.detach().float().clone()
        finally:
            norm.forward = original_forward
        reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)
        grad_delta = native_grad - reference_grad
        write_delta = native_write - reference_write
        gradient_deltas.append(grad_delta.detach().cpu())
        write_deltas.append(write_delta.detach().cpu())
        reference_gradients.append(reference_grad.detach().cpu())
        reference_writes.append(reference_write.detach().cpu())
        rows.append(
            {
                "state_id": state.get("state_id"),
                "loss_native": float(native_loss.item()),
                "loss_reference": float(reference_loss.item()),
                "loss_difference": float((native_loss - reference_loss).item()),
                "gradient_effect_rms_over_reference": ratio(grad_delta, reference_grad),
                "write_effect_rms_over_reference": ratio(write_delta, reference_write),
                "gradient_dot_reference": float(torch.sum(grad_delta * reference_grad).item()),
                "write_dot_reference": float(torch.sum(write_delta * reference_write).item()),
            }
        )

    def direction_summary(deltas: list[torch.Tensor], repairs: list[torch.Tensor]) -> dict[str, Any]:
        midpoint = len(deltas) // 2
        calibration = torch.stack(deltas[:midpoint], dim=0).double()
        confirmation = torch.stack(deltas[midpoint:], dim=0).double()
        if midpoint == 0 or confirmation.shape[0] == 0:
            return {"status": "INSUFFICIENT"}
        mean_direction = calibration.mean(dim=0)
        direction_norm = float(torch.linalg.vector_norm(mean_direction).item())
        if direction_norm == 0.0:
            return {"status": "NOT_IDENTIFIABLE_ZERO_CALIBRATION_MEAN"}
        unit = mean_direction / direction_norm
        projections = confirmation @ unit
        aligned = torch.tensor(
            [
                float(torch.dot(deltas[i].double(), repairs[i].double()).item())
                / max(float(torch.dot(repairs[i].double(), repairs[i].double()).item()), 1e-30)
                for i in range(midpoint, len(deltas))
            ],
            dtype=torch.float64,
        )
        return {
            "status": "DESCRIPTIVE_HELD_OUT",
            "calibration_count": midpoint,
            "confirmation_count": int(confirmation.shape[0]),
            "calibration_direction_norm": direction_norm,
            "projection_mean": float(projections.mean().item()),
            "projection_interval_normal_95": interval([float(x) for x in projections]),
            "projection_positive_count": int((projections > 0).sum().item()),
            "projection_negative_count": int((projections < 0).sum().item()),
            "aligned_mean": float(aligned.mean().item()),
            "aligned_interval_normal_95": interval([float(x) for x in aligned]),
        }

    return {
        "schema": "kernel-analyzer-gemma4-rmsnorm-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "layer": args.layer,
        "target": "model.language_model.layers[layer].input_layernorm.weight",
        "candidate": "native Gemma-4 RMSNorm: FP32 weight multiplication, one final cast",
        "reference": "cast normalized value to BF16 before BF16 weight multiplication",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate, "epsilon": args.eps},
        "claim_boundary": "Natural Gemma-4 text inputs at one checkpoint and one RMSNorm; not a population or long-horizon guarantee.",
        "rows": rows,
        "summaries": {
            "gradient_effect_rms_over_reference": {
                "mean": float(torch.tensor([r["gradient_effect_rms_over_reference"] for r in rows], dtype=torch.float64).mean().item()),
                "normal_95_interval": interval([r["gradient_effect_rms_over_reference"] for r in rows]),
            },
            "write_effect_rms_over_reference": {
                "mean": float(torch.tensor([r["write_effect_rms_over_reference"] for r in rows], dtype=torch.float64).mean().item()),
                "normal_95_interval": interval([r["write_effect_rms_over_reference"] for r in rows]),
            },
            "loss_difference": {
                "mean": float(torch.tensor([r["loss_difference"] for r in rows], dtype=torch.float64).mean().item()),
                "normal_95_interval": interval([r["loss_difference"] for r in rows]),
            },
            "gradient_direction": direction_summary(gradient_deltas, reference_gradients),
            "write_direction": direction_summary(write_deltas, reference_writes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument("--input-bank", type=Path, default=Path("results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "states": len(result["rows"]), "summary": result["summaries"]}, sort_keys=True))


if __name__ == "__main__":
    main()
