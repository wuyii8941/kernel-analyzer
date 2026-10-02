#!/usr/bin/env python3
"""Natural training probe for an OLMoE RMSNorm materialization choice.

It compares the model's native RMSNorm with a one-variable reference that
casts the normalized value before multiplying by the BF16 weight.  The comparison is a real loss,
gradient, and first-step AdamW-write probe; it is not a synthetic operator
test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/allenai/OLMoE-1B-7B-0125")
DEFAULT_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/olmoe_1b7b_text128.json"


def first_step_write(grad: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    # Zero-moment AdamW, zero weight decay.  This is the same declared
    # one-step endpoint used by the other natural probes.
    g = grad.float()
    return -learning_rate * g / (g.abs() + eps)


def install_reference(norm: torch.nn.Module, mode: str) -> None:
    def reference_forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        value = hidden_states.float()
        variance = value.pow(2).mean(-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.variance_epsilon)
        if mode == "cast_before_weight":
            return self.weight * normalized.to(input_dtype)
        if mode == "fp32_weight_multiply":
            return (self.weight.float() * normalized).to(input_dtype)
        raise ValueError(f"unknown reference mode: {mode}")

    norm.forward = types.MethodType(reference_forward, norm)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text())
    states = bank["states"][: args.states]
    if not states:
        raise ValueError("input bank contains no states")
    device = torch.device(args.device)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    layers = model.model.layers
    norm = getattr(layers[args.layer], args.norm_name)
    if not isinstance(norm, torch.nn.Module):
        raise TypeError(f"not a module: layer {args.layer} {args.norm_name}")
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
        install_reference(norm, args.reference_mode)
        try:
            reference = model(input_ids=ids, labels=labels, use_cache=False)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_grad = target.grad.detach().float().clone()
        finally:
            norm.forward = original_forward
        reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)

        def rel(value: torch.Tensor, base: torch.Tensor) -> float:
            den = float(torch.linalg.vector_norm(base).item())
            return float(torch.linalg.vector_norm(value).item()) / den if den else 0.0

        grad_delta = native_grad - reference_grad
        write_delta = native_write - reference_write
        gradient_deltas.append(grad_delta.detach().cpu())
        write_deltas.append(write_delta.detach().cpu())
        reference_gradients.append(reference_grad.detach().cpu())
        reference_writes.append(reference_write.detach().cpu())
        reference_write_norm = float(torch.linalg.vector_norm(reference_write).item())
        rows.append(
            {
                "state_id": state.get("state_id"),
                "loss_native": float(native_loss.item()),
                "loss_reference": float(reference_loss.item()),
                "loss_difference": float((native_loss - reference_loss).item()),
                "gradient_effect_rms_over_reference": rel(grad_delta, reference_grad),
                "write_effect_rms_over_reference": rel(write_delta, reference_write),
                "gradient_signed_mean": float(grad_delta.mean().item()),
                "write_signed_mean": float(write_delta.mean().item()),
                "gradient_dot_reference": float(torch.sum(grad_delta * reference_grad).item()),
                "write_dot_reference": float(torch.sum(write_delta * reference_write).item()),
                "reference_write_norm": reference_write_norm,
            }
        )

    def summarize(key: str, base_key: str) -> dict[str, Any]:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
        signs = {"positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "zero": int((values == 0).sum())}
        mean = float(values.mean().item())
        sd = float(values.std(unbiased=True).item()) if len(values) > 1 else 0.0
        half = 1.96 * sd / (len(values) ** 0.5) if len(values) > 1 else 0.0
        rms = torch.tensor([row[key.replace("gradient_effect_rms_over_reference", "gradient_effect_rms_over_reference").replace("write_effect_rms_over_reference", "write_effect_rms_over_reference")] for row in rows], dtype=torch.float64)
        return {
            "state_count": len(rows),
            "mean": mean,
            "normal_95_interval": [mean - half, mean + half],
            "signs": signs,
            "rms_mean": float(rms.mean().item()),
            "rms_max": float(rms.max().item()),
            "base_reference": base_key,
        }

    def direction_summary(deltas: list[torch.Tensor], repairs: list[torch.Tensor]) -> dict[str, Any]:
        midpoint = len(deltas) // 2
        calibration = torch.stack(deltas[:midpoint], dim=0).double()
        confirmation = torch.stack(deltas[midpoint:], dim=0).double()
        repair_confirmation = torch.stack(repairs[midpoint:], dim=0).double()
        mean_direction = calibration.mean(dim=0)
        direction_norm = float(torch.linalg.vector_norm(mean_direction).item())
        if direction_norm == 0.0 or confirmation.shape[0] == 0:
            return {"status": "UNIDENTIFIABLE", "calibration_count": midpoint, "confirmation_count": int(confirmation.shape[0])}
        unit = mean_direction / direction_norm
        projections = confirmation @ unit
        aligned = torch.stack([
            torch.dot(deltas[midpoint + i].double(), repairs[midpoint + i].double())
            / max(float(torch.dot(repairs[midpoint + i].double(), repairs[midpoint + i].double()).item()), 1e-30)
            for i in range(len(repairs) - midpoint)
        ])
        def interval(values: torch.Tensor) -> list[float]:
            mean = float(values.mean().item())
            sd = float(values.std(unbiased=True).item()) if values.numel() > 1 else 0.0
            half = 1.96 * sd / (values.numel() ** 0.5) if values.numel() > 1 else 0.0
            return [mean - half, mean + half]
        return {
            "status": "COMPLETE",
            "calibration_count": midpoint,
            "confirmation_count": int(confirmation.shape[0]),
            "calibration_direction_norm": direction_norm,
            "projection_mean": float(projections.mean().item()),
            "projection_interval_normal_95": interval(projections),
            "projection_signs": {
                "positive": int((projections > 0).sum()),
                "negative": int((projections < 0).sum()),
                "zero": int((projections == 0).sum()),
            },
            "aligned_mean": float(aligned.mean().item()),
            "aligned_interval_normal_95": interval(aligned),
        }

    result = {
        "schema": "kernel-analyzer-olmoe-rmsnorm-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "layer": args.layer,
        "norm_name": args.norm_name,
        "candidate": "native OLMoE RMSNorm forward",
        "reference": args.reference_mode,
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate, "epsilon": args.eps},
        "claim_boundary": "Natural OLMoE training inputs at one checkpoint and one RMSNorm; not a population or long-horizon guarantee.",
        "rows": rows,
        "summaries": {
            "gradient_effect_rms_over_reference": summarize("gradient_effect_rms_over_reference", "reference gradient"),
            "write_effect_rms_over_reference": summarize("write_effect_rms_over_reference", "reference first-step write"),
            "loss_difference": summarize("loss_difference", "reference loss"),
            "gradient_direction": direction_summary(gradient_deltas, reference_gradients),
            "write_direction": direction_summary(write_deltas, reference_writes),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--norm-name", default="input_layernorm")
    parser.add_argument("--reference-mode", choices=("cast_before_weight", "fp32_weight_multiply"), default="cast_before_weight")
    parser.add_argument("--states", type=int, default=26)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "states": len(result["rows"]), "summary": result["summaries"]}, sort_keys=True))


if __name__ == "__main__":
    main()
