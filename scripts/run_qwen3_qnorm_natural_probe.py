#!/usr/bin/env python3
"""Natural Qwen3 attention Q/K-RMSNorm materialization probe.

The native Qwen3 attention q_norm/k_norm casts the normalized value to the
activation dtype before multiplying by the BF16 weight.  The paired reference
keeps that multiplication in FP32 and casts once at the end.  This is a
same-input model-level boundary probe; it is not a population or loss claim.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean().item())
        return [m, m]
    m = float(x.mean().item())
    h = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [m - h, m + h]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    den = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(den, 1e-30)


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


def install_fp32_weight_materialization(norm: torch.nn.Module) -> None:
    def reference_forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        value = hidden_states.float()
        variance = value.pow(2).mean(-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * normalized).to(input_dtype)

    norm.forward = types.MethodType(reference_forward, norm)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    if not states:
        raise ValueError("input bank contains no states")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).train()
    model.config.use_cache = False
    attention = model.model.layers[args.layer].self_attn
    norm = attention.q_norm if args.parameter == "q_norm" else attention.k_norm
    target = norm.weight
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    references: list[torch.Tensor] = []

    for state in states:
        values = state.get("token_ids", state.get("input_ids"))
        if values is None:
            raise KeyError("input bank state has neither token_ids nor input_ids")
        ids = torch.tensor([values], dtype=torch.long, device=device)
        labels = ids.clone()

        model.zero_grad(set_to_none=True)
        native = model(input_ids=ids, labels=labels, use_cache=False)
        native_loss = native.loss
        native_loss.backward()
        native_grad = target.grad.detach().float().cpu().clone()
        native_write = first_step_write(native_grad, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        original_forward = norm.forward
        install_fp32_weight_materialization(norm)
        try:
            reference = model(input_ids=ids, labels=labels, use_cache=False)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_grad = target.grad.detach().float().cpu().clone()
        finally:
            norm.forward = original_forward
        reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)
        grad_effect = native_grad - reference_grad
        write_effect = native_write - reference_write
        effects.append(write_effect)
        references.append(reference_write)
        rows.append(
            {
                "state_id": state.get("state_id", state.get("sequence_id")),
                "loss_difference": float((native_loss - reference_loss).detach().cpu().item()),
                "gradient_effect_rms_over_reference": ratio(native_grad - reference_grad, reference_grad),
                "write_effect_rms_over_reference": ratio(write_effect, reference_write),
                "gradient_aligned": float(torch.sum(grad_effect * reference_grad).item())
                / max(float(torch.sum(reference_grad * reference_grad).item()), 1e-30),
                "write_aligned": float(torch.sum(write_effect * reference_write).item())
                / max(float(torch.sum(reference_write * reference_write).item()), 1e-30),
            }
        )
        del ids, labels, native, reference
        torch.cuda.empty_cache()

    split = len(rows) // 2
    calibration = torch.stack([x.double() for x in effects[:split]])
    confirmation = torch.stack([x.double() for x in effects[split:]])
    mean_direction = calibration.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(mean_direction).item())
    projections: list[float] = []
    if direction_norm:
        unit = mean_direction / direction_norm
        projections = [float(torch.sum(x.double() * unit).item()) for x in effects[split:]]

    return {
        "schema": "kernel-analyzer-qwen3-qnorm-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "parameter": args.parameter,
        "operator": "Qwen3 attention Q/K RMSNorm materialization",
        "candidate": "native Qwen3RMSNorm: cast normalized value before BF16 weight multiply",
        "reference": "same RMSNorm formula with FP32 weight multiplication and one final cast",
        "input_source": "declared real Qwen3 text input bank",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "claim_boundary": "One Qwen3 checkpoint, one attention layer/parameter, and the declared text bank; not a population or loss-quality guarantee.",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_attention_path": True,
            "single_changed_boundary": "selected q_norm/k_norm materialization",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "gradient_aligned_interval_normal_95": interval([r["gradient_aligned"] for r in rows]),
            "write_aligned_interval_normal_95": interval([r["write_aligned"] for r in rows]),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "heldout_write_projection_interval_normal_95": interval(projections) if projections else None,
            "heldout_write_projection_positive": sum(x > 0 for x in projections),
            "heldout_write_projection_negative": sum(x < 0 for x in projections),
            "calibration_direction_norm": direction_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, default=Path("results/coverage/qwen_seq128_input_bank.json"))
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--parameter", choices=("q_norm", "k_norm"), default="q_norm")
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
