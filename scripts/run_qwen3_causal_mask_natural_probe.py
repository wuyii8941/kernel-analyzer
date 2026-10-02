#!/usr/bin/env python3
"""Natural Qwen3 implicit-versus-explicit causal-mask probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if x.numel() < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    den = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(den, 1e-30)


def adamw_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -learning_rate * grad / (grad.abs() + eps)


def explicit_causal_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, 1, length, length), device=device, dtype=dtype)
    return mask.masked_fill(torch.triu(torch.ones((length, length), device=device, dtype=torch.bool), diagonal=1), float("-inf"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).train()
    model.config.use_cache = False
    target_module = model.model.layers[args.layer].self_attn
    target = target_module.q_proj.weight if args.parameter == "q_proj" else target_module.k_proj.weight
    rows: list[dict[str, Any]] = []
    gradients: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    ref_writes: list[torch.Tensor] = []
    for state in states:
        values = state.get("token_ids", state.get("input_ids"))
        if values is None:
            raise KeyError("input bank state has neither token_ids nor input_ids")
        ids = torch.tensor([values], dtype=torch.long, device=device)
        labels = ids.clone()
        model.zero_grad(set_to_none=True)
        candidate = model(input_ids=ids, labels=labels, use_cache=False)
        candidate_loss = candidate.loss
        candidate_loss.backward()
        candidate_grad = target.grad.detach().float().cpu().clone()
        candidate_write = adamw_write(candidate_grad, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        mask = explicit_causal_mask(ids.shape[-1], device, torch.bfloat16)
        reference = model(input_ids=ids, labels=labels, attention_mask=mask, use_cache=False)
        reference_loss = reference.loss
        reference_loss.backward()
        reference_grad = target.grad.detach().float().cpu().clone()
        reference_write = adamw_write(reference_grad, args.learning_rate, args.eps)

        grad_effect = candidate_grad - reference_grad
        write_effect = candidate_write - reference_write
        gradients.append(grad_effect)
        writes.append(write_effect)
        ref_writes.append(reference_write)
        rows.append({
            "state_id": state.get("state_id", state.get("sequence_id")),
            "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
            "gradient_effect_rms_over_reference": ratio(grad_effect, reference_grad),
            "write_effect_rms_over_reference": ratio(write_effect, reference_write),
            "write_aligned": float(torch.sum(write_effect * reference_write).item()) / max(float(torch.sum(reference_write * reference_write).item()), 1e-30),
        })
        del ids, labels, candidate, reference, mask
        torch.cuda.empty_cache()

    split = len(rows) // 2
    calibration = torch.stack([x.double() for x in writes[:split]])
    direction = calibration.mean(dim=0)
    norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if norm:
        direction = direction / norm
        projections = [float(torch.sum(x.double() * direction).item()) for x in writes[split:]]
    return {
        "schema": "kernel-analyzer-qwen3-causal-mask-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "parameter": args.parameter,
        "operator": "Qwen3 causal-mask materialization",
        "candidate": "native implicit causal mask",
        "reference": "same model with explicit additive upper-triangular mask",
        "input_source": "declared real Qwen3 text input bank",
        "claim_boundary": "One Qwen3 checkpoint, one attention layer/parameter, and the declared text bank; not a population or loss-quality guarantee.",
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_attention_path": True, "single_changed_boundary": "causal mask representation"},
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_aligned_interval_normal_95": interval([r["write_aligned"] for r in rows]),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "heldout_write_projection_interval_normal_95": interval(projections) if projections else None,
            "heldout_write_projection_positive": sum(x > 0 for x in projections),
            "heldout_write_projection_negative": sum(x < 0 for x in projections),
            "calibration_direction_norm": norm,
            "exact_all_states": all(r["gradient_effect_rms_over_reference"] == 0.0 and r["write_effect_rms_over_reference"] == 0.0 and r["loss_difference"] == 0.0 for r in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, default=Path("results/coverage/qwen_seq128_input_bank.json"))
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--parameter", choices=("q_proj", "k_proj"), default="q_proj")
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
