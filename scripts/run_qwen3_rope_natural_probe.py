#!/usr/bin/env python3
"""Natural Qwen3 RoPE materialization probe.

The native Qwen3 rotary helper is compared with the same formula evaluated in
explicit FP32 before the native-dtype write-back.  The model, tokens, and
attention path are otherwise unchanged.  This is a scoped natural training
probe, not a universal RoPE or loss claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3 import modeling_qwen3


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


def explicit_fp32_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    cos32 = cos.float().unsqueeze(unsqueeze_dim)
    sin32 = sin.float().unsqueeze(unsqueeze_dim)
    q32 = q.float()
    k32 = k.float()
    half = q32.shape[-1] // 2
    qrot = torch.cat((-q32[..., half:], q32[..., :half]), dim=-1)
    krot = torch.cat((-k32[..., half:], k32[..., :half]), dim=-1)
    return ((q32 * cos32) + (qrot * sin32)).to(q.dtype), ((k32 * cos32) + (krot * sin32)).to(k.dtype)


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
    original_rope = modeling_qwen3.apply_rotary_pos_emb
    deltas_gradient: list[torch.Tensor] = []
    deltas_write: list[torch.Tensor] = []
    refs_gradient: list[torch.Tensor] = []
    refs_write: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []

    try:
        for state in states:
            values = state.get("token_ids", state.get("input_ids"))
            if values is None:
                raise KeyError("input bank state has neither token_ids nor input_ids")
            input_ids = torch.tensor([values], dtype=torch.long, device=device)
            labels = input_ids.clone()

            model.zero_grad(set_to_none=True)
            modeling_qwen3.apply_rotary_pos_emb = original_rope
            candidate = model(input_ids=input_ids, labels=labels, use_cache=False)
            candidate_loss = candidate.loss
            candidate_loss.backward()
            candidate_gradient = target.grad.detach().float().cpu().clone()
            candidate_write = adamw_write(candidate_gradient, args.learning_rate, args.eps)

            model.zero_grad(set_to_none=True)
            modeling_qwen3.apply_rotary_pos_emb = explicit_fp32_rope
            reference = model(input_ids=input_ids, labels=labels, use_cache=False)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_gradient = target.grad.detach().float().cpu().clone()
            reference_write = adamw_write(reference_gradient, args.learning_rate, args.eps)

            gradient_effect = candidate_gradient - reference_gradient
            write_effect = candidate_write - reference_write
            deltas_gradient.append(gradient_effect)
            deltas_write.append(write_effect)
            refs_gradient.append(reference_gradient)
            refs_write.append(reference_write)
            rows.append({
                "state_id": state.get("state_id", state.get("sequence_id")),
                "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
                "gradient_effect_rms_over_reference": ratio(gradient_effect, reference_gradient),
                "write_effect_rms_over_reference": ratio(write_effect, reference_write),
                "gradient_aligned": float(torch.sum(gradient_effect * reference_gradient).item()) / max(float(torch.sum(reference_gradient * reference_gradient).item()), 1e-30),
                "write_aligned": float(torch.sum(write_effect * reference_write).item()) / max(float(torch.sum(reference_write * reference_write).item()), 1e-30),
            })
            del input_ids, labels, candidate, reference
            torch.cuda.empty_cache()
    finally:
        modeling_qwen3.apply_rotary_pos_emb = original_rope

    split = len(rows) // 2
    cal = torch.stack([x.double() for x in deltas_write[:split]])
    mean_direction = cal.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(mean_direction).item())
    if direction_norm:
        unit = mean_direction / direction_norm
        projections = [float(torch.sum(x.double() * unit).item()) for x in deltas_write[split:]]
    else:
        projections = []

    return {
        "schema": "kernel-analyzer-qwen3-rope-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "parameter": args.parameter,
        "operator": "Qwen3 rotary position embedding materialization",
        "candidate": "native modeling_qwen3.apply_rotary_pos_emb",
        "reference": "same formula with explicit FP32 multiply and native-dtype final write",
        "input_source": "declared real Qwen3 text input bank",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "claim_boundary": "One Qwen3 checkpoint, one attention layer/parameter, and the declared text bank; not a population or loss-quality guarantee.",
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_attention_path": True, "single_changed_boundary": "rotary helper"},
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
    parser.add_argument("--parameter", choices=("q_proj", "k_proj"), default="k_proj")
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
