#!/usr/bin/env python3
"""Probe the real GQA repeat-kv backward boundary on a DeepSeek/Qwen3 model.

The candidate uses the model's native ``repeat_kv`` implementation.  The
reference keeps the same model, inputs, weights, and attention path but
replaces only the grouped-head expansion with ``repeat_interleave``.  This is
intended to isolate the backward reduction/materialisation boundary that
appears in the existing layout-region records.

The result is scoped to the selected checkpoint, layer, parameter, and input
bank.  It is not a population or model-wide claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
DEFAULT_BANK = ROOT / "results/coverage/deepseek8b_seq256_input_bank.json"


def normal_interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (len(values) ** 0.5)
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(denominator, 1e-30)


def adamw_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -learning_rate * grad / (grad.abs() + eps)


def explicit_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    layer = model.model.layers[args.layer]
    target = layer.self_attn.k_proj.weight if args.parameter == "k_proj" else layer.self_attn.v_proj.weight
    original_repeat = modeling_qwen3.repeat_kv
    rows: list[dict[str, Any]] = []
    gradient_effects: list[torch.Tensor] = []
    write_effects: list[torch.Tensor] = []
    write_references: list[torch.Tensor] = []

    try:
        for state in states:
            token_values = state.get("token_ids", state.get("input_ids"))
            if token_values is None:
                raise KeyError("input bank state has neither token_ids nor input_ids")
            input_ids = torch.tensor([token_values], dtype=torch.long, device=device)
            labels = input_ids.clone()

            model.zero_grad(set_to_none=True)
            modeling_qwen3.repeat_kv = original_repeat
            candidate_loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            candidate_loss.backward()
            candidate_gradient = target.grad.detach().float().cpu().clone()
            candidate_write = adamw_write(candidate_gradient, args.learning_rate, args.eps)

            model.zero_grad(set_to_none=True)
            modeling_qwen3.repeat_kv = explicit_repeat_kv
            reference_loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            reference_loss.backward()
            reference_gradient = target.grad.detach().float().cpu().clone()
            reference_write = adamw_write(reference_gradient, args.learning_rate, args.eps)

            gradient_effect = candidate_gradient - reference_gradient
            write_effect = candidate_write - reference_write
            gradient_effects.append(gradient_effect)
            write_effects.append(write_effect)
            write_references.append(reference_write)
            rows.append({
                "state_id": state.get("state_id", state.get("sequence_id")),
                "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
                "gradient_effect_rms_over_reference": ratio(gradient_effect, reference_gradient),
                "write_effect_rms_over_reference": ratio(write_effect, reference_write),
                "gradient_signed_mean": float(gradient_effect.mean().item()),
                "write_signed_mean": float(write_effect.mean().item()),
            })
            del input_ids, labels, candidate_loss, reference_loss
            torch.cuda.empty_cache()
    finally:
        modeling_qwen3.repeat_kv = original_repeat

    split = len(rows) // 2
    calibration = torch.stack([x.double() for x in gradient_effects[:split]])
    direction = calibration.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if direction_norm > 0:
        direction = direction / direction_norm
        projections = [float(torch.sum(x.double() * direction).item()) for x in gradient_effects[split:]]
    aligned: list[float] = []
    for effect, reference in zip(write_effects[split:], write_references[split:]):
        effect_flat = effect.double().reshape(-1)
        reference_flat = reference.double().reshape(-1)
        denominator = float(torch.dot(reference_flat, reference_flat).item())
        aligned.append(float(torch.dot(effect_flat, reference_flat).item()) / max(denominator, 1e-30))

    return {
        "schema": "kernel-analyzer-deepseek-gqa-repeat-kv-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "parameter": args.parameter,
        "operator": "Qwen3 grouped-key/value head repeat backward",
        "candidate": "native modeling_qwen3.repeat_kv (expand + reshape)",
        "reference": "same model/input with repeat_interleave head expansion",
        "input_source": "declared real DeepSeek/Qwen3 text input bank",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "claim_boundary": "One checkpoint, one attention layer/parameter, and the declared text bank; not a population or loss-quality guarantee.",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_attention_path": True,
            "single_changed_boundary": "repeat_kv implementation",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": normal_interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections),
            "projection_negative": sum(x < 0 for x in projections),
            "aligned_write_mean": sum(aligned) / len(aligned) if aligned else None,
            "aligned_write_interval_normal_95": normal_interval(aligned) if aligned else None,
            "loss_difference_interval_normal_95": normal_interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--parameter", choices=("k_proj", "v_proj"), default="k_proj")
    parser.add_argument("--states", type=int, default=8)
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
