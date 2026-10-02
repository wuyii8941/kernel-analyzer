#!/usr/bin/env python3
"""Natural Qwen3 attention-contraction probe.

Compare native eager attention matmul contractions with an equivalent einsum
implementation on the same model, inputs, and weights.  This is a scoped
candidate for a contraction/reassociation source, not a claim that all
attention backends are biased.
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
    if x.numel() < 2:
        return [float(x.mean()), float(x.mean())]
    half = 1.96 * float(x.std(unbiased=True)) / (x.numel() ** 0.5)
    mean = float(x.mean())
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)


def adamw_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -learning_rate * grad / (grad.abs() + eps)


def einsum_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key_states) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.einsum("bhqk,bhkd->bhqd", weights, value_states)
    return output.transpose(1, 2).contiguous(), weights


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).train()
    model.config.use_cache = False
    target_module = model.model.layers[args.layer].self_attn
    target = target_module.q_proj.weight if args.parameter == "q_proj" else target_module.v_proj.weight
    original = modeling_qwen3.eager_attention_forward
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    try:
        for state in states:
            values = state.get("token_ids", state.get("input_ids"))
            ids = torch.tensor([values], dtype=torch.long, device=device)
            labels = ids.clone()
            model.zero_grad(set_to_none=True)
            modeling_qwen3.eager_attention_forward = original
            loss_c = model(input_ids=ids, labels=labels, use_cache=False).loss
            loss_c.backward()
            grad_c = target.grad.detach().float().cpu().clone()
            write_c = adamw_write(grad_c, args.learning_rate, args.eps)
            model.zero_grad(set_to_none=True)
            modeling_qwen3.eager_attention_forward = einsum_attention
            loss_r = model(input_ids=ids, labels=labels, use_cache=False).loss
            loss_r.backward()
            grad_r = target.grad.detach().float().cpu().clone()
            write_r = adamw_write(grad_r, args.learning_rate, args.eps)
            effect = write_c - write_r
            effects.append(effect)
            rows.append({
                "state_id": state.get("state_id", state.get("sequence_id")),
                "loss_difference": float((loss_c - loss_r).detach().cpu()),
                "gradient_effect_rms_over_reference": ratio(grad_c - grad_r, grad_r),
                "write_effect_rms_over_reference": ratio(effect, write_r),
                "write_aligned": float(torch.sum(effect * write_r)) / max(float(torch.sum(write_r * write_r)), 1e-30),
            })
            del ids, labels, loss_c, loss_r
            torch.cuda.empty_cache()
    finally:
        modeling_qwen3.eager_attention_forward = original
    split = len(effects) // 2
    direction = torch.stack([x.double() for x in effects[:split]]).mean(0)
    norm = float(torch.linalg.vector_norm(direction))
    projections = []
    if norm:
        direction /= norm
        projections = [float(torch.sum(x.double() * direction)) for x in effects[split:]]
    return {
        "schema": "kernel-analyzer-qwen3-attention-contraction-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model), "layer": args.layer, "parameter": args.parameter,
        "operator": "Qwen3 attention score/value contraction",
        "candidate": "native eager torch.matmul contractions",
        "reference": "same eager attention with equivalent torch.einsum contractions",
        "input_source": "declared real Qwen3 text input bank",
        "claim_boundary": "One Qwen3 checkpoint, one layer/parameter, and the declared bank; no population or loss-quality claim.",
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_attention_path": True, "single_changed_boundary": "attention contraction expression"},
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows)-split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows)/len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows)/len(rows),
            "write_aligned_interval_normal_95": interval([r["write_aligned"] for r in rows]),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "heldout_write_projection_interval_normal_95": interval(projections) if projections else None,
            "heldout_write_projection_positive": sum(x > 0 for x in projections),
            "heldout_write_projection_negative": sum(x < 0 for x in projections),
            "calibration_direction_norm": norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, default=Path("results/coverage/qwen_seq128_input_bank.json"))
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--parameter", choices=("q_proj", "v_proj"), default="q_proj")
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
