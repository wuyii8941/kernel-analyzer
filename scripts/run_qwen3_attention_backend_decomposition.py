#!/usr/bin/env python3
"""Decompose default-SDPA versus math-SDPA effects on one Qwen3 boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3 import modeling_qwen3


def write(grad: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = grad.float()
    return -lr * g / (g.abs() + eps)


def ratio(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item()) / max(
        float(torch.linalg.vector_norm(reference).item()), 1e-30
    )


def math_sdpa(*args, **kwargs):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return sdpa_attention_forward(*args, **kwargs)


def run(args: argparse.Namespace) -> dict:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).train()
    model.config.use_cache = False
    attention = model.model.layers[args.layer].self_attn
    target = attention.q_proj.weight if args.parameter == "q_proj" else attention.v_proj.weight
    original_eager = modeling_qwen3.eager_attention_forward
    rows = []
    effects: dict[str, list[torch.Tensor]] = {"default_sdpa": [], "math_sdpa": []}
    try:
        for state in states:
            token_ids = state.get("token_ids", state.get("input_ids"))
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            labels = ids.clone()

            modeling_qwen3.eager_attention_forward = original_eager
            model.zero_grad(set_to_none=True)
            eager_loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            eager_loss.backward()
            eager_grad = target.grad.detach().float().cpu().clone()
            eager_write = write(eager_grad, args.learning_rate, args.eps)

            state_row = {"state_id": state.get("state_id", state.get("sequence_id")), "variants": {}}
            for name, fn in (("default_sdpa", sdpa_attention_forward), ("math_sdpa", math_sdpa)):
                modeling_qwen3.eager_attention_forward = fn
                model.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, labels=labels, use_cache=False).loss
                loss.backward()
                grad = target.grad.detach().float().cpu().clone()
                current_write = write(grad, args.learning_rate, args.eps)
                effect = eager_write - current_write
                effects[name].append(effect.double().reshape(-1))
                state_row["variants"][name] = {
                    "loss_difference_from_eager": float((eager_loss - loss).detach().cpu()),
                    "write_effect_rms_over_variant": ratio(effect, current_write),
                    "write_aligned_over_variant": float(torch.sum(effect * current_write).item()) / max(float(torch.sum(current_write * current_write).item()), 1e-30),
                }
            rows.append(state_row)
            del ids, labels, eager_loss
            torch.cuda.empty_cache()
    finally:
        modeling_qwen3.eager_attention_forward = original_eager

    comparisons = []
    for default_effect, math_effect in zip(effects["default_sdpa"], effects["math_sdpa"]):
        denom = max(float(torch.linalg.vector_norm(default_effect).item()), 1e-30)
        comparisons.append({
            "math_minus_default_rms_over_default": float(torch.linalg.vector_norm(math_effect - default_effect).item()) / denom,
            "effect_cosine": float(torch.dot(default_effect, math_effect).item()) / max(float(torch.linalg.vector_norm(default_effect).item()) * float(torch.linalg.vector_norm(math_effect).item()), 1e-30),
        })
    return {
        "schema": "kernel-analyzer-qwen3-attention-backend-decomposition-v1",
        "status": "COMPLETE",
        "model": str(args.model), "layer": args.layer, "parameter": args.parameter,
        "candidate": "native eager attention",
        "references": ["default SDPA", "math-backend SDPA"],
        "input_source": str(args.input_bank),
        "comparison_scope": {
            "same_model_weights": True, "same_input_ids": True,
            "same_eager_baseline": True,
            "single_changed_boundary": "SDPA backend implementation: default versus forced math backend",
        },
        "rows": rows,
        "backend_effect_comparison": comparisons,
        "summary": {
            "state_count": len(rows),
            "default_sdpa_write_effect_rms_mean": sum(r["variants"]["default_sdpa"]["write_effect_rms_over_variant"] for r in rows) / len(rows),
            "math_sdpa_write_effect_rms_mean": sum(r["variants"]["math_sdpa"]["write_effect_rms_over_variant"] for r in rows) / len(rows),
            "math_minus_default_rms_over_default_mean": sum(r["math_minus_default_rms_over_default"] for r in comparisons) / len(comparisons),
            "effect_cosine_mean": sum(r["effect_cosine"] for r in comparisons) / len(comparisons),
            "effect_cosine_min": min(r["effect_cosine"] for r in comparisons),
        },
        "claim_boundary": "One Qwen3 checkpoint/layer and declared text bank; backend-level source decomposition, not a population or loss-quality claim.",
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
