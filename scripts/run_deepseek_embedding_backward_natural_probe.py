#!/usr/bin/env python3
"""Probe native embedding backward accumulation on a real DeepSeek/Qwen3 run.

The forward values are identical.  The reference uses a small custom
autograd function whose embedding-weight gradient is accumulated in FP32
before being cast back to the BF16 parameter dtype.  This isolates repeated
token accumulation from the fused NLL region and records the resulting
parameter-write effect on the real model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
DEFAULT_BANK = ROOT / "results/coverage/deepseek8b_seq256_input_bank.json"


class Fp32Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        return F.embedding(input_ids, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_ids,) = ctx.saved_tensors
        grad_weight = torch.zeros(ctx.weight_shape, device=grad_output.device, dtype=torch.float32)
        grad_weight.index_add_(0, input_ids.reshape(-1), grad_output.reshape(-1, grad_output.shape[-1]).float())
        return grad_weight.to(ctx.weight_dtype), None


def fp32_embedding_forward(module: torch.nn.Embedding, input_ids: torch.Tensor) -> torch.Tensor:
    return Fp32Embedding.apply(module.weight, input_ids)


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (len(values) ** 0.5)
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def adamw_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -learning_rate * grad / (grad.abs() + eps)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    embedding = model.model.embed_tokens
    original_forward = embedding.forward
    target = embedding.weight
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []

    try:
        for state in states:
            # The historical DeepSeek bank uses ``token_ids`` while the
            # Qwen held-out bank stores the same declared sequence as
            # ``input_ids``.  Accept both representations; this probe is
            # about the embedding backward boundary, not the bank schema.
            token_ids = state.get("token_ids", state.get("input_ids"))
            if token_ids is None:
                raise KeyError("state must contain token_ids or input_ids")
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            labels = input_ids.clone()

            model.zero_grad(set_to_none=True)
            embedding.forward = original_forward
            candidate_loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            candidate_loss.backward()
            candidate_gradient = target.grad.detach().float().cpu().clone()
            candidate_write = adamw_write(candidate_gradient, args.learning_rate, args.eps)

            model.zero_grad(set_to_none=True)
            embedding.forward = lambda ids, _module=embedding: fp32_embedding_forward(_module, ids)
            reference_loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            reference_loss.backward()
            reference_gradient = target.grad.detach().float().cpu().clone()
            reference_write = adamw_write(reference_gradient, args.learning_rate, args.eps)

            effect = candidate_gradient - reference_gradient
            write_effect = candidate_write - reference_write
            effects.append(effect)
            writes.append(write_effect)
            references.append(reference_write)
            rows.append({
                "state_id": state.get("state_id"),
                "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
                "gradient_effect_rms_over_reference": ratio(effect, reference_gradient),
                "write_effect_rms_over_reference": ratio(write_effect, reference_write),
                "gradient_signed_mean": float(effect.mean().item()),
                "write_signed_mean": float(write_effect.mean().item()),
            })
            del input_ids, labels, candidate_loss, reference_loss
            torch.cuda.empty_cache()
    finally:
        embedding.forward = original_forward

    split = len(rows) // 2
    direction = torch.stack([x.double() for x in effects[:split]]).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(torch.sum(x.double() * direction).item()) for x in effects[split:]]
    aligned: list[float] = []
    for effect, reference in zip(writes[split:], references[split:]):
        e, r = effect.double().reshape(-1), reference.double().reshape(-1)
        aligned.append(float(torch.dot(e, r).item()) / max(float(torch.dot(r, r).item()), 1e-30))

    return {
        "schema": "kernel-analyzer-deepseek-embedding-backward-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "embedding backward repeated-token accumulation",
        "candidate": "native Qwen3 embedding backward",
        "reference": "same embedding forward with FP32 index-add gradient accumulation",
        "input_source": "declared real DeepSeek/Qwen3 text input bank",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "claim_boundary": "One checkpoint, embedding weight, and declared text bank; not a population or loss-quality guarantee.",
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "single_changed_boundary": "embedding weight backward accumulation"},
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "projection_positive": sum(v > 0 for v in projections), "projection_negative": sum(v < 0 for v in projections),
            "aligned_write_mean": sum(aligned) / len(aligned) if aligned else None,
            "aligned_write_interval_normal_95": interval(aligned) if aligned else None,
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
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
