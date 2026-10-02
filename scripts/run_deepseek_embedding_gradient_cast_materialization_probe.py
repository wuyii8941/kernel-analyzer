#!/usr/bin/env python3
"""Isolate the DeepSeek embedding-gradient BF16 materialisation boundary.

The upstream embedding cotangent is captured once from the real model.  The
same cotangent is accumulated in FP32, then compared as (a) the native BF16
gradient write and (b) an FP32 gradient retained for the optimizer update.
This separates final gradient materialisation from repeated-index accumulation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def write(g: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = g.float()
    return -lr * g / (g.abs() + eps)


def ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a).item()) / max(float(torch.linalg.vector_norm(b).item()), 1e-30)


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    m = float(x.mean().item())
    if len(values) < 2:
        return [m, m]
    h = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(len(values))
    return [m - h, m + h]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    if len(states) != args.states or args.states < 4 or args.states % 2:
        raise ValueError("states must be an even number available in the input bank")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    model.config.use_cache = False
    embedding = model.model.embed_tokens
    target = embedding.weight
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state in states:
        values = state.get("token_ids", state.get("input_ids"))
        ids = torch.tensor([values], dtype=torch.long, device=device)
        labels = ids.clone()
        captured: dict[str, torch.Tensor] = {}

        def capture_output(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            output.register_hook(lambda grad: captured.setdefault("cotangent", grad.detach()))
            return output

        handle = embedding.register_forward_hook(capture_output)
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        handle.remove()
        cotangent = captured.get("cotangent")
        if cotangent is None:
            raise RuntimeError("embedding cotangent was not captured")
        cotangent = cotangent.float().cpu()
        # The DeepSeek embedding matrix is too large for a second full
        # FP32 buffer on the measurement GPU.  The source intervention is
        # arithmetic-only, so moving the captured cotangent and its exact
        # index-add to CPU preserves the compared vectors without changing
        # the model forward/backward path.
        raw = torch.zeros(tuple(target.shape), dtype=torch.float32, device="cpu")
        raw.index_add_(0, ids.reshape(-1).cpu(), cotangent.reshape(-1, cotangent.shape[-1]))
        native_gradient = raw.to(torch.bfloat16).float()
        reference_gradient = raw
        native_write = write(native_gradient, args.learning_rate, args.eps)
        reference_write = write(reference_gradient, args.learning_rate, args.eps)
        effect = native_write - reference_write
        effects.append(effect.reshape(-1).double().cpu())
        writes.append(effect.reshape(-1).double().cpu())
        references.append(reference_write.reshape(-1).double().cpu())
        rows.append({
            "state_id": state.get("state_id", state.get("sequence_id")),
            "loss": float(loss.detach().cpu()),
            "gradient_effect_rms_over_fp32_reference": ratio(native_gradient - reference_gradient, reference_gradient),
            "write_effect_rms_over_fp32_reference": ratio(effect, reference_write),
            "write_aligned": float(torch.dot(effect.reshape(-1), reference_write.reshape(-1)).item())
            / max(float(torch.dot(reference_write.reshape(-1), reference_write.reshape(-1)).item()), 1e-30),
        })
        del ids, labels, loss, cotangent, raw
        torch.cuda.empty_cache()

    split = len(effects) // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm().item())
    projections: list[float] = []
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(torch.dot(x, direction).item()) for x in effects[split:]]
    aligned = [
        float(torch.dot(effect, reference).item()) / max(float(torch.dot(reference, reference).item()), 1e-30)
        for effect, reference in zip(writes[split:], references[split:])
    ]
    result = {
        "schema": "kernel-analyzer-deepseek-embedding-gradient-cast-materialization-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "embedding_dense_backward FP32-to-BF16 gradient materialisation",
        "candidate": "native BF16 embedding gradient write",
        "reference": "same captured embedding cotangent accumulated and retained in FP32",
        "input_source": str(args.input_bank),
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_embedding_cotangent": True, "single_changed_boundary": "embedding gradient dtype materialisation"},
        "claim_boundary": "One DeepSeek checkpoint, embedding carrier and declared text bank; source and fixed-suite update only, not a population or quality claim.",
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_fp32_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_fp32_reference"] for r in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections), "projection_negative": sum(x < 0 for x in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "calibration_direction_norm": direction_norm,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
