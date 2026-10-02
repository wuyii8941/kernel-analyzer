#!/usr/bin/env python3
"""Natural BERT MLP GELU materialization probe.

Only the activation evaluation is changed.  The real BERT checkpoint,
document-derived token windows, classifier and target parameter are shared by
both branches.  A stable aligned write interval is evidence for this declared
boundary only; it is not a population or long-run quality claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if len(values) < 2:
        return [float(x.mean()), float(x.mean())]
    mean = float(x.mean())
    half = 1.96 * float(x.std(unbiased=True)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def windows(tokenizer: Any, source: Path, sequence_length: int, count: int) -> list[torch.Tensor]:
    text = source.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    stride = max(1, sequence_length // 2)
    need = (count - 1) * stride + sequence_length
    if ids.numel() < need:
        raise RuntimeError("text source has {} tokens, need {}".format(ids.numel(), need))
    return [ids[i * stride : i * stride + sequence_length].clone() for i in range(count)]


def run_branch(model, layer, ids, labels, target, reference: bool):
    original = layer.intermediate_act_fn

    class FP32GELU(nn.Module):
        def forward(self, value):
            # Exact GELU is the declared semantic reference for BERT's config.
            return F.gelu(value.float(), approximate="none").to(value.dtype)

    if reference:
        layer.intermediate_act_fn = FP32GELU()
    model.zero_grad(set_to_none=True)
    try:
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient missing")
        return float(loss.detach()), target.grad.detach().float().cpu().clone()
    finally:
        layer.intermediate_act_fn = original


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/prajjwal1/bert-tiny"))
    parser.add_argument("--text-source", type=Path, default=ROOT / "docs/root_cause_closure_current.md")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, dtype=dtype
    ).to(device).eval()
    layer = model.bert.encoder.layer[args.layer].intermediate
    target = model.bert.encoder.layer[args.layer].output.dense.weight

    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(device)
        labels = torch.tensor([state_id % 2], dtype=torch.long, device=device)
        native_loss, native_grad = run_branch(model, layer, ids, labels, target, False)
        fp32_loss, fp32_grad = run_branch(model, layer, ids, labels, target, True)
        effect = fp32_grad - native_grad
        native_write = -args.learning_rate * native_grad / (native_grad.abs() + args.eps)
        reference_write = -args.learning_rate * fp32_grad / (fp32_grad.abs() + args.eps)
        write_effect = reference_write - native_write
        aligned = float(torch.dot(write_effect.reshape(-1), native_write.reshape(-1))) / max(
            float(torch.dot(native_write.reshape(-1), native_write.reshape(-1))), 1e-30
        )
        effects.append(effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        references.append(native_write.double().reshape(-1))
        rows.append(
            {
                "state_id": state_id,
                "native_loss": native_loss,
                "fp32_gelu_loss": fp32_loss,
                "loss_difference_fp32_minus_native": fp32_loss - native_loss,
                "gradient_effect_rms_over_native": ratio(effect, native_grad),
                "write_effect_rms_over_native": ratio(write_effect, native_write),
                "write_aligned_fp32_minus_native": aligned,
            }
        )

    split = args.states // 2
    calibration = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(calibration))
    projections: list[float] = []
    if direction_norm > 0:
        direction = calibration / direction_norm
        projections = [float(torch.dot(value, direction)) for value in effects[split:]]
    aligned_values = [row["write_aligned_fp32_minus_native"] for row in rows[split:]]
    result = {
        "schema": "kernel-analyzer-bert-gelu-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "BERT MLP GELU evaluation/materialization",
        "parameter": "bert.encoder.layer.{}.output.dense.weight".format(args.layer),
        "candidate": "native BERT model-dtype GELU",
        "reference": "same BERT GELU evaluated in FP32 exact mode and written in the original dtype",
        "layer": args.layer,
        "input_source": "real document-derived token windows",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_classifier_and_loss": True,
            "single_changed_boundary": "MLP GELU evaluation/materialization",
        },
        "claim_boundary": "One BERT-tiny checkpoint, one MLP layer and the declared text bank; source and fixed-suite update evidence only.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "confirmation_aligned_write_mean": sum(aligned_values) / len(aligned_values),
            "confirmation_aligned_write_interval_normal_95": interval(aligned_values),
            "confirmation_projection_interval_normal_95": interval(projections),
            "confirmation_projection_positive": sum(x > 0 for x in projections),
            "confirmation_projection_negative": sum(x < 0 for x in projections),
            "loss_difference_interval_normal_95": interval([r["loss_difference_fp32_minus_native"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
