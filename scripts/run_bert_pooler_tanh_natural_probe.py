#!/usr/bin/env python3
"""Natural BERT pooler-tanh materialization probe.

The candidate uses the model-dtype pooler tanh.  The reference changes only
that boundary to an FP32 tanh followed by one write-back, while keeping the
same BERT weights, token states, classifier and loss.  This is a natural
training-path screen for a semantic family not represented by GELU/SiLU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def windows(tokenizer: Any, source: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    ids = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    stride = max(1, seq_len // 2)
    need = (count - 1) * stride + seq_len
    if ids.numel() < need:
        raise RuntimeError(f"text source has {ids.numel()} tokens, need {need}")
    return [ids[i * stride : i * stride + seq_len].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if len(values) < 2:
        return [float(x.mean()), float(x.mean())]
    m = float(x.mean())
    h = 1.96 * float(x.std(unbiased=True)) / (len(values) ** 0.5)
    return [m - h, m + h]


def run_once(model: torch.nn.Module, ids: torch.Tensor, label: torch.Tensor, target: torch.nn.Parameter, fp32_tanh: bool) -> tuple[float, torch.Tensor]:
    pooler = model.bert.pooler
    original = pooler.forward

    def patched(hidden_states: torch.Tensor) -> torch.Tensor:
        first = hidden_states[:, 0]
        pooled = pooler.dense(first)
        if fp32_tanh:
            return torch.tanh(pooled.float()).to(pooled.dtype)
        return pooler.activation(pooled)

    pooler.forward = patched
    model.zero_grad(set_to_none=True)
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=label)
        output.loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient missing")
        return float(output.loss.detach()), target.grad.detach().float().cpu().clone()
    finally:
        pooler.forward = original


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/data1/tzh/models/prajjwal1/bert-tiny"))
    p.add_argument("--text-source", type=Path, default=ROOT / "docs/root_cause_closure_current.md")
    p.add_argument("--parameter", default="bert.embeddings.word_embeddings.weight")
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True, dtype=dtype).to(device).eval()
    named = dict(model.named_parameters())
    if args.parameter not in named:
        raise KeyError(args.parameter)
    target = named[args.parameter]
    gradients: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(device)
        label = torch.tensor([state_id % 2], dtype=torch.long, device=device)
        native_loss, native_grad = run_once(model, ids, label, target, False)
        fp32_loss, fp32_grad = run_once(model, ids, label, target, True)
        effect = fp32_grad - native_grad
        reference = native_grad
        aligned = float(torch.sum(effect.double() * reference.double()) / torch.sum(reference.double() ** 2).clamp_min(1e-30))
        gradients.append(effect.double().reshape(-1))
        writes.append(effect.double().reshape(-1))
        references.append(reference.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_tanh_loss": fp32_loss,
            "loss_difference_fp32_minus_native": fp32_loss - native_loss,
            "gradient_effect_rms_over_native": relative(effect, native_grad),
            "aligned_gradient_ratio": aligned,
        })

    split = args.states // 2
    direction = torch.stack(gradients[:split]).mean(dim=0)
    norm = float(torch.linalg.vector_norm(direction))
    projections: list[float] = []
    if norm:
        direction = direction / norm
        projections = [float(torch.dot(value, direction)) for value in gradients[split:]]
    aligned_values = [row["aligned_gradient_ratio"] for row in rows[split:]]
    result = {
        "schema": "kernel-analyzer-bert-pooler-tanh-natural-probe-v1",
        "status": "COMPLETE_NATURAL_TANH_POOLER_PROBE",
        "model": str(args.model),
        "operator": "BERT pooler tanh materialization",
        "parameter": args.parameter,
        "candidate": "native model-dtype pooler tanh",
        "reference": "same pooler with FP32 tanh followed by one original-dtype write",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_classifier_and_loss": True,
            "single_changed_boundary": "pooler tanh evaluation/materialization",
        },
        "state_count": len(rows),
        "calibration_count": split,
        "confirmation_count": len(rows) - split,
        "rows": rows,
        "summary": {
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "aligned_gradient_mean_confirmation": sum(aligned_values) / len(aligned_values),
            "aligned_gradient_interval_normal_95": interval(aligned_values),
            "aligned_positive_count": sum(v > 0 for v in aligned_values),
            "aligned_negative_count": sum(v < 0 for v in aligned_values),
            "confirmation_projection_interval_normal_95": interval(projections),
            "confirmation_projection_positive": sum(v > 0 for v in projections),
            "confirmation_projection_negative": sum(v < 0 for v in projections),
        },
        "claim_boundary": "One BERT-tiny checkpoint, one sequence-classification pooler path and the declared text bank; no population or long-run quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
