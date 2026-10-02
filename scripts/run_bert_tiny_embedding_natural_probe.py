#!/usr/bin/env python3
"""Probe the real BERT embedding-backward accumulation boundary.

The model executes an ordinary masked-LM training graph, but the decoder is
untied so the target embedding gradient has one semantic consumer.  A hook
captures the actual gradient entering the word-embedding output.  The native
embedding backward result is compared with an explicit FP32 index-add using
the identical token ids and captured upstream gradient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import BertForMaskedLM, BertTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/prajjwal1/bert-tiny")
DEFAULT_OUTPUT = ROOT / "results/property/new_problem_group_search_v1/bert_tiny_embedding_natural_20260918.json"


def build_inputs(tokenizer: Any, source: Path, count: int, length: int) -> list[dict[str, torch.Tensor]]:
    text = "\n".join(line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = max(1, length // 2)
    rows = []
    for index in range(count):
        start = index * stride
        if start + length > encoded.numel():
            break
        ids = encoded[start : start + length].clone()
        labels = ids.clone()
        mask = torch.zeros(length, dtype=torch.bool)
        mask[(index * 7 + 3) % length] = True
        mask[(index * 11 + 9) % length] = True
        labels[~mask] = -100
        rows.append({
            "input_ids": ids.unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "labels": labels.unsqueeze(0),
            "state_id": "docs_window_{}".format(index),
        })
    if len(rows) < count:
        raise RuntimeError("source text did not provide enough token windows")
    return rows


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -learning_rate * grad / (grad.abs() + eps)


def rel_norm(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).item())
    return 0.0 if denominator == 0.0 else float(torch.linalg.vector_norm(effect).item()) / denominator


def normal_interval(values: list[float]) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(tensor.std(unbiased=True).item()) / (len(values) ** 0.5)
    return [mean - half, mean + half]


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    tokenizer = BertTokenizerFast.from_pretrained(str(args.model), local_files_only=True)
    states = build_inputs(tokenizer, ROOT / "README.md", args.states, args.sequence_length)
    model = BertForMaskedLM.from_pretrained(str(args.model), local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    # Avoid the tied decoder contributing to the embedding parameter gradient.
    decoder = model.cls.predictions.decoder
    decoder.weight = torch.nn.Parameter(decoder.weight.detach().clone())
    word_embeddings = model.bert.embeddings.word_embeddings
    target = word_embeddings.weight
    rows = []
    effects = []
    write_effects = []
    write_references = []
    for state in states:
        captured: dict[str, torch.Tensor] = {}

        def capture_output(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            output.register_hook(lambda grad: captured.setdefault("upstream", grad.detach()))

        handle = word_embeddings.register_forward_hook(capture_output)
        input_ids = state["input_ids"].to(device)
        attention_mask = state["attention_mask"].to(device)
        labels = state["labels"].to(device)
        model.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        output.loss.backward()
        handle.remove()
        native_gradient = target.grad.detach().float().cpu().clone()
        upstream = captured["upstream"].float().detach().cpu()
        ids = input_ids.detach().cpu().reshape(-1)
        explicit_gradient = torch.zeros_like(target.detach().float().cpu())
        explicit_gradient.index_add_(0, ids, upstream.reshape(-1, upstream.shape[-1]))
        native_write = first_step_write(native_gradient, args.learning_rate, args.eps)
        explicit_write = first_step_write(explicit_gradient, args.learning_rate, args.eps)
        gradient_effect = native_gradient - explicit_gradient
        write_effect = native_write - explicit_write
        effects.append(gradient_effect.reshape(-1).double())
        write_effects.append(write_effect.reshape(-1).double())
        write_references.append(explicit_write.reshape(-1).double())
        rows.append({
            "state_id": state["state_id"],
            "repeated_token_count": int(ids.numel() - ids.unique().numel()),
            "loss": float(output.loss.detach().cpu().item()),
            "gradient_effect_rms_over_reference": rel_norm(gradient_effect, explicit_gradient),
            "write_effect_rms_over_reference": rel_norm(write_effect, explicit_write),
            "gradient_signed_mean": float(gradient_effect.mean().item()),
            "write_signed_mean": float(write_effect.mean().item()),
        })
        del input_ids, attention_mask, labels, output
        torch.cuda.empty_cache()

    split = len(effects) // 2
    calibration = torch.stack(effects[:split])
    confirmation = torch.stack(effects[split:])
    direction = calibration.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    projections = []
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(value.item()) for value in confirmation @ direction]
    aligned = []
    for effect, repair in zip(write_effects[split:], write_references[split:]):
        aligned.append(float(torch.dot(effect, repair).item()) / max(float(torch.dot(repair, repair).item()), 1e-30))
    return {
        "schema": "kernel-analyzer-bert-embedding-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "BERT word-embedding backward accumulation",
        "candidate": "native embedding backward in ordinary masked-LM graph",
        "reference": "explicit FP32 index-add from identical captured upstream gradient",
        "input_source": "repository README token windows with deterministic masked labels",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate, "weight_decay": 0.0},
        "claim_boundary": "One real pretrained checkpoint, untied MLM decoder, one embedding table and the declared input bank; not a population or loss-quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "mean_repeated_token_count": sum(row["repeated_token_count"] for row in rows) / len(rows),
            "gradient_effect_rms_mean": sum(row["gradient_effect_rms_over_reference"] for row in rows) / len(rows),
            "write_effect_rms_mean": sum(row["write_effect_rms_over_reference"] for row in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": normal_interval(projections) if projections else None,
            "projection_positive": sum(value > 0 for value in projections),
            "projection_negative": sum(value < 0 for value in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": normal_interval(aligned),
            "direction_norm": direction_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
