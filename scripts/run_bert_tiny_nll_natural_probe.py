#!/usr/bin/env python3
"""Probe a real BERT-tiny NLL boundary without provenance side channels.

The model forward and logits are shared.  Only the loss evaluation is changed:
the candidate uses native cross entropy on the model output, while the
reference uses an explicit FP32 log-softmax/gather expression.  This keeps the
comparison at the NLL boundary rather than comparing two independently
executed model regions.
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
DEFAULT_OUTPUT = ROOT / "results/property/new_problem_group_search_v1/bert_tiny_nll_natural_20260918.json"


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


def explicit_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    log_probs = values.log_softmax(dim=-1)
    flat_labels = labels.reshape(-1)
    flat_values = log_probs.reshape(-1, log_probs.shape[-1])
    keep = flat_labels != -100
    return -flat_values[keep, flat_labels[keep]].mean()


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
    states = build_inputs(tokenizer, args.text_source, args.states, args.sequence_length)
    model = BertForMaskedLM.from_pretrained(str(args.model), local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    compiled = torch.compile(model, backend="inductor", fullgraph=False, dynamic=False)
    target = model.cls.predictions.decoder.weight
    rows = []
    effects = []
    references = []
    write_effects = []
    write_references = []
    for state in states:
        input_ids = state["input_ids"].to(device)
        attention_mask = state["attention_mask"].to(device)
        labels = state["labels"].to(device)
        model.zero_grad(set_to_none=True)
        logits = compiled(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
        candidate_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        candidate_loss.backward(retain_graph=True)
        candidate_gradient = target.grad.detach().float().cpu().clone()
        candidate_write = first_step_write(candidate_gradient, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        reference_loss = explicit_nll(logits, labels)
        reference_loss.backward()
        reference_gradient = target.grad.detach().float().cpu().clone()
        reference_write = first_step_write(reference_gradient, args.learning_rate, args.eps)

        gradient_effect = candidate_gradient - reference_gradient
        write_effect = candidate_write - reference_write
        effects.append(gradient_effect.reshape(-1))
        references.append(reference_gradient.reshape(-1))
        write_effects.append(write_effect.reshape(-1))
        write_references.append(reference_write.reshape(-1))
        rows.append({
            "state_id": state["state_id"],
            "loss_candidate": float(candidate_loss.detach().cpu().item()),
            "loss_reference": float(reference_loss.detach().cpu().item()),
            "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
            "gradient_effect_rms_over_reference": rel_norm(gradient_effect, reference_gradient),
            "write_effect_rms_over_reference": rel_norm(write_effect, reference_write),
            "gradient_signed_mean": float(gradient_effect.mean().item()),
            "write_signed_mean": float(write_effect.mean().item()),
        })
        del input_ids, attention_mask, labels, logits, candidate_loss, reference_loss
        torch.cuda.empty_cache()

    split = len(effects) // 2
    calibration = torch.stack([x.double() for x in effects[:split]])
    confirmation = torch.stack([x.double() for x in effects[split:]])
    direction = calibration.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    projections = []
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(value.item()) for value in (confirmation @ direction)]
    aligned = []
    for effect, repair in zip(write_effects[split:], write_references[split:]):
        denominator = float(torch.dot(repair.double(), repair.double()).item())
        aligned.append(float(torch.dot(effect.double(), repair.double()).item()) / max(denominator, 1e-30))
    return {
        "schema": "kernel-analyzer-bert-nll-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "BERT masked-LM NLL boundary",
        "candidate": "native cross entropy on shared compiled logits",
        "reference": "explicit FP32 log-softmax and gather on identical logits",
        "input_source": str(args.text_source),
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate, "weight_decay": 0.0},
        "claim_boundary": "One real pretrained checkpoint, one NLL boundary, and the declared document-derived input bank; not a population or loss-quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(row["gradient_effect_rms_over_reference"] for row in rows) / len(rows),
            "write_effect_rms_mean": sum(row["write_effect_rms_over_reference"] for row in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": normal_interval(projections) if projections else None,
            "projection_positive": sum(value > 0 for value in projections),
            "projection_negative": sum(value < 0 for value in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": normal_interval(aligned),
            "loss_difference_interval_normal_95": normal_interval([row["loss_difference"] for row in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--text-source", type=Path, default=ROOT / "README.md")
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
