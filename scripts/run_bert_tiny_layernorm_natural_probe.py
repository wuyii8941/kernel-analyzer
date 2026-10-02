#!/usr/bin/env python3
"""Natural BERT LayerNorm probe with a declared arithmetic reference.

The model and token windows are real, but the result is deliberately scoped to
the selected checkpoint, LayerNorm, and input bank.  The candidate is the
compiled native BERT path.  The reference changes only the selected LayerNorm
reduction and materialisation to an explicitly BF16 implementation.  No
provenance side-channel is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import types
from typing import Any

import torch
from transformers import BertForMaskedLM, BertTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/prajjwal1/bert-tiny")
DEFAULT_OUTPUT = ROOT / "results/property/new_problem_group_search_v1/bert_tiny_layernorm_natural_20260918.json"


def build_inputs(tokenizer: Any, source: Path, count: int, length: int) -> list[dict[str, torch.Tensor]]:
    text = "\n".join(
        line.strip() for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = max(1, length // 2)
    rows = []
    for index in range(count):
        start = index * stride
        if start + length > encoded.numel():
            break
        ids = encoded[start : start + length].clone()
        rows.append({
            "input_ids": ids.unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "state_id": "docs_window_{}".format(index),
        })
    if len(rows) < count:
        raise RuntimeError("source text did not provide enough token windows")
    return rows


def install_layernorm_reference(module: torch.nn.Module, mode: str) -> None:
    def forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        value = hidden_states.float() if mode == "fp32" else hidden_states.to(hidden_states.dtype)
        mean = value.mean(dim=-1, keepdim=True)
        centered = value - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        output = normalized * self.weight.to(value.dtype) + self.bias.to(value.dtype)
        return output.to(hidden_states.dtype)

    module.forward = types.MethodType(forward, module)


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -learning_rate * grad / (grad.abs() + eps)


def rel_norm(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).item())
    if denominator == 0.0:
        return 0.0
    return float(torch.linalg.vector_norm(effect).item()) / denominator


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
    source = ROOT / "README.md"
    states = build_inputs(tokenizer, source, args.states, args.sequence_length)

    candidate_model = BertForMaskedLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    reference_model = BertForMaskedLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    candidate_norm = candidate_model.bert.encoder.layer[args.layer].attention.output.LayerNorm
    reference_norm = reference_model.bert.encoder.layer[args.layer].attention.output.LayerNorm
    target_candidate = candidate_norm.weight
    target_reference = reference_norm.weight
    if args.reference_mode in ("bf16", "fp32"):
        install_layernorm_reference(reference_norm, args.reference_mode)

    candidate = torch.compile(candidate_model, backend="inductor", fullgraph=False, dynamic=False)
    reference = (
        reference_model
        if args.reference_mode == "eager"
        else torch.compile(reference_model, backend="inductor", fullgraph=False, dynamic=False)
    )
    rows = []
    effects = []
    references = []
    write_effects = []
    write_references = []
    for state in states:
        input_ids = state["input_ids"].to(device)
        attention_mask = state["attention_mask"].to(device)
        candidate_model.zero_grad(set_to_none=True)
        candidate_loss = candidate(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=input_ids, return_dict=True).loss
        candidate_loss.backward()
        candidate_gradient = target_candidate.grad.detach().float().cpu().clone()
        candidate_write = first_step_write(candidate_gradient, args.learning_rate, args.eps)

        reference_model.zero_grad(set_to_none=True)
        reference_loss = reference(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=input_ids, return_dict=True).loss
        reference_loss.backward()
        reference_gradient = target_reference.grad.detach().float().cpu().clone()
        reference_write = first_step_write(reference_gradient, args.learning_rate, args.eps)

        gradient_effect = candidate_gradient - reference_gradient
        write_effect = candidate_write - reference_write
        effects.append(gradient_effect)
        references.append(reference_gradient)
        write_effects.append(write_effect)
        write_references.append(reference_write)
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
        del input_ids, attention_mask, candidate_loss, reference_loss
        torch.cuda.empty_cache()

    split = len(effects) // 2
    calibration = torch.stack([x.double() for x in effects[:split]])
    confirmation = torch.stack([x.double() for x in effects[split:]])
    direction = calibration.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    if direction_norm:
        direction = direction / direction_norm
        projections = confirmation @ direction
        projection_values = [float(value.item()) for value in projections]
    else:
        projection_values = []
    aligned = []
    for effect, repair in zip(write_effects[split:], write_references[split:]):
        denominator = float(torch.dot(repair.double(), repair.double()).item())
        aligned.append(float(torch.dot(effect.double(), repair.double()).item()) / max(denominator, 1e-30))

    return {
        "schema": "kernel-analyzer-bert-layernorm-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "BERT LayerNorm attention.output.LayerNorm",
        "layer": args.layer,
        "candidate": "compiled native BERT LayerNorm",
        "reference": "compiled explicit {} reduction and materialisation".format(args.reference_mode.upper()),
        "input_source": "repository README token windows",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate, "weight_decay": 0.0},
        "claim_boundary": "One real pretrained checkpoint, one LayerNorm, and the declared document-derived input bank; not a population or loss-quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(row["gradient_effect_rms_over_reference"] for row in rows) / len(rows),
            "write_effect_rms_mean": sum(row["write_effect_rms_over_reference"] for row in rows) / len(rows),
            "projection_mean": (sum(projection_values) / len(projection_values)) if projection_values else None,
            "projection_interval_normal_95": normal_interval(projection_values) if projection_values else None,
            "projection_positive": sum(value > 0 for value in projection_values),
            "projection_negative": sum(value < 0 for value in projection_values),
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
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--reference-mode", choices=("eager", "bf16", "fp32"), default="eager")
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
