#!/usr/bin/env python3
"""Probe the Gemma-4 soft-capped NLL boundary on a real text bank.

The model forward and soft-capped logits are shared.  The candidate uses the
model's native causal-LM loss implementation; the reference uses an explicit
FP32 shift/log-softmax/gather on the identical logits.  This is deliberately
an NLL-boundary probe, not a claim about the upstream model kernels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import Gemma4ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/google/gemma-4-E2B")
DEFAULT_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


def relative_norm(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(denominator, 1e-30)


def normal_interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(len(values))
    return [mean - half, mean + half]


def explicit_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    # Match the standard causal-LM shift, then make the comparison explicit.
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    values = logits.float().view(-1, vocab_size)
    targets = shifted.reshape(-1).to(values.device)
    keep = targets != -100
    return -values[keep].log_softmax(dim=-1).gather(1, targets[keep, None]).mean()


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    if len(states) != args.states or args.states < 4 or args.states % 2:
        raise ValueError("states must be an even number available in the input bank")
    device = torch.device(args.device)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()
    model.config.use_cache = False
    target = model.lm_head.weight
    vocab_size = model.config.text_config.vocab_size
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state in states:
        ids = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        labels = ids.clone()
        model.zero_grad(set_to_none=True)
        output = model(input_ids=ids, labels=None, use_cache=False)
        logits = output.logits
        candidate_loss = model.loss_function(logits, labels, vocab_size)
        candidate_loss.backward(retain_graph=True)
        candidate_gradient = target.grad.detach().float().cpu().clone()
        candidate_write = first_step_write(candidate_gradient, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        reference_loss = explicit_loss(logits, labels, vocab_size)
        reference_loss.backward()
        reference_gradient = target.grad.detach().float().cpu().clone()
        reference_write = first_step_write(reference_gradient, args.learning_rate, args.eps)

        effect = candidate_gradient - reference_gradient
        write_effect = candidate_write - reference_write
        effects.append(effect.reshape(-1).double())
        writes.append(write_effect.reshape(-1).double())
        references.append(reference_write.reshape(-1).double())
        rows.append({
            "state_id": state.get("state_id"),
            "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
            "gradient_effect_rms_over_reference": relative_norm(effect, reference_gradient),
            "write_effect_rms_over_reference": relative_norm(write_effect, reference_write),
            "write_signed_mean": float(write_effect.mean().item()),
        })
        del ids, labels, output, logits, candidate_loss, reference_loss
        torch.cuda.empty_cache()

    split = len(effects) // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm().item())
    projections: list[float] = []
    if direction_norm:
        unit = direction / direction_norm
        projections = [float(torch.dot(x, unit).item()) for x in effects[split:]]
    aligned = [
        float(torch.dot(effect, reference).item())
        / max(float(torch.dot(reference, reference).item()), 1e-30)
        for effect, reference in zip(writes[split:], references[split:])
    ]
    return {
        "schema": "kernel-analyzer-gemma4-softcapped-nll-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Gemma-4 soft-capped causal NLL boundary",
        "candidate": "native Gemma-4 causal-LM loss on shared soft-capped logits",
        "reference": "explicit FP32 shifted log-softmax/gather on identical logits",
        "input_source": "declared real Gemma-4 text input bank",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate, "weight_decay": 0.0},
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_logits_graph": True, "single_changed_boundary": "NLL loss evaluation"},
        "claim_boundary": "One Gemma-4 checkpoint, lm_head carrier and declared text bank; not a population or loss-quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": normal_interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections),
            "projection_negative": sum(x < 0 for x in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": normal_interval(aligned),
            "loss_difference_interval_normal_95": normal_interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
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
