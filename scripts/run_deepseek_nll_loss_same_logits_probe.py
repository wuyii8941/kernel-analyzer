#!/usr/bin/env python3
"""Same-logits DeepSeek causal-NLL boundary probe.

The model forward and logits are shared.  Only the loss evaluation changes:
the native causal-LM loss is compared with an explicit FP32 shifted
log-softmax/gather on the same logits.  The probe is scoped to the declared
checkpoint, text bank and lm_head/embedding carrier; it does not claim a
population or quality consequence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -lr * g / (g.abs() + eps)


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(len(values))
    return [mean - half, mean + half]


def explicit_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    values = logits.float().reshape(-1, vocab_size)
    targets = shifted.reshape(-1).to(values.device)
    keep = targets != -100
    return -values[keep].log_softmax(dim=-1).gather(1, targets[keep, None]).mean()


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
    target = model.lm_head.weight
    vocab_size = int(model.config.vocab_size)
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state in states:
        values = state.get("token_ids", state.get("input_ids"))
        ids = torch.tensor([values], dtype=torch.long, device=device)
        labels = ids.clone()
        model.zero_grad(set_to_none=True)
        output = model(input_ids=ids, labels=None, use_cache=False)
        logits = output.logits
        native_loss = model.loss_function(logits, labels, vocab_size)
        native_loss.backward(retain_graph=True)
        native_grad = target.grad.detach().float().cpu().clone()
        native_write = write(native_grad, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        reference_loss = explicit_loss(logits, labels, vocab_size)
        reference_loss.backward()
        reference_grad = target.grad.detach().float().cpu().clone()
        reference_write = write(reference_grad, args.learning_rate, args.eps)
        effect = native_grad - reference_grad
        write_effect = native_write - reference_write
        effects.append(effect.reshape(-1).double())
        writes.append(write_effect.reshape(-1).double())
        references.append(reference_write.reshape(-1).double())
        rows.append({
            "state_id": state.get("state_id", state.get("sequence_id")),
            "loss_difference": float((native_loss - reference_loss).detach().cpu()),
            "gradient_effect_rms_over_reference": ratio(effect, reference_grad),
            "write_effect_rms_over_reference": ratio(write_effect, reference_write),
            "write_aligned": float(torch.dot(write_effect.reshape(-1), reference_write.reshape(-1)).item())
            / max(float(torch.dot(reference_write.reshape(-1), reference_write.reshape(-1)).item()), 1e-30),
        })
        del ids, labels, output, logits, native_loss, reference_loss
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
        "schema": "kernel-analyzer-deepseek-nll-same-logits-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "DeepSeek causal NLL loss evaluation",
        "candidate": "native causal-LM loss on shared logits",
        "reference": "explicit FP32 shifted log-softmax/gather on identical logits",
        "input_source": str(args.input_bank),
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_logits_graph": True, "single_changed_boundary": "NLL loss evaluation"},
        "claim_boundary": "One DeepSeek checkpoint, lm_head/embedding carrier and declared text bank; not a population or loss-quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections), "projection_negative": sum(x < 0 for x in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
