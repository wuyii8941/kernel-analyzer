#!/usr/bin/env python3
"""Natural Gemma-4 RMSNorm row-reduction-order probe.

The model, text states, downstream graph and target parameter are shared.  The
only changed boundary is the order used to reduce the FP32 row square-sum in
the RMSNorm denominator.  This is a source experiment, not a claim about the
released kernel implementation unless the runtime binding is separately
confirmed.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True)) / (len(values) ** 0.5)
    return [mean - half, mean + half]


def first_step_write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -lr * g / (g.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    den = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / den if den else 0.0


def reversed_row_reduction(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    value = hidden_states.float()
    squares = value * value
    reversed_sum = torch.flip(squares, dims=(-1,)).sum(dim=-1, keepdim=True)
    mean_squared = reversed_sum / squares.shape[-1] + self.eps
    output = value * torch.pow(mean_squared, -0.5)
    if self.with_scale:
        output = output * self.weight.float()
    return output.type_as(hidden_states)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    norm = model.model.language_model.layers[args.layer].input_layernorm
    target = norm.weight
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        ids = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
        labels = ids.clone()
        model.zero_grad(set_to_none=True)
        native = model(input_ids=ids, labels=labels, use_cache=False)
        native_loss = native.loss
        native_loss.backward()
        native_gradient = target.grad.detach().float().clone()
        native_write = first_step_write(native_gradient, args.learning_rate, args.eps)

        original = norm.forward
        norm.forward = types.MethodType(reversed_row_reduction, norm)
        try:
            model.zero_grad(set_to_none=True)
            reference = model(input_ids=ids, labels=labels, use_cache=False)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_gradient = target.grad.detach().float().clone()
        finally:
            norm.forward = original
        reference_write = first_step_write(reference_gradient, args.learning_rate, args.eps)
        effect = native_write - reference_write
        rows.append(
            {
                "state_id": state.get("state_id", index),
                "loss_difference": float((native_loss - reference_loss).detach().cpu()),
                "gradient_effect_rms_over_reference": relative(native_gradient - reference_gradient, reference_gradient),
                "write_effect_rms_over_reference": relative(effect, reference_write),
                "write_aligned": float(torch.sum(effect * reference_write).item())
                / max(float(torch.sum(reference_write * reference_write).item()), 1e-30),
            }
        )
        del native, reference, native_loss, reference_loss, ids, labels
        torch.cuda.empty_cache()

    split = len(rows) // 2
    aligned = [row["write_aligned"] for row in rows[split:]]
    return {
        "schema": "kernel-analyzer-gemma4-rms-row-reduction-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "target": f"model.language_model.layers[{args.layer}].input_layernorm.weight",
        "candidate": "native Gemma-4 RMSNorm row square-sum reduction",
        "reference": "same RMSNorm with reversed FP32 row square-sum reduction and original write-back",
        "input_bank": str(args.input_bank),
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "aligned_positive": sum(x > 0 for x in aligned),
            "aligned_negative": sum(x < 0 for x in aligned),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
        },
        "claim_boundary": "One Gemma-4 checkpoint, one RMSNorm layer and the declared real text bank; source candidate only, no population or loss-quality claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument("--input-bank", type=Path, default=Path("results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--states", type=int, default=26)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
