#!/usr/bin/env python3
"""Measure Gemma-4 final-logit soft-cap effects at the lm_head write.

The hidden state and lm_head weights are shared.  Only the soft-cap
arithmetic is changed before the common loss, so this is a parameter-level
follow-up to the logits-only source probe.
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


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(len(values))
    return [mean - half, mean + half]


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


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
    cap = model.config.get_text_config().final_logit_softcapping
    vocab_size = model.config.get_text_config().vocab_size
    target = model.lm_head.weight
    rows: list[dict[str, Any]] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    losses: list[float] = []
    for state in states:
        ids = torch.tensor([state["token_ids"][: args.sequence_length]], dtype=torch.long, device=device)
        labels = ids.clone()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        # Keep the hidden state fixed so the only changed boundary is the
        # final-logit soft-cap arithmetic.
        pre = torch.nn.functional.linear(hidden.detach(), target)
        candidate = (pre.to(torch.bfloat16) / cap).tanh() * cap
        reference = (pre / cap).tanh() * cap
        candidate_loss = torch.nn.functional.cross_entropy(
            candidate[:, :-1, :].float().reshape(-1, vocab_size), labels[:, 1:].reshape(-1)
        )
        candidate_loss.backward(retain_graph=True)
        candidate_gradient = target.grad.detach().float().cpu().clone()
        candidate_write = first_step_write(candidate_gradient, args.learning_rate, args.eps)
        model.zero_grad(set_to_none=True)
        reference_loss = torch.nn.functional.cross_entropy(
            reference[:, :-1, :].reshape(-1, vocab_size), labels[:, 1:].reshape(-1)
        )
        reference_loss.backward()
        reference_gradient = target.grad.detach().float().cpu().clone()
        reference_write = first_step_write(reference_gradient, args.learning_rate, args.eps)
        write_effect = candidate_write - reference_write
        write_reference = reference_write
        rms = float(torch.linalg.vector_norm(write_effect).item()) / max(
            float(torch.linalg.vector_norm(write_reference).item()), 1e-30
        )
        aligned = float(torch.sum(write_effect * write_reference).item()) / max(
            float(torch.sum(write_reference * write_reference).item()), 1e-30
        )
        writes.append(write_effect.reshape(-1).double())
        references.append(write_reference.reshape(-1).double())
        losses.append(float((candidate_loss - reference_loss).detach().cpu().item()))
        rows.append(
            {
                "state_id": state.get("state_id"),
                "loss_difference": losses[-1],
                "write_effect_rms_over_reference": rms,
                "aligned_write": aligned,
            }
        )
        del ids, labels, hidden, pre, candidate, reference, candidate_loss, reference_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    split = len(rows) // 2
    aligned_confirmation = [row["aligned_write"] for row in rows[split:]]
    return {
        "schema": "kernel-analyzer-gemma4-final-logit-softcap-parameter-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Gemma-4 final-logit soft-cap materialization",
        "candidate": "native bfloat16 divide/tanh/multiply",
        "reference": "float32 divide/tanh/multiply followed by bfloat16 writeback",
        "input_source": "declared real Gemma-4 text input bank",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_hidden_state": True,
            "single_changed_boundary": "final-logit soft-cap arithmetic",
        },
        "claim_boundary": "Fixed-suite lm_head parameter write only; no population or long-run quality claim.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "write_effect_rms_mean": sum(row["write_effect_rms_over_reference"] for row in rows) / len(rows),
            "confirmation_aligned_mean": sum(aligned_confirmation) / len(aligned_confirmation),
            "confirmation_aligned_interval_normal_95": interval(aligned_confirmation),
            "confirmation_aligned_positive": sum(value > 0 for value in aligned_confirmation),
            "confirmation_aligned_negative": sum(value < 0 for value in aligned_confirmation),
            "loss_difference_interval_normal_95": interval(losses),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--states", type=int, default=26)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
