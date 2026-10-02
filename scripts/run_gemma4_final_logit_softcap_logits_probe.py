#!/usr/bin/env python3
"""Probe Gemma-4 final-logit soft-cap arithmetic on a real text bank.

The model produces one shared pre-soft-cap logit tensor.  The candidate keeps
the native bfloat16 division/tanh/multiply sequence; the reference evaluates
that same sequence in float32 and writes back to bfloat16 before the loss.
This is a local/logit-gradient probe.  It is deliberately not a training
parameter or loss-quality certificate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import Gemma4ForConditionalGeneration


def normal_interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(len(values))
    return [mean - half, mean + half]


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
    rows: list[dict[str, Any]] = []
    aligned: list[float] = []
    rms: list[float] = []
    for state in states:
        ids = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        labels = ids[:, 1:].reshape(-1)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
            logits = model.lm_head(hidden)
        pre = logits.detach().float().requires_grad_()
        candidate = (pre.to(torch.bfloat16) / cap).tanh() * cap
        reference = (pre / cap).tanh() * cap
        candidate_loss = torch.nn.functional.cross_entropy(
            candidate[:, :-1, :].float().reshape(-1, vocab_size), labels
        )
        reference_loss = torch.nn.functional.cross_entropy(
            reference[:, :-1, :].reshape(-1, vocab_size), labels
        )
        candidate_grad = torch.autograd.grad(candidate_loss, pre, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(reference_loss, pre)[0]
        effect = candidate_grad - reference_grad
        denominator = float(torch.sum(reference_grad * reference_grad).item())
        aligned_value = float(torch.sum(effect * reference_grad).item()) / max(denominator, 1e-30)
        rms_value = float(torch.linalg.vector_norm(effect).item()) / max(
            float(torch.linalg.vector_norm(reference_grad).item()), 1e-30
        )
        aligned.append(aligned_value)
        rms.append(rms_value)
        rows.append(
            {
                "state_id": state.get("state_id"),
                "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
                "logit_gradient_effect_rms_over_reference": rms_value,
                "logit_gradient_aligned": aligned_value,
            }
        )
    return {
        "schema": "kernel-analyzer-gemma4-final-logit-softcap-logit-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Gemma-4 final-logit soft-cap materialization",
        "candidate": "native bfloat16 divide/tanh/multiply",
        "reference": "float32 divide/tanh/multiply followed by bfloat16 writeback",
        "input_source": "declared real Gemma-4 text input bank",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_pre_softcap_logits": True,
            "single_changed_boundary": "final-logit soft-cap arithmetic",
        },
        "claim_boundary": "Logit-gradient boundary only; not a parameter-write or population certificate.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "aligned_mean": sum(aligned) / len(aligned),
            "aligned_interval_normal_95": normal_interval(aligned),
            "aligned_positive": sum(x > 0 for x in aligned),
            "aligned_negative": sum(x < 0 for x in aligned),
            "effect_rms_mean": sum(rms) / len(rms),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument(
        "--input-bank",
        type=Path,
        default=Path("results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"),
    )
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
