#!/usr/bin/env python3
"""Freeze the three-case optimizer/checkpoint follow-up before running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = [
    {
        "case_id": "deepseek8b_seq256_backward_1714_in_out_ptr0",
        "model": "deepseek8b",
        "sequence_length": 256,
        "task_id": "backward:1714:in_out_ptr0",
        "role": "previously_confirmed_update_effect",
        "target_parameter": "model.layers.6.self_attn.o_proj.weight",
    },
    {
        "case_id": "deepseek8b_seq128_backward_1256_out_ptr0",
        "model": "deepseek8b",
        "sequence_length": 128,
        "task_id": "backward:1256:out_ptr0",
        "role": "previously_confirmed_update_effect",
        "target_parameter": "model.layers.18.self_attn.v_proj.weight",
    },
    {
        "case_id": "phi4_seq64_backward_495_out_ptr1",
        "model": "phi4",
        "sequence_length": 64,
        "task_id": "backward:495:out_ptr1",
        "role": "previously_centered_control",
        "target_parameter": "model.embed_tokens.weight",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": "kernel-analyzer-optimizer-condition-benchmark-v1",
        "status": "FROZEN_BEFORE_FOLLOWUP_MEASUREMENT",
        "question": (
            "Do previously measured update effects remain when the declared target "
            "parameter and its AdamW moments have progressed beyond the first step?"
        ),
        "cases": CASES,
        "conditions": [
            {"name": "cold_start", "warmup_steps": 0, "reset_moments": False},
            {"name": "warm_step_8", "warmup_steps": 8, "reset_moments": False},
            {"name": "warm_step_32", "warmup_steps": 32, "reset_moments": False},
            {"name": "warm_step_32_moments_reset", "warmup_steps": 32, "reset_moments": True},
            {
                "name": "stateless_sgd_at_warm_step_32",
                "warmup_steps": 32,
                "source": "the matched gradient difference and repair gradient from warm_step_32",
            },
        ],
        "measurement": {
            "scope": "only the declared target parameter is updated during warmup",
            "matched_states_per_condition": 32,
            "calibration_states": 16,
            "confirmation_states": 16,
            "primary_stage": "actual AdamW update of the declared target parameter",
            "explanation_stages": ["declared implementation output", "parameter gradient"],
            "effects": ["additive", "repair_aligned", "residual_direction"],
            "multiplicity": "Holm correction over all case x condition x update-effect tests",
        },
        "fixed_boundaries": [
            "the same previously frozen case ID is used",
            "candidate and repair begin each measurement from identical parameters and moments",
            "the warmup input states do not overlap the 32 measurement input states",
            "no result may be replaced by a different case after measurement",
        ],
        "claim_boundary": (
            "This benchmark varies the checkpoint and AdamW state of one declared target "
            "parameter. It is not full-parameter pretraining and does not estimate a universal rate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "cases": len(CASES)}))


if __name__ == "__main__":
    main()
