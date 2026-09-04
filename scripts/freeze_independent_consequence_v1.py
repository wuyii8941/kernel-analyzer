#!/usr/bin/env python3
"""Freeze disjoint input streams for repeated paired training consequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CASES = [
    {
        "case_id": "deepseek8b_seq256_backward_1714_in_out_ptr0",
        "role": "confirmed_update_effect",
        "source_bank": "results/property/tcmp_allop_v1/input_banks/deepseek8b_seq256_trajectory4096.json",
    },
    {
        "case_id": "deepseek8b_seq128_backward_1256_out_ptr0",
        "role": "confirmed_update_effect",
        "source_bank": "results/property/tcmp_allop_v1/input_banks/deepseek8b_seq128_trajectory4096.json",
    },
    {
        "case_id": "phi4_seq64_backward_495_out_ptr1",
        "role": "previously_centered_control",
        "source_bank": "results/property/tcmp_allop_v1/input_banks/phi4_seq64_trajectory4096.json",
    },
]
OFFSETS = (256, 1024, 1792, 2560)
STEPS = 32


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_cases = []
    for case in CASES:
        source_path = Path(case["source_bank"])
        source = json.loads(source_path.read_text())
        records = source.get("states", source.get("records", []))
        generated = []
        used_ids: set[str] = set()
        for repeat, offset in enumerate(OFFSETS):
            selected = records[offset : offset + STEPS]
            if len(selected) != STEPS:
                raise RuntimeError(f"{source_path} is too short for offset {offset}")
            ids = [str(row.get("state_id", row.get("sequence_id"))) for row in selected]
            if used_ids.intersection(ids):
                raise RuntimeError("repeated training input streams overlap")
            used_ids.update(ids)
            target = args.output_dir / f"{case['case_id']}_repeat{repeat}.json"
            payload = {
                "schema": "kernel-analyzer-independent-consequence-input-stream-v1",
                "status": "FROZEN_WITHOUT_READING_CONSEQUENCE_RESULTS",
                "case_id": case["case_id"],
                "repeat": repeat,
                "source_bank": str(source_path),
                "source_bank_sha256": digest(source_path),
                "source_offset": offset,
                "states": selected,
            }
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            generated.append({"repeat": repeat, "input_bank": str(target), "state_ids": ids})
        frozen_cases.append({**case, "repeats": generated})
    protocol = {
        "schema": "kernel-analyzer-independent-consequence-protocol-v1",
        "status": "FROZEN_BEFORE_REPEATED_CONSEQUENCE_RUNS",
        "cases": frozen_cases,
        "steps_per_repeat": STEPS,
        "repeats_per_case": len(OFFSETS),
        "paired_design": (
            "Each repeat starts candidate and repair from identical model and optimizer state "
            "and uses one input stream disjoint from every other repeat of that case."
        ),
        "primary_outputs": [
            "32-step paired training-loss gap curve",
            "final target-parameter separation",
            "direct, feedback, and actual update summaries",
        ],
        "stopping": "run all 32 steps; do not stop at the first nonzero loss gap",
        "claim_boundary": (
            "Repeats use disjoint data streams from the same pretrained checkpoint. "
            "They are independent paired training repeats for this controlled target-parameter "
            "protocol, not independent pretraining initializations or full-parameter training."
        ),
    }
    target = args.output_dir / "protocol.json"
    target.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": protocol["status"], "runs": len(CASES) * len(OFFSETS)}))


if __name__ == "__main__":
    main()
