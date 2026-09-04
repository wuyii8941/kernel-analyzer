#!/usr/bin/env python3
"""Summarize a loss-direction stress result without treating it as training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    rows = source["rows"]
    scales = sorted({float(row["scale"]) for row in rows})
    summary = []
    for scale in scales:
        selected = [row for row in rows if float(row["scale"]) == scale]
        named = {row["direction"]: row for row in selected}
        random_changes = [
            float(row["mean_loss_change"])
            for row in selected
            if row["direction"].startswith("random_direction_seed_")
        ]
        measured = float(named["measured_direction"]["mean_loss_change"])
        opposite = float(named["opposite_direction"]["mean_loss_change"])
        normal = float(named["normal_training_direction"]["mean_loss_change"])
        random_abs_median = statistics.median(abs(value) for value in random_changes)
        summary.append({
            "scale": scale,
            "measured_direction_loss_change": measured,
            "opposite_direction_loss_change": opposite,
            "normal_training_direction_loss_change": normal,
            "random_direction_loss_change_min": min(random_changes),
            "random_direction_loss_change_max": max(random_changes),
            "random_direction_absolute_change_median": random_abs_median,
            "measured_absolute_over_random_median": (
                abs(measured) / random_abs_median if random_abs_median else None
            ),
            "measured_outside_random_range": bool(
                measured < min(random_changes) or measured > max(random_changes)
            ),
        })
    payload = {
        "schema": "kernel-analyzer-loss-direction-stress-summary-v1",
        "status": "COMPLETE",
        "source": str(args.input),
        "case": {
            "model": source["model"],
            "parameter": source["parameter"],
            "evaluation_state_count": source["evaluation_state_count"],
            "candidate_minus_repair_l2": source["candidate_minus_repair_l2"],
        },
        "rows": summary,
        "observations": {
            "measured_loss_change_signs": [
                "positive" if row["measured_direction_loss_change"] > 0 else
                "negative" if row["measured_direction_loss_change"] < 0 else "zero"
                for row in summary
            ],
            "scales_outside_sampled_random_range": [
                row["scale"] for row in summary if row["measured_outside_random_range"]
            ],
        },
        "interpretation": (
            "This controlled test reports whether the measured parameter displacement follows "
            "a loss-sensitive direction relative to norm-matched random directions. The sign and "
            "scale dependence are reported directly; no straight-line perturbation is treated as "
            "a prediction of future training."
        ),
        "claim_boundary": source["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
