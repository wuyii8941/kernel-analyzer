#!/usr/bin/env python3
"""Summarize repeated paired-training loss and parameter differences."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]]
    # Two-sided 95% Student interval; the frozen design has four repeats.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    center = mean(values)
    half = critical * stdev(values) / math.sqrt(len(values))
    return [center - half, center + half]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    cases = {}
    complete = True
    for case in protocol["cases"]:
        case_id = case["case_id"]
        rows = []
        for repeat in case["repeats"]:
            index = int(repeat["repeat"])
            path = args.result_root / case_id / f"repeat{index}.json"
            if not path.exists():
                complete = False
                rows.append({"repeat": index, "status": "ABSTAIN_RESULT_MISSING"})
                continue
            item = json.loads(path.read_text())
            if item.get("status") != "COMPLETE":
                complete = False
                rows.append({
                    "repeat": index,
                    "status": "ABSTAIN_RESULT_NOT_COMPLETE",
                    "source_artifact": str(path),
                    "observed_status": item.get("status"),
                })
                continue
            loss = item["loss_audit"]
            levels = item.get("statistics", {}).get("levels", {})
            direct = levels.get("local", {})
            feedback = levels.get("feedback", {})
            actual = levels.get("actual", {})
            rows.append({
                "repeat": index,
                "status": item["status"],
                "steps": item["step_count"],
                "final_loss_gap": loss["final_gap"],
                "mean_loss_gap": loss["last_512_mean"],
                "maximum_absolute_loss_gap": loss["max_abs_gap"],
                "final_parameter_distance": item["final_drift_l2"],
                "direct_direction_score": direct.get("coherence_amplification"),
                "direct_direction_above_own_null": direct.get("above_sign_flip_95"),
                "feedback_direction_score": feedback.get("coherence_amplification"),
                "feedback_direction_above_own_null": feedback.get("above_sign_flip_95"),
                "actual_direction_score": actual.get("coherence_amplification"),
                "actual_direction_above_own_null": actual.get("above_sign_flip_95"),
                "direct_feedback_actual_cosines": item.get("statistics", {}).get(
                    "resultant_cosines"
                ),
                "source_artifact": str(path),
            })
        measured = [row for row in rows if row["status"] == "COMPLETE"]
        summary = {"role": case["role"], "repeats": rows}
        if measured:
            final_gaps = [float(row["final_loss_gap"]) for row in measured]
            mean_gaps = [float(row["mean_loss_gap"]) for row in measured]
            distances = [float(row["final_parameter_distance"]) for row in measured]
            summary["aggregate"] = {
                "completed_repeats": len(measured),
                "final_loss_gap_mean": mean(final_gaps),
                "final_loss_gap_95_interval_across_repeats": interval(final_gaps),
                "same_loss_gap_sign_count": max(
                    sum(value > 0 for value in final_gaps),
                    sum(value < 0 for value in final_gaps),
                ),
                "stream_mean_loss_gap_mean": mean(mean_gaps),
                "stream_mean_loss_gap_95_interval_across_repeats": interval(mean_gaps),
                "same_stream_mean_loss_gap_sign_count": max(
                    sum(value > 0 for value in mean_gaps),
                    sum(value < 0 for value in mean_gaps),
                ),
                "final_parameter_distance_mean": mean(distances),
                "final_parameter_distance_95_interval_across_repeats": interval(distances),
                "direct_direction_above_own_null_count": sum(
                    row.get("direct_direction_above_own_null") is True for row in measured
                ),
                "feedback_direction_above_own_null_count": sum(
                    row.get("feedback_direction_above_own_null") is True for row in measured
                ),
                "actual_direction_above_own_null_count": sum(
                    row.get("actual_direction_above_own_null") is True for row in measured
                ),
            }
        cases[case_id] = summary
    payload = {
        "schema": "kernel-analyzer-independent-consequence-summary-v1",
        "status": "COMPLETE" if complete else "COMPLETE_WITH_ABSTENTIONS",
        "protocol": str(args.protocol),
        "cases": cases,
        "claim_boundary": (
            "Intervals describe four disjoint input streams from one pretrained checkpoint. "
            "They do not represent independent pretraining initializations or full-parameter training."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "case_count": len(cases)}))


if __name__ == "__main__":
    main()
