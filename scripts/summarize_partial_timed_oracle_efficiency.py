#!/usr/bin/env python3
"""Summarize only wall times that were actually measured.

This is deliberately fail-closed: a missing Mamba time never becomes an
estimate, and a timed row outside the active Oracle cohort is not used to
compute recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--timed-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    timed = json.loads(args.timed_audit.read_text(encoding="utf-8"))
    comparison_rows = {
        str(row["case_id"]): row for row in comparison["rows"]
    }
    timed_rows = {
        str(row["case_id"]): row for row in timed["rows"]
    }
    active = [
        comparison_rows[case_id]
        for case_id in sorted(set(comparison_rows) & set(timed_rows))
    ]
    flagged = [
        row for row in active
        if float(row["oracle_prefix16_local_A"]) > 1.0
    ]
    all_seconds = sum(float(row["elapsed_seconds"]) for row in timed_rows.values())
    active_seconds = sum(
        float(timed_rows[row["case_id"]]["elapsed_seconds"]) for row in active
    )
    labels = [bool(row["label_32step_local_persistent"]) for row in active]
    positives = sum(labels)
    true_positives = sum(
        bool(row["label_32step_local_persistent"])
        for row in flagged
    )
    payload = {
        "schema": "kernel-analyzer-partial-timed-oracle-efficiency-v1",
        "status": "PARTIAL_RETROSPECTIVE_TIMES_ONLY",
        "timed_audit_status": timed["status"],
        "timed_rows": len(timed_rows),
        "expected_timed_rows": int(timed.get("expected_case_count", 12)),
        "timed_missing_case_ids": timed.get("missing_case_ids", []),
        "active_comparison_rows_with_time": len(active),
        "active_comparison_rows_without_time": sorted(
            set(comparison_rows) - set(timed_rows)
        ),
        "all_timed_wall_seconds": all_seconds,
        "all_timed_gpu_hours_as_one_gpu_per_case": all_seconds / 3600.0,
        "active_comparison_wall_seconds": active_seconds,
        "active_comparison_gpu_hours_as_one_gpu_per_case": active_seconds / 3600.0,
        "flagged_active_rows": len(flagged),
        "flagged_active_case_ids": [row["case_id"] for row in flagged],
        "flag_rate_on_timed_active_rows": len(flagged) / max(len(active), 1),
        "positive_rows_with_time": positives,
        "recall": (
            true_positives / positives if positives else None
        ),
        "recall_status": (
            "ESTIMABLE_ON_TIMED_SUBSET" if positives else "NOT_ESTIMABLE_NO_POSITIVE_IN_TIMED_SUBSET"
        ),
        "potential_avoided_active_gpu_hours": (
            active_seconds
            - sum(float(timed_rows[row["case_id"]]["elapsed_seconds"]) for row in flagged)
        ) / 3600.0,
        "claim_boundary": (
            "These are measured wall times only. The missing Mamba row remains unresolved, "
            "timed rows are mostly controls, and no complete Oracle GPU-saving or recall "
            "claim is made."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "timed_rows": payload["timed_rows"],
        "active_rows": payload["active_comparison_rows_with_time"],
        "gpu_hours": payload["all_timed_gpu_hours_as_one_gpu_per_case"],
    }))


if __name__ == "__main__":
    main()
