#!/usr/bin/env python3
"""Combine measured consequence wall times with the frozen Oracle operating point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--timed-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    timed = json.loads(args.timed_audit.read_text())
    rows = {str(row["case_id"]): row for row in timed["cases"]}
    require = lambda condition, message: (_ for _ in ()).throw(RuntimeError(message)) if not condition else None
    require(timed["status"] == "COMPLETE_12_CASES_WITH_WALL_TIMES", "timed audit incomplete")
    require(len(rows) == 12, "timed audit does not contain twelve unique cases")
    flagged = {
        str(row["case_id"])
        for row in comparison["rows"]
        if float(row["oracle_prefix16_local_A"]) > 1.0
    }
    all_seconds = sum(float(row["elapsed_seconds"]) for row in rows.values())
    flagged_seconds = sum(float(rows[case]["elapsed_seconds"]) for case in flagged)
    unflagged_seconds = all_seconds - flagged_seconds
    op = comparison["operating_point"]
    payload = {
        "schema": "kernel-analyzer-oracle-timed-efficiency-v2",
        "status": "COMPLETE_RETROSPECTIVE_WITH_MEASURED_WALL_TIMES",
        "comparison_sha256": sha256(args.comparison),
        "timed_audit_sha256": sha256(args.timed_audit),
        "cohort_size": len(rows),
        "flagged_case_count": len(flagged),
        "flagged_case_ids": sorted(flagged),
        "flag_rate": float(op["flag_rate"]),
        "source_persistence_recall": float(op["recall"]),
        "false_positive_rate": float(op["false_positive_rate"]),
        "miss_rate": float(op["miss_rate"]),
        "full_consequence_wall_seconds": all_seconds,
        "full_consequence_gpu_hours_one_gpu_per_case": all_seconds / 3600.0,
        "flagged_full_consequence_wall_seconds": flagged_seconds,
        "unflagged_full_consequence_wall_seconds": unflagged_seconds,
        "potential_avoided_full_consequence_gpu_hours": unflagged_seconds / 3600.0,
        "potential_avoided_fraction_of_full_consequence_gpu_hours": unflagged_seconds / max(all_seconds, 1e-30),
        "runtime_measurement": "Each value is process wall time on the recorded RTX A6000 host; screening overhead and compilation-cache reuse are not subtracted.",
        "claim_boundary": "This quantifies the cost of the frozen retrospective audit. It is not a prospective universal recall or a net end-to-end speedup claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gpu_hours": payload["full_consequence_gpu_hours_one_gpu_per_case"], "potential_avoided_gpu_hours": payload["potential_avoided_full_consequence_gpu_hours"]}))


if __name__ == "__main__":
    main()
