#!/usr/bin/env python3
"""Reconcile timed replay metadata against the actual output files.

The original timing runner wrote a 12-case status even when a filtered run
contained fewer rows.  This small checker counts the frozen case IDs and
rewrites the status fail-closed; it never treats an old scientific JSON as a
new wall-clock measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timed", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, default=ROOT / "results/property/joint_bias_formation_v1/oracle_baselines/comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    timed = json.loads(args.timed.read_text(encoding="utf-8"))
    expected = json.loads(args.comparison.read_text(encoding="utf-8"))["rows"]
    expected_ids = {str(row["case_id"]) for row in expected}
    rows: list[dict[str, Any]] = []
    for row in timed.get("cases", []):
        output = Path(str(row["output"]))
        scientific_status = None
        steps = None
        if output.is_file():
            try:
                artifact = json.loads(output.read_text(encoding="utf-8"))
                scientific_status = artifact.get("status")
                steps = artifact.get("step_count", artifact.get("steps"))
            except Exception:
                scientific_status = "MALFORMED_OUTPUT"
        row = dict(row)
        row["scientific_status"] = scientific_status
        row["scientific_steps"] = steps
        rows.append(row)
    complete_ids = {
        str(row["case_id"])
        for row in rows
        if row.get("return_code") == 0
        and row.get("result_status") == "COMPLETE"
        and row.get("scientific_status") == "COMPLETE"
        and int(row.get("scientific_steps") or 0) == 32
    }
    missing = sorted(expected_ids - complete_ids)
    status = (
        f"COMPLETE_{len(complete_ids)}_OF_{len(expected_ids)}_TIMED_CASES"
        if not missing else f"PARTIAL_TIMED_{len(complete_ids)}_OF_{len(expected_ids)}"
    )
    payload = {
        "schema": "kernel-analyzer-reconciled-timed-negative-consequence-v1",
        "status": status,
        "expected_case_count": len(expected_ids),
        "complete_case_count": len(complete_ids),
        "complete_case_ids": sorted(complete_ids),
        "missing_case_ids": missing,
        "sum_case_wall_seconds": sum(float(row.get("elapsed_seconds", 0.0)) for row in rows if str(row["case_id"]) in complete_ids),
        "rows": rows,
        "claim_boundary": "Only rows with a real COMPLETE 32-step output and a measured process wall time are counted as timed. Missing rows remain unresolved.",
        "source": str(args.timed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "complete": len(complete_ids), "expected": len(expected_ids), "missing": missing}, sort_keys=True))


if __name__ == "__main__":
    main()
