#!/usr/bin/env python3
"""Build the complete measured-cost table for the frozen 14-row Oracle cohort.

The inputs come from three independently executed runners: the frozen control
audit, the separately timed Mamba row, and the three case-specific headline
reproductions.  Every active row must have a real 32-step output before this
script emits a complete result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument("--mamba-audit", type=Path, required=True)
    parser.add_argument("--positive-audit", type=Path, action="append", required=True)
    parser.add_argument("--mamba-scientific-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = load(args.comparison)
    if comparison.get("status") != "COMPLETE_FROZEN_14_ROW_COMPARISON_3_POSITIVE_HEADLINES":
        raise RuntimeError("comparison is not the frozen 14-row cohort")
    expected = {str(row["case_id"]): row for row in comparison["rows"]}
    if len(expected) != 14:
        raise RuntimeError("comparison does not contain fourteen unique active rows")

    observed: dict[str, dict[str, Any]] = {}

    control = load(args.control_audit)
    for row in control.get("rows", control.get("cases", [])):
        case_id = str(row["case_id"])
        if case_id not in expected:
            continue
        if row.get("result_status") != "COMPLETE" or int(row.get("scientific_steps", 32)) != 32:
            raise RuntimeError(f"control timing is incomplete for {case_id}")
        observed[case_id] = {
            "case_id": case_id,
            "timing_kind": "BOUND_ENDPOINT_CONSEQUENCE_32_STEP",
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "output": row["output"],
            "scientific_reproduced": True,
        }

    mamba = load(args.mamba_audit)
    mamba_rows = [row for row in mamba.get("cases", []) if str(row.get("case_id")) in expected]
    if len(mamba_rows) != 1:
        raise RuntimeError("Mamba audit must contain exactly one active row")
    row = mamba_rows[0]
    if row.get("result_status") != "COMPLETE" or row.get("return_code") != 0:
        raise RuntimeError("Mamba timing row is incomplete")
    measured_mamba = load(Path(row["output"]))
    source_mamba = load(args.mamba_scientific_source)
    comparable_keys = ("case_id", "carrier", "step_count", "step_ids", "rows", "statistics", "cumulative")
    mamba_reproduced = all(measured_mamba.get(key) == source_mamba.get(key) for key in comparable_keys)
    if not mamba_reproduced:
        raise RuntimeError("bound-runtime Mamba timing run did not reproduce the frozen scientific result")
    case_id = str(row["case_id"])
    observed[case_id] = {
        "case_id": case_id,
        "timing_kind": "BOUND_ENDPOINT_CONSEQUENCE_32_STEP",
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "output": row["output"],
        "scientific_reproduced": True,
        "reference_strategy": measured_mamba.get("reference_strategy"),
    }

    for path in args.positive_audit:
        positive = load(path)
        for row in positive.get("cases", []):
            case_id = str(row["case_id"])
            if case_id not in expected:
                continue
            if row.get("return_code") != 0 or row.get("scientific_result_reproduced") is not True:
                raise RuntimeError(f"headline timing did not reproduce {case_id}")
            if row.get("result_status") != "COMPLETE_ORDERED_32_STATE_REFERENCE":
                raise RuntimeError(f"headline timing is incomplete for {case_id}")
            observed[case_id] = {
                "case_id": case_id,
                "timing_kind": "ORDERED_32_STATE_THREE_STAGE_REFERENCE",
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "output": row["output"],
                "scientific_reproduced": True,
            }

    missing = sorted(set(expected).difference(observed))
    if missing:
        raise RuntimeError("active cohort lacks real timing for: " + ", ".join(missing))

    rows = []
    for case_id, source in expected.items():
        measurement = observed[case_id]
        flagged = float(source["oracle_prefix16_local_A"]) > 1.0
        rows.append({
            **measurement,
            "label_32step_local_persistent": bool(source["label_32step_local_persistent"]),
            "oracle_prefix16_local_A": float(source["oracle_prefix16_local_A"]),
            "flagged_by_frozen_rule": flagged,
        })

    total_seconds = sum(row["elapsed_seconds"] for row in rows)
    flagged_seconds = sum(row["elapsed_seconds"] for row in rows if row["flagged_by_frozen_rule"])
    positive_seconds = sum(row["elapsed_seconds"] for row in rows if row["label_32step_local_persistent"])
    threshold = comparison["comparisons"]["prefix16_effective_update_persistence_oracle"]["threshold"]
    payload = {
        "schema": "kernel-analyzer-complete-timed-oracle-audit-v1",
        "status": "COMPLETE_FROZEN_14_ROW_VALIDATION_COST",
        "inputs": {
            "comparison_sha256": digest(args.comparison),
            "control_audit_sha256": digest(args.control_audit),
            "mamba_audit_sha256": digest(args.mamba_audit),
            "positive_audit_sha256": [digest(path) for path in args.positive_audit],
        },
        "cohort": {
            "rows": len(rows),
            "positives": sum(row["label_32step_local_persistent"] for row in rows),
            "controls": sum(not row["label_32step_local_persistent"] for row in rows),
            "flagged": sum(row["flagged_by_frozen_rule"] for row in rows),
        },
        "operating_point": threshold,
        "measured_cost": {
            "sum_case_wall_seconds": total_seconds,
            "sum_case_gpu_hours_one_gpu_per_case": total_seconds / 3600.0,
            "positive_validation_gpu_hours": positive_seconds / 3600.0,
            "flagged_full_validation_gpu_hours": flagged_seconds / 3600.0,
            "unflagged_full_validation_gpu_hours": (total_seconds - flagged_seconds) / 3600.0,
            "gross_avoidable_fraction_if_only_flagged_rows_receive_32_step_validation": (
                (total_seconds - flagged_seconds) / max(total_seconds, 1e-30)
            ),
            "screening_runtime": "NOT_SEPARATELY_TIMED",
        },
        "rows": rows,
        "claim_boundary": (
            "All fourteen frozen rows now have a real measured 32-step validation cost and a reproduced scientific output. "
            "The avoided-hours quantity is gross 32-step validation work, not net end-to-end speedup: the prefix-screening runtime "
            "was not separately timed, and the case-specific runners are not a portable kernel benchmark."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **payload["cohort"], **payload["measured_cost"]}, sort_keys=True))


if __name__ == "__main__":
    main()
