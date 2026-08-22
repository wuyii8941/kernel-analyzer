#!/usr/bin/env python3
"""Summarize operator, gradient, and update coherence on common 32-step runs.

This is deliberately a small reporting tool.  It does not change labels or
search for a favourable horizon: it reads the precomputed horizon-32 values
from each stage and writes one table for the bias -> gradient -> update story.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def horizon_value(stage: dict, horizon: int = 32) -> dict:
    curve = stage["coherence_curve"]
    rows = [row for row in curve if int(row["horizon"]) == horizon]
    if len(rows) != 1:
        raise ValueError(f"expected one horizon={horizon} row, found {len(rows)}")
    row = rows[0]
    return {
        "coherence_amplification": float(row["coherence_amplification"]),
        "path_rms_l2": float(row["path_rms_l2"]),
        "resultant_l2": float(row["resultant_l2"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = []
    for path in args.input:
        payload = json.loads(path.read_text())
        stages = payload["stages"]
        cases.append(
            {
                "case_id": payload["case_id"],
                "status": payload["status"],
                "state_count": int(payload["state_count"]),
                "source_file": str(path),
                "source_sha256": file_sha256(path),
                "claim_boundary": payload["claim_boundary"],
                "horizon": 32,
                "operator_output_error": horizon_value(stages["operator_output_error"]),
                "parameter_gradient_error": horizon_value(stages["parameter_gradient_error"]),
                "effective_update_error": horizon_value(stages["effective_update_error"]),
            }
        )

    payload = {
        "schema": "kernel-analyzer-three-stage-summary-v1",
        "status": "COMPLETE_ORDERED_32_STATE_SUMMARY",
        "measurement": "Each row uses the same ordered states and the same repair-reference trajectory. The three stages are reported separately; no trajectory label is used to create a formation label.",
        "cases": cases,
        "claim_boundary": "This summary establishes where directionality becomes visible for these three declared carriers. It is not a universal property and is not a full-parameter training result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
