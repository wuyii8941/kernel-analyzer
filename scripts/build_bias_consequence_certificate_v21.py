#!/usr/bin/env python3
"""Build a fail-closed four-counterfactual consequence certificate."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.bias_consequence_v21 import BiasConsequenceTrace  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("consequence input must be a JSON object")
    return payload


def build(payload: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"first_bias_stage", "first_confirmed_bias_stage", "formation_point"}
    if forbidden & payload.keys():
        raise ValueError("consequence input cannot carry formation-stage labels")
    step_ids = [str(value) for value in payload.get("step_ids", ())]
    if not step_ids:
        raise ValueError("step_ids are required")
    trace = BiasConsequenceTrace(
        str(payload.get("case_id", "")),
        step_ids,
        float(payload.get("recurrence_tolerance", 1e-6)),
    )
    rows = payload.get("rows", ())
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each consequence row must be an object")
        if "recurrence_residual" in row:
            raise ValueError("recurrence residual must be computed by the library")
        trace.add(
            str(row.get("step_id", "")),
            candidate_at_candidate_state=row.get("candidate_at_candidate_state"),
            repair_at_candidate_state=row.get("repair_at_candidate_state"),
            candidate_at_repair_state=row.get("candidate_at_repair_state"),
            repair_at_repair_state=row.get("repair_at_repair_state"),
            drift_before=row.get("drift_before"),
            drift_after=row.get("drift_after"),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        )
    return trace.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    opener = gzip.open if args.output.name.endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps({
        "case_id": result["case_id"],
        "output": str(args.output),
        "status": result["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
