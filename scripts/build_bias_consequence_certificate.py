#!/usr/bin/env python3
"""Build a v2 closed-loop consequence certificate."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.bias_consequence import BiasConsequenceTrace  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("consequence input must be a JSON object")
    return payload


def build(payload: dict[str, Any]) -> dict[str, Any]:
    if "first_bias_stage" in payload or "formation_point" in payload:
        raise ValueError("consequence input cannot carry formation-stage labels")
    step_ids = [str(x) for x in payload.get("step_ids", ())]
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
        trace.add(
            str(row.get("step_id", "")),
            local_increment=row.get("local_increment"),
            feedback_increment=row.get("feedback_increment"),
            actual_drift_increment=row.get("actual_drift_increment"),
            final_drift=row.get("final_drift"),
            recurrence_residual=row.get("recurrence_residual"),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        )
    return trace.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(_load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    opener = gzip.open if args.output.name.endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "case_id": result["case_id"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
