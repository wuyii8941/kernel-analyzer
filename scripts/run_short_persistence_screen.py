#!/usr/bin/env python3
"""Run the generic short persistence screen on a compact vector input bank.

Input schema:

{
  "schema": "kernel-analyzer-short-screen-input-v1",
  "cases": {"case_id": {"vectors": [[...], ...]}}
}

The input is normally produced by a runtime capture adapter.  This CLI is
also useful for CPU synthetic tests; it never requires model execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_analyzer.short_persistence import SharedShortPersistenceScreen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=20260822)
    parser.add_argument("--null-draws", type=int, default=2000)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema") != "kernel-analyzer-short-screen-input-v1":
        raise ValueError("unexpected short-screen input schema")
    cases = payload.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("short-screen input must contain a nonempty cases mapping")
    lengths = {len(row["vectors"]) for row in cases.values()}
    if len(lengths) != 1:
        raise ValueError("all cases must use the same ordered state count")
    screen = SharedShortPersistenceScreen(
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        expected_steps=next(iter(lengths)),
        null_draws=args.null_draws,
    )
    for case_id, row in cases.items():
        for vector in row["vectors"]:
            screen.add(case_id, vector)
    result = screen.finalize()
    result["input_schema"] = payload["schema"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "status": result["status"]}))


if __name__ == "__main__":
    main()
