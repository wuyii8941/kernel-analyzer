#!/usr/bin/env python3
"""Split a frozen case plan into resource-bounded, definition-identical batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    plan = json.loads(args.plan.read_text())
    cases = plan["cases"]
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if str(case.get("case_id")) in requested]
        if {str(case.get("case_id")) for case in cases} != requested:
            raise ValueError("a requested case ID is absent from the plan")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(cases), args.batch_size):
        batch = dict(plan)
        batch["parent_plan"] = str(args.plan)
        batch["batch_index"] = start // args.batch_size
        batch["batch_count"] = (len(cases) + args.batch_size - 1) // args.batch_size
        batch["cases"] = cases[start:start + args.batch_size]
        target = args.output_dir / f"batch_{batch['batch_index']:02d}.json"
        target.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(target), "cases": len(batch["cases"])}))


if __name__ == "__main__":
    main()
