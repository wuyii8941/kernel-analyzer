#!/usr/bin/env python3
"""Remove statically possible but dynamically zero carriers from a screen plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reach", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    reach = json.loads(args.reach.read_text())
    reached = {row["case_id"] for row in reach["cases"] if row["carrier_reached_any_state"]}
    cases = [row for row in plan["cases"] if row["case_id"] in reached]
    payload = dict(plan)
    payload.update({
        "schema": "kernel-analyzer-backward-dynamic-reach-filtered-plan-v1",
        "pre_filter_count": len(plan["cases"]), "post_filter_count": len(cases),
        "zero_carrier_cases_excluded": sorted(
            row["case_id"] for row in plan["cases"] if row["case_id"] not in reached
        ),
        "cases": cases,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "before": len(plan["cases"]),
                      "after": len(cases)}))


if __name__ == "__main__":
    main()
