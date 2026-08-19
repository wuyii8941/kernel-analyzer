#!/usr/bin/env python3
"""Promote short-screen backward cells to independent 16+16 confirmation.

Promotion never requires local bias.  It retains every cell with appreciable
four-state gradient directionality and also the strongest cells in each
semantic family so a single global threshold does not erase a mechanism.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--carriers", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-ratio", type=float, default=0.10)
    parser.add_argument("--top-per-family", type=int, default=2)
    parser.add_argument("--task-id", action="append", default=None,
                        help="explicit task IDs to confirm; bypasses broad promotion")
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text())
    carriers = json.loads(args.carriers.read_text())
    cells = {row["cell_id"]: row for row in carriers["cells"] if row["model"] == args.model}
    candidates = []
    for row in screen["cases"]:
        cell = cells[row["case_id"]]
        local = row["layers"]["LOCAL_ENDPOINT"]
        gradient = row["layers"]["PARAMETER_GRADIENT"]
        candidates.append({
            "case_id": row["case_id"], "task_id": row["task_id"],
            "carrier": row["carrier"], "family": cell["family"],
            "depth_stratum": cell["depth_stratum"],
            "member_count": cell["member_count"],
            "screen_local_ratio": local["cross_state_ratio"],
            "screen_gradient_ratio": gradient["cross_state_ratio"],
            "screen_transport_gain": gradient["cross_state_ratio"] - local["cross_state_ratio"],
            "screen_gradient_energy": gradient["average_state_energy"],
        })
    if args.task_id:
        requested = set(args.task_id)
        missing = requested - {row["task_id"] for row in candidates}
        if missing:
            raise ValueError(f"requested task IDs absent from screen: {sorted(missing)}")
        promoted = {row["case_id"] for row in candidates if row["task_id"] in requested}
    else:
        promoted = {row["case_id"] for row in candidates
                    if row["screen_gradient_ratio"] >= args.gradient_ratio}
    by_family = collections.defaultdict(list)
    for row in candidates:
        by_family[row["family"]].append(row)
    if not args.task_id:
        for rows in by_family.values():
            rows.sort(key=lambda row: (row["screen_gradient_ratio"], row["screen_transport_gain"]),
                      reverse=True)
            promoted.update(row["case_id"] for row in rows[:args.top_per_family])
    selected = [row for row in candidates if row["case_id"] in promoted]
    selected.sort(key=lambda row: row["screen_gradient_ratio"], reverse=True)
    payload = {
        "schema": "kernel-analyzer-backward-confirmation-plan-v1",
        "model": args.model,
        "selection_population": "FIRST_FOUR_OPEN_LOOP_STATES",
        "confirmation_population": "DISJOINT_LAST_SIXTEEN_OPEN_LOOP_STATES",
        "local_direction_required": False,
        "selection_rule": {
            "gradient_ratio_at_least": args.gradient_ratio,
            "plus_top_per_semantic_family": args.top_per_family,
        },
        "screened_case_count": len(candidates),
        "cases": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "screened": len(candidates),
                      "promoted": len(selected)}))


if __name__ == "__main__":
    main()
