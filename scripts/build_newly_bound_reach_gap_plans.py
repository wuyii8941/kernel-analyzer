#!/usr/bin/env python3
"""Emit exact F+B cases recovered by a more specific parameter binding."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_formation/hotspot_search"
BOUND = BASE / "multishape_backward_carriers.json"
OUT = BASE / "newly_bound_reach_gap_plans"


def main() -> None:
    binding = json.loads(BOUND.read_text())
    emitted = 0
    for model in ("qwen", "phi4", "deepseek8b", "mamba"):
        for sequence_length in (64, 128, 256):
            existing_path = BASE / f"multishape_reach_plans/{model}_seq{sequence_length}.json"
            existing = json.loads(existing_path.read_text())
            existing_tasks = {str(row["task_id"]) for row in existing["cases"]}
            cases = []
            for cell in binding["cells"]:
                if cell["model"] != model or int(cell["sequence_length"]) != sequence_length:
                    continue
                task_id = str(cell["representative"]["task_id"])
                carrier = cell.get("nearest_carrier")
                if (
                    carrier is None
                    or task_id in existing_tasks
                    or cell["capture_boundary"] != "EXACT_AOT_ENDPOINT"
                ):
                    continue
                cases.append({
                    "case_id": cell["cell_id"],
                    "task_id": task_id,
                    "carrier": carrier["name"],
                    "carrier_binding": "DEEPEST_EXACT_MODULE_STACK_PARAMETER_BINDING",
                    "family": cell["family"],
                    "depth_stratum": cell["depth_stratum"],
                    "member_count": cell["member_count"],
                })
            target = OUT / f"{model}_seq{sequence_length}.json"
            if cases:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({
                    "schema": "kernel-analyzer-newly-bound-reach-gap-plan-v1",
                    "model": model,
                    "sequence_length": sequence_length,
                    "selection_uses_candidate_values_or_historical_verdict": False,
                    "cases": cases,
                }, indent=2, sort_keys=True) + "\n")
                emitted += len(cases)
                print(f"{model}:seq{sequence_length}: {len(cases)}")
    print(json.dumps({"output": str(OUT), "cases": emitted}))


if __name__ == "__main__":
    main()
