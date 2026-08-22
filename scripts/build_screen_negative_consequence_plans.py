#!/usr/bin/env python3
"""Bind the preregistered screen-negative sample to exact model/shape plans.

The audit sample is selected without T4/SEUP values.  This helper only joins
those rows to the already-bound multishape reach plans; it does not change the
sample or promote a screen result to a consequence verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/property/joint_bias_formation_v1/screen_negative_control_audit.json"
PLAN_ROOT = ROOT / "results/property/bias_formation/hotspot_search/multishape_reach_plans"
DEFAULT_OUT = ROOT / "results/property/joint_bias_formation_v1/negative_consequence_plans"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    audit = load(AUDIT)
    selected = audit["sample"]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    unresolved: list[dict[str, Any]] = []
    for row in selected:
        model = str(row["model"])
        seq = int(row["sequence_length"])
        source = PLAN_ROOT / f"{model}_seq{seq}.json"
        if not source.exists():
            unresolved.append({**row, "reason": "missing_exact_reach_plan", "plan": str(source)})
            continue
        plan = load(source)
        matches = [case for case in plan["cases"] if str(case["task_id"]) == str(row["task_id"])]
        if len(matches) != 1:
            unresolved.append({**row, "reason": f"expected_one_plan_row_got_{len(matches)}", "plan": str(source)})
            continue
        groups.setdefault((model, seq), []).append(matches[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for (model, seq), rows in sorted(groups.items()):
        unique: dict[str, dict[str, Any]] = {str(row["task_id"]): row for row in rows}
        payload = {
            "schema": "kernel-analyzer-screen-negative-consequence-plan-v1",
            "model": model,
            "sequence_length": seq,
            "sample_rule": audit["selection"],
            "uses_t4_or_seup": False,
            "cases": list(unique.values()),
            "claim_boundary": "Plan binding only; full 32-step consequence must still be run before any negative verdict.",
        }
        target = args.output_dir / f"{model}_seq{seq}.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append({"model": model, "sequence_length": seq, "cases": len(unique), "output": str(target)})
    summary = {
        "schema": "kernel-analyzer-screen-negative-consequence-plan-index-v1",
        "status": "COMPLETE_BOUNDED_BINDING",
        "sample_size": len(selected),
        "bound_rows": sum(item["cases"] for item in outputs),
        "groups": outputs,
        "unresolved": unresolved,
    }
    (args.output_dir / "index.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"sample_size": len(selected), "bound_rows": summary["bound_rows"], "groups": len(outputs), "unresolved": len(unresolved)}, sort_keys=True))


if __name__ == "__main__":
    main()
