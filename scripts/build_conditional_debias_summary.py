#!/usr/bin/env python3
"""Reduce one fixed-state conditional-debias campaign to compact evidence."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


ROLES = (
    "repair_local_residual",
    "candidate_local_effect_removed",
    "candidate_gradient_effect_removed",
    "candidate_sgd_update_effect_removed",
    "candidate_adamw_zero_update_effect_removed",
)


def build(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("status") != "COMPLETE_CONDITIONAL_DEBIAS_CONFIRMATION":
        raise ValueError("conditional-debias campaign is not a formal confirmation")
    if source.get("conditional_debias_enabled") is not True:
        raise ValueError("conditional-debias measurement is absent")
    if len(source.get("states", ())) < 16:
        raise ValueError("formal confirmation requires at least sixteen conditions")
    arms = source.get("arms", ())
    randomized = [arm for arm in arms if arm in {"ROUNDING_ONLY", "JOINT"}]
    if not randomized:
        raise ValueError("no randomized source-debiasing arm is present")
    result_arms: dict[str, Any] = {}
    for arm in randomized:
        aggregate = source["conditional_debias_summary"][arm]
        conditions = []
        for state in source["states"]:
            if state.get("sham_exact") is not True:
                raise ValueError("matched sham is not exact")
            layers = state["arms"][arm]["conditional_debias"]["layers"]
            if set(layers) != set(ROLES):
                raise ValueError("conditional layer set is incomplete")
            conditions.append({
                "state_id": state["state_id"],
                "repeats": state["arms"][arm]["repeats"],
                "layers": {
                    role: {
                        "status": layers[role]["status"],
                        "cross_repeat_ratio": layers[role]["cross_state_ratio"],
                        "bootstrap_95": [
                            layers[role]["bootstrap_lower"],
                            layers[role]["bootstrap_upper"],
                        ],
                        "estimand": layers[role]["estimand"],
                        "reference": layers[role]["reference"],
                    }
                    for role in ROLES
                },
            })
        result_arms[arm] = {
            "aggregate": aggregate,
            "conditions": conditions,
        }
    return {
        "schema": "kernel-analyzer-conditional-debias-summary-v1",
        "status": "COMPLETE",
        "candidate_id": source["candidate_id"],
        "architecture": source["architecture"],
        "carrier_parameter": source["carrier_parameter"],
        "condition_count": len(source["states"]),
        "arms": result_arms,
        "bindings": source["bindings"],
        "scientific_conclusion": (
            "A centered local repair residual and a nonzero candidate-minus-repair "
            "gradient/update effect are distinct claims. The latter identifies a "
            "systematic candidate effect removed under fixed states, but cannot certify "
            "absolute downstream repair bias without an exact downstream reference."
        ),
        "global_direction_used_as_gate": False,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fixed-state conditional debiasing",
        "",
        f"Candidate: `{summary['candidate_id']}`.",
        f"Fixed conditions: {summary['condition_count']}.",
        "",
        "Cross-condition direction is not a gate. Each row below is certified from "
        "independent repair draws at one unchanged training state.",
        "",
    ]
    for arm, value in summary["arms"].items():
        aggregate = value["aggregate"]
        lines.extend([
            f"## {arm}",
            "",
            f"Aggregate: `{aggregate['status']}`.",
            "",
            "| Layer estimand | Centered conditions | Biased conditions | Total |",
            "|---|---:|---:|---:|",
        ])
        for role, row in aggregate["roles"].items():
            counts = row["status_counts"]
            lines.append(
                f"| `{role}` | {counts.get('CONDITIONAL_CENTERED', 0)} | "
                f"{counts.get('CONDITIONAL_BIAS', 0)} | {row['condition_count']} |"
            )
        lines.extend([
            "",
            "The local repair residual is referenced to the exact declared source "
            "component. Gradient/update rows are candidate-minus-repair-ensemble "
            "effects; they are not an absolute certificate for the repaired arm.",
            "",
        ])
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = build(load(args.input))
    write(args.output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write(args.output_md, markdown(summary))


if __name__ == "__main__":
    main()
