#!/usr/bin/env python3
"""Freeze a result-blind held-out split for sample completion.

The split is chosen from the frozen roster by model group, before any new
2/8/16/32-step measurement.  The expectation bucket is only a pre-registered
screening expectation; it is never a scientific label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / "results/property/sample_completion_v1/roster.json"
OUT = ROOT / "results/property/sample_completion_v1/heldout_roster.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_for(case_id: str) -> str:
    for prefix in ("qwen_", "phi4_", "deepseek8b_", "mamba_"):
        if case_id.startswith(prefix):
            return prefix[:-1]
    raise ValueError(f"cannot assign case to frozen model group: {case_id}")


def expectation(implementation_family: str) -> str:
    # Frozen before new measurements.  These are screening expectations only.
    if implementation_family in {"ATTENTION_BACKWARD", "STATE_SPACE_BACKWARD"}:
        return "ESCALATE_CANDIDATE"
    return "NO_ESCALATION_CANDIDATE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, default=ROSTER)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    roster = load(args.roster)
    search = roster["search_units"]
    # Whole groups are held out so that no model/operand family leaks from
    # development into confirmation.  This is a mechanical, pre-measurement
    # split, not a result-driven selection.
    heldout_groups = {"deepseek8b", "mamba"}
    rows = []
    for row in search:
        group = group_for(str(row["case_id"]))
        if group not in heldout_groups:
            continue
        item = {
            "case_id": row["case_id"],
            "model_group": group,
            "model_family": row.get("model_family"),
            "implementation_family": row["implementation_family"],
            "expectation_bucket": expectation(str(row["implementation_family"])),
            "scientific_label": None,
        }
        rows.append(item)
    rows.sort(key=lambda item: item["case_id"])
    if len(rows) < 8:
        raise RuntimeError(f"held-out split has only {len(rows)} units")
    if len({row["model_group"] for row in rows}) < 2:
        raise RuntimeError("held-out split must include two model groups")
    if len({row["implementation_family"] for row in rows}) < 2:
        raise RuntimeError("held-out split must include two implementation families")
    buckets = {row["expectation_bucket"] for row in rows}
    if buckets != {"ESCALATE_CANDIDATE", "NO_ESCALATION_CANDIDATE"}:
        raise RuntimeError("held-out split must contain both expectation buckets")
    result = {
        "schema": "kernel-analyzer-sample-completion-heldout-roster-v1",
        "status": "FROZEN_BEFORE_MEASUREMENT",
        "source_roster": str(args.roster.relative_to(ROOT)),
        "selection_rule": "hold out complete deepseek8b and mamba search groups; do not inspect measurements",
        "heldout_groups": sorted(heldout_groups),
        "required_properties": {
            "minimum_units": 8,
            "model_group_count": len({row["model_group"] for row in rows}),
            "implementation_family_count": len({row["implementation_family"] for row in rows}),
            "has_both_expectation_buckets": True,
        },
        "scientific_labels_frozen": False,
        "rows": rows,
        "claim_boundary": "Expectation buckets are frozen screening inputs, not observed labels; all scientific labels remain null until the 32-step run.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "units": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
