#!/usr/bin/env python3
"""Apply frozen engineering equivalence ranges to measured update profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.training_equivalence import (  # noqa: E402
    BRANCHES,
    classify_training_equivalence,
    simultaneous_intervals_from_joint_gram,
)


MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    cases = {}
    for case_id, item in source["cases"].items():
        update = item.get("stages", {}).get("ADAMW_UPDATE")
        if not update or "branches" not in update:
            cases[case_id] = classify_training_equivalence(
                {name: [0.0, 0.0] for name in BRANCHES}, MARGINS,
                valid_protocol=False,
            )
            continue
        suite = update.get("suite", {})
        joint_gram = suite.get("joint_gram")
        if joint_gram is None:
            cases[case_id] = classify_training_equivalence(
                {name: [0.0, 0.0] for name in BRANCHES}, MARGINS,
                valid_protocol=False,
            )
            cases[case_id]["reason"] = "JOINT_GRAM_MISSING"
            continue
        intervals = simultaneous_intervals_from_joint_gram(joint_gram)
        cases[case_id] = classify_training_equivalence(intervals, MARGINS)
        cases[case_id]["source_update_result"] = item.get("primary_update_result")
        cases[case_id]["interval_rule"] = (
            "95% Bonferroni simultaneous intervals across the three frozen update effects"
        )
    payload = {
        "schema": "kernel-analyzer-training-equivalence-decisions-v1",
        "status": "COMPLETE",
        "source_summary": str(args.summary),
        "margins": MARGINS,
        "cases": cases,
        "claim_boundary": (
            "These are decisions under frozen engineering ranges for this benchmark. "
            "The ranges are not universal LLM-training safety thresholds."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    counts = {}
    for item in cases.values():
        counts[item["decision"]] = counts.get(item["decision"], 0) + 1
    print(json.dumps({"status": payload["status"], "decisions": counts}))


if __name__ == "__main__":
    main()
