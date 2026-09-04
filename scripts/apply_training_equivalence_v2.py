#!/usr/bin/env python3
"""Apply the corrected fixed-suite update-equivalence certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.training_equivalence import (  # noqa: E402
    BRANCHES,
    classify_fixed_suite_update_equivalence,
    fixed_suite_total_rms_from_joint_gram,
    simultaneous_intervals_from_joint_gram,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    protocol = json.loads(args.protocol.read_text())
    margins = {name: float(protocol["engineering_margins"][name]) for name in BRANCHES}
    total_rms_margin = float(protocol["engineering_margins"]["full_update_rms"])
    cases = {}
    for case_id, item in source["cases"].items():
        update = item.get("stages", {}).get("ADAMW_UPDATE")
        if not update or "branches" not in update:
            cases[case_id] = classify_fixed_suite_update_equivalence(
                {name: [0.0, 0.0] for name in BRANCHES},
                margins,
                total_rms=0.0,
                total_rms_margin=total_rms_margin,
                valid_protocol=False,
            )
            continue
        joint_gram = update.get("suite", {}).get("joint_gram")
        if joint_gram is None:
            cases[case_id] = classify_fixed_suite_update_equivalence(
                {name: [0.0, 0.0] for name in BRANCHES},
                margins,
                total_rms=0.0,
                total_rms_margin=total_rms_margin,
                valid_protocol=False,
            )
            cases[case_id]["reason"] = "JOINT_GRAM_MISSING"
            continue
        intervals = simultaneous_intervals_from_joint_gram(joint_gram)
        raw_path = args.raw_dir / f"{case_id}.json"
        if not raw_path.is_file():
            cases[case_id] = classify_fixed_suite_update_equivalence(
                {name: [0.0, 0.0] for name in BRANCHES},
                margins,
                total_rms=0.0,
                total_rms_margin=total_rms_margin,
                valid_protocol=False,
            )
            cases[case_id]["reason"] = "RAW_MEASUREMENT_MISSING"
            continue
        raw = json.loads(raw_path.read_text())
        raw_views = raw.get("stages", {}).get("ADAMW_UPDATE", {})
        if not raw_views:
            cases[case_id] = classify_fixed_suite_update_equivalence(
                {name: [0.0, 0.0] for name in BRANCHES},
                margins,
                total_rms=0.0,
                total_rms_margin=total_rms_margin,
                valid_protocol=False,
            )
            cases[case_id]["reason"] = "RAW_UPDATE_MEASUREMENT_MISSING"
            continue
        rms_by_view = {
            view: fixed_suite_total_rms_from_joint_gram(wrapped["profile"]["suite"]["joint_gram"])
            for view, wrapped in raw_views.items()
        }
        exact_view = set(raw_views) == {"EXACT"}
        expected_sketches = {
            "SKETCH_SEED_20260831", "SKETCH_SEED_20260861", "SKETCH_SEED_20260891",
        }
        if not exact_view and set(raw_views) != expected_sketches:
            cases[case_id] = classify_fixed_suite_update_equivalence(
                {name: [0.0, 0.0] for name in BRANCHES},
                margins,
                total_rms=0.0,
                total_rms_margin=total_rms_margin,
                valid_protocol=False,
            )
            cases[case_id]["reason"] = "ALL_COORDINATE_SUMMARY_SET_INCOMPLETE"
            continue
        total_rms = max(rms_by_view.values())
        cases[case_id] = classify_fixed_suite_update_equivalence(
            intervals,
            margins,
            total_rms=total_rms,
            total_rms_margin=total_rms_margin,
            consequence_status="NOT_DECLARED",
            exact_identity_verified=exact_view and total_rms == 0.0,
        )
        cases[case_id]["full_update_rms_by_view"] = rms_by_view
        cases[case_id]["full_update_measurement"] = (
            "EXACT_VECTOR" if exact_view else "MAX_OF_THREE_FROZEN_COUNT_SKETCHES"
        )
        cases[case_id]["original_coordinate_count"] = int(
            next(iter(raw_views.values()))["coordinate_count"]
        )
        cases[case_id]["stored_coordinate_count_by_view"] = {
            view: int(wrapped["profile"]["suite"]["coordinate_count"])
            for view, wrapped in raw_views.items()
        }
        cases[case_id]["source_update_result"] = item.get("primary_update_result")
        cases[case_id]["interval_rule"] = (
            "95% Bonferroni simultaneous intervals across the three frozen update effects"
        )

    counts = {}
    for item in cases.values():
        counts[item["decision"]] = counts.get(item["decision"], 0) + 1
    payload = {
        "schema": "kernel-analyzer-fixed-suite-update-equivalence-decisions-v2",
        "status": "COMPLETE",
        "source_summary": str(args.summary),
        "raw_measurement_directory": str(args.raw_dir),
        "protocol": str(args.protocol),
        "margins": {**margins, "full_update_rms": total_rms_margin},
        "decision_counts": counts,
        "cases": cases,
        "claim_boundary": (
            "These are fixed-suite update decisions, not random-state-population "
            "or training-quality equivalence claims. This post-reveal method correction "
            "adds a mandatory all-coordinate update-energy envelope. For large vectors, "
            "the result uses the maximum of three frozen CountSketch summaries and is not "
            "described as bitwise identity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "decisions": counts}))


if __name__ == "__main__":
    main()
