#!/usr/bin/env python3
"""Apply the frozen family-wise correction to the FP32 order experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.training_bias_profile import BRANCHES, holm_adjusted_p  # noqa: E402
from kernel_analyzer.training_equivalence import (  # noqa: E402
    classify_training_equivalence,
    simultaneous_intervals_from_joint_gram,
)


MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    raw = {}
    for stage, item in source["profiles"].items():
        for branch in BRANCHES:
            raw[f"{stage}|{branch}"] = float(
                item["population_inference"]["branches"][branch]["raw_studentized_signflip_p"]
            )
    adjusted = holm_adjusted_p(raw)
    stages = {}
    for stage, item in source["profiles"].items():
        rows = {}
        for branch in BRANCHES:
            measured = item["population_inference"]["branches"][branch]
            corrected = adjusted[f"{stage}|{branch}"]
            rows[branch] = {
                "estimate": measured["estimate"],
                "confidence_interval_95": measured["confidence_interval_95"],
                "raw_p": measured["raw_studentized_signflip_p"],
                "holm_adjusted_p": corrected,
                "confirmed": bool(corrected < 0.05 and measured["confirmation_direction_matches_calibration"]),
            }
        stages[stage] = rows
    payload = {
        "schema": "kernel-analyzer-liger-fp32-chunk-order-summary-v1",
        "status": "COMPLETE",
        "source_artifact": str(args.input),
        "comparison": "Both implementations use FP32; only dW chunk-addition order differs.",
        "source_prediction": source["source_prediction"],
        "multiplicity": {"method": "Holm family-wise correction", "test_count": len(raw)},
        "stages": stages,
        "update_equivalence": classify_training_equivalence(
            simultaneous_intervals_from_joint_gram(
                source["profiles"]["ADAMW_UPDATE"]["suite"]["joint_gram"]
            ),
            MARGINS,
        ),
        "claim_boundary": (
            "This confirms a short matched-state direction caused by FP32 addition order. "
            "It does not by itself establish a long full-training loss consequence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "confirmed": {
        stage: [name for name, row in branches.items() if row["confirmed"]]
        for stage, branches in stages.items()
    }}))


if __name__ == "__main__":
    main()
