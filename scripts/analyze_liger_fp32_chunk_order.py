#!/usr/bin/env python3
"""Apply the frozen family-wise correction to the FP32 order experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.training_bias_profile import BRANCHES, holm_adjusted_p  # noqa: E402
from kernel_analyzer.mean_inference import mean_test_p, student_quantile
from kernel_analyzer.training_equivalence import (  # noqa: E402
    classify_fixed_suite_update_equivalence,
    fixed_suite_total_rms_from_joint_gram,
    profile_samples_from_joint_gram,
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
    recovered = {}
    for stage, item in source["profiles"].items():
        samples = profile_samples_from_joint_gram(item["suite"]["joint_gram"])
        recovered[stage] = samples
        for branch in BRANCHES:
            values = samples[branch]
            estimate = float(values.mean())
            standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
            raw[f"{stage}|{branch}"] = mean_test_p(
                estimate, standard_error, int(values.size - 1)
            )
    adjusted = holm_adjusted_p(raw)
    stages = {}
    for stage, item in source["profiles"].items():
        rows = {}
        for branch in BRANCHES:
            values = recovered[stage][branch]
            estimate = float(values.mean())
            standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
            critical = student_quantile(int(values.size - 1), 0.975)
            interval = [estimate - critical * standard_error, estimate + critical * standard_error]
            corrected = adjusted[f"{stage}|{branch}"]
            direction_matches = bool(item["population_inference"]["branches"][branch]["confirmation_direction_matches_calibration"])
            rows[branch] = {
                "estimate": estimate,
                "confidence_interval_95": interval,
                "standard_error": standard_error,
                "raw_p": raw[f"{stage}|{branch}"],
                "statistics_version": "mean-inference-v3",
                "holm_adjusted_p": corrected,
                "confirmation_direction_matches_calibration": direction_matches,
                "confirmed": bool(corrected < 0.05 and direction_matches),
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
        "update_equivalence": classify_fixed_suite_update_equivalence(
            simultaneous_intervals_from_joint_gram(
                source["profiles"]["ADAMW_UPDATE"]["suite"]["joint_gram"]
            ),
            MARGINS,
            total_rms=fixed_suite_total_rms_from_joint_gram(
                source["profiles"]["ADAMW_UPDATE"]["suite"]["joint_gram"]
            ),
            total_rms_margin=0.01,
        ),
        "claim_boundary": (
            "Reanalysis with conditional mean-inference-v3, not fresh prospective evidence. "
            "Frozen-state directional diagnostics do not establish a population guarantee "
            "or a long full-training loss consequence. The fixed-suite update decision "
            "includes the saved-coordinate total-energy envelope."
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
