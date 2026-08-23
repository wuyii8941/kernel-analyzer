#!/usr/bin/env python3
"""Verify and summarize the three inserted follow-up measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "priority_followup_summary.json")
    args = parser.parse_args()
    oracle = load(BASE / "oracle_baselines/frozen_evaluation_v2/comparison_v2.json")
    random = load(BASE / "carrier_distribution/random_null_v2/distribution.json")
    random_loss = load(BASE / "four_scale_arms/random_null_loss.json")
    adamw = load(BASE / "phi_three_stage_adamw.json")
    sgd = load(BASE / "phi_three_stage_reference.json")
    cohort = oracle["cohort"]
    if cohort["positive_rows"] != 3 or cohort["rows"] != 14:
        raise RuntimeError("the active Oracle cohort must contain 3 positives and 14 rows")
    if random["carrier_count"] != 12 or len(random["random_null_summary"]) != 5:
        raise RuntimeError("random null must cover all 12 carriers and 5 seeds")
    if random_loss["status"] != "COMPLETE_UNSEEN_FP32_EVALUATION":
        raise RuntimeError("random null loss is incomplete")
    if adamw["status"] != "COMPLETE_ORDERED_32_STATE_COMMON_STATE_ADAMW":
        raise RuntimeError("AdamW mapping is incomplete")
    adamw_update = adamw["stages"]["effective_update_error"]["coherence_curve"][-1]
    adamw_gradient = adamw["stages"]["parameter_gradient_error"]["coherence_curve"][-1]
    sgd_update = sgd["stages"]["effective_update_error"]["coherence_curve"][-1]
    summary = {
        "schema": "kernel-analyzer-priority-followup-summary-v1",
        "status": "COMPLETE_VERIFIED",
        "oracle": {
            "rows": cohort["rows"],
            "positive_rows": cohort["positive_rows"],
            "negative_rows": cohort["negative_rows"],
            "auroc": oracle["comparisons"]["prefix16_effective_update_persistence_oracle"]["auroc"],
            "operating_point": oracle["comparisons"]["prefix16_effective_update_persistence_oracle"]["threshold"],
            "excluded_legacy_rows": cohort["excluded_legacy_rows"],
        },
        "random_null": {
            "carrier_count": random["carrier_count"],
            "seed_count": len(random["random_null_summary"]),
            "seed_summary": random["random_null_summary"],
            "natural_final_norm_A": next(row["measurement"]["coherence_amplification"] for row in random["rows"] if row["carrier"] == "model.norm.weight"),
        },
        "random_null_loss": {
            "status": random_loss["status"],
            "seed": random_loss["seed"],
            "random_null_A": random_loss["random_null_A"],
            "absolute_loss_gap_random_minus_repair": random_loss["absolute_loss_gap_random_minus_repair"],
        },
        "adamw_mapping": {
            "status": adamw["status"],
            "gradient_A_32": adamw_gradient["coherence_amplification"],
            "effective_update_A_32": adamw_update["coherence_amplification"],
            "sgd_effective_update_A_32": sgd_update["coherence_amplification"],
            "claim_boundary": adamw["claim_boundary"],
        },
        "claim_boundary": "These additions repair the evaluation cohort and provide carrier-wide random-null and AdamW response measurements. They do not upgrade the bounded study to a universal all-operator property.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": summary["status"], "oracle_positive_rows": cohort["positive_rows"], "random_carriers": random["carrier_count"], "adamw_update_A": adamw_update["coherence_amplification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
