#!/usr/bin/env python3
"""Fill the v4 severity/tolerance files with only available evidence.

This intentionally leaves missing fields as ABSTAIN.  It does not turn the
old 15-row retrospective cohort or the one Gemma negative into a universal
performance estimate.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/direct_persistence_v4"
OLD = ROOT / "results/property/joint_bias_formation_v1/oracle_repair_v3/same_optimizer_oracle_v3.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cohort = load(OLD)
    gemma = load(OUT / "heldout_gemma_confirmation.json")
    gemma_row = gemma["rows"][0]

    severity = load(OUT / "severity.json")
    severity["status"] = "PARTIAL_AVAILABLE_CONSEQUENCE_MEASUREMENTS"
    severity["rows"] = [{
        "case_id": gemma_row["case_id"],
        "role": "NEW_IMPL",
        "direct_A32": gemma_row["confirmation"]["A32_local"],
        "actual_A32": gemma_row["confirmation"]["A32_actual"],
        "feedback_A32": gemma_row["confirmation"]["A32_feedback"],
        "final_parameter_distance": gemma_row["confirmation"]["final_drift_l2"],
        "observed_consequence": "feedback-dominated trajectory separation",
        "severity_proxy_status": "ABSTAIN_MISSING_PARAMETER_NORM_NORMAL_UPDATE_AND_LOSS_PROJECTION",
    }]
    severity["retrospective_rows_available"] = len(cohort.get("rows", []))
    severity["claim_boundary"] = (
        "Only direct/actual resultants and final distance are available for the "
        "Gemma held-out row. Parameter-normalized severity, normal-update scale, "
        "loss projection and runtime remain ABSTAIN."
    )
    (OUT / "severity.json").write_text(json.dumps(severity, indent=2, sort_keys=True) + "\n")

    tolerance = load(OUT / "tolerance_comparison.json")
    tolerance["status"] = "PARTIAL_AVAILABLE_RMS_BASELINE"
    tolerance["available_retrospective_baselines"] = {
        "prefix16_local_A": cohort.get("comparisons", {}).get("prefix16_local_A"),
        "prefix16_local_rms": cohort.get("comparisons", {}).get("prefix16_local_rms"),
        "frozen_fail_closed_threshold_score_gt_1": cohort.get("comparisons", {}).get("frozen_fail_closed_threshold_score_gt_1"),
    }
    tolerance["external_new_impl"] = {
        "case_id": gemma_row["case_id"],
        "direct_persistence_verdict": gemma_row["confirmation"]["verdict"],
        "local_A16": gemma_row["short_screen"]["A16"],
        "local_A32": gemma_row["confirmation"]["A32_local"],
        "result": "negative_for_direct_persistence",
    }
    tolerance["missing_baselines"] = [
        "max absolute error", "relative L2", "ULP", "rtol/atol sweep",
        "output RMS", "gradient RMS", "update RMS on the independent Gemma row",
    ]
    tolerance["claim_boundary"] = (
        "The current evidence compares persistence directionality with the old "
        "local RMS retrospective baseline. A complete tolerance comparison still "
        "requires the same frozen held-out pool to provide all listed metrics."
    )
    (OUT / "tolerance_comparison.json").write_text(json.dumps(tolerance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PARTIAL_AVAILABLE_METRICS", "gemma_rows": 1, "retrospective_rows": len(cohort.get("rows", []))}, sort_keys=True))


if __name__ == "__main__":
    main()
