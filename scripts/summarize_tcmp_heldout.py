#!/usr/bin/env python3
"""Render the concise, deduplicated TCMP held-out result summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for cell in ("llama32_3b_text128", "ministral3_3b_text128"):
        base = args.root / "heldout" / cell
        prediction = json.loads(
            (base / "lmhead_orbit_crossfit_retrospective.json").read_text()
        )
        consequence = json.loads((base / "lmhead_consequence32.json").read_text())
        repeated_null = json.loads(
            (base / "lmhead_repeated_orbit_null32.json").read_text()
        )
        orbit = prediction["certificate"]["statistics"][
            "tiling_conditional_orbit_mean"
        ]
        levels = consequence["statistics"]["levels"]
        nulls = repeated_null["nulls"]
        rows.append({
            "cell": cell,
            "generalization_stratum": "SEEN_IMPL_NEW_OPERANDS",
            "mathematical_family": "LM_HEAD_DX_GEMM_VJP",
            "predictor": prediction["prediction"],
            "predictor_evidence_role": "RETROSPECTIVE_STRICT_CROSSFIT",
            "predictor_amplification": orbit["coherence_amplification"],
            "predictor_signflip_p": orbit["sign_flip_null"]["one_sided_p"],
            "crossfit_mean_to_orbit_sigma": prediction["certificate"]["statistics"][
                "aggregate_crossfit_mean_to_orbit_sigma"
            ],
            "actual_amplification": levels["actual"]["coherence_amplification"],
            "actual_signflip_p": levels["actual"]["sign_flip_null"]["one_sided_p"],
            "local_amplification": levels["local"]["coherence_amplification"],
            "feedback_amplification": levels["feedback"]["coherence_amplification"],
            "final_master_drift_l2": consequence["final_master_drift_l2"],
            "telescoping_residual_l2": consequence["telescoping_residual_l2"],
            "matched_random_feedback_null": consequence["matched_random_feedback_null"],
            "repeated_real_orbit_null": {
                "kind": "NEW_NONDEFAULT_REDUCTION_ORBIT_EVERY_STEP",
                "seeds": repeated_null["null_seeds"],
                "mean_final_norm_over_natural": mean(
                    row["final_norm_over_natural"] for row in nulls
                ),
                "mean_final_cosine_with_natural": mean(
                    row["final_cosine_with_natural"] for row in nulls
                ),
                "mean_local_error_norm_over_natural": mean(
                    row["local_error_norm_ratio_mean"] for row in nulls
                ),
                "joint_drift_effective_rank": repeated_null[
                    "drift_effective_rank_participation_ratio"
                ],
            },
            "prediction_matches_observed_persistent_actual_drift": bool(
                orbit["above_sign_flip_95"] and levels["actual"]["above_sign_flip_95"]
            ),
        })
    ministral = args.root / "heldout" / "ministral3_3b_text128"
    yarn = json.loads((ministral / "yarn_fb_repair_probe.json").read_text())
    softmax = json.loads((ministral / "attention_softmax_fb_repair_probe.json").read_text())
    payload = {
        "schema": "kernel-analyzer-tcmp-heldout-summary-v1",
        "status": "PILOT_WITH_RETROSPECTIVE_PROTOCOL_CORRECTION",
        "lmhead_heldout_predictions": rows,
        "new_semantic_representatives": [
            {
                "case_id": yarn["case_id"], "verdict": "EXACT_SAFE_UNDER_PROTOCOL",
                "changed_parameter_gradient_count": yarn["changed_parameter_gradient_count"],
                "repeated_regions_in_denominator": yarn["repeated_regions_in_coverage_denominator"],
            },
            {
                "case_id": softmax["case_id"],
                "verdict": "LOCAL_DIFFERENCE_NOT_PARAMETER_REACHABLE_UNDER_PROTOCOL",
                "changed_parameter_gradient_count": softmax["changed_parameter_gradient_count"],
                "repeated_regions_in_denominator": softmax["repeated_regions_in_coverage_denominator"],
            },
        ],
        "deduplication": {
            "all_actual_invocations_remain_in_denominator": True,
            "deep_measurement_unit": "ONE_REPRESENTATIVE_PER_NEW_SEMANTIC_IMPLEMENTATION_PATTERN",
            "sequence_512_repeated_semantics_rescreened": False,
            "lmhead_two_models_count_as_one_mathematical_family": True,
        },
        "claim_boundary": (
            "Prediction preceded consequence, but the original pilot deviated from the frozen "
            "cross-fit and repeated-null protocol. Strict default-plus-eight 4+4 cross-fit and "
            "five-seed repeated real-orbit results are retrospective on these revealed models. "
            "They support a tiling-conditional orbit mean that survives schedule randomization; "
            "they do not constitute held-out NEW_IMPL confirmation. The new Ministral semantic "
            "patterns are NOT_APPLICABLE controls."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
