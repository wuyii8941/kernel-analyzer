#!/usr/bin/env python3
"""Render the concise, deduplicated TCMP held-out result summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for cell in ("llama32_3b_text128", "ministral3_3b_text128"):
        base = args.root / "heldout" / cell
        prediction = json.loads((base / "lmhead_orbit_prediction.json").read_text())
        consequence = json.loads((base / "lmhead_consequence32.json").read_text())
        orbit = prediction["certificate"]["statistics"]["orbit_mean"]
        levels = consequence["statistics"]["levels"]
        rows.append({
            "cell": cell,
            "generalization_stratum": "SEEN_IMPL_NEW_OPERANDS",
            "mathematical_family": "LM_HEAD_DX_GEMM_VJP",
            "predictor": prediction["prediction"],
            "predictor_amplification": orbit["coherence_amplification"],
            "predictor_signflip_p": orbit["sign_flip_null"]["one_sided_p"],
            "actual_amplification": levels["actual"]["coherence_amplification"],
            "actual_signflip_p": levels["actual"]["sign_flip_null"]["one_sided_p"],
            "local_amplification": levels["local"]["coherence_amplification"],
            "feedback_amplification": levels["feedback"]["coherence_amplification"],
            "final_master_drift_l2": consequence["final_master_drift_l2"],
            "telescoping_residual_l2": consequence["telescoping_residual_l2"],
            "matched_random_feedback_null": consequence["matched_random_feedback_null"],
            "prediction_matches_observed_persistent_actual_drift": bool(
                orbit["above_sign_flip_95"] and levels["actual"]["above_sign_flip_95"]
            ),
        })
    ministral = args.root / "heldout" / "ministral3_3b_text128"
    yarn = json.loads((ministral / "yarn_fb_repair_probe.json").read_text())
    softmax = json.loads((ministral / "attention_softmax_fb_repair_probe.json").read_text())
    payload = {
        "schema": "kernel-analyzer-tcmp-heldout-summary-v1",
        "status": "COMPLETE",
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
            "The two held-out positives validate cross-model/new-operand prediction for a seen "
            "lm-head dX implementation family. They do not establish NEW_IMPL generalization. "
            "The genuinely new Ministral semantic patterns are negative controls, not cases."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
