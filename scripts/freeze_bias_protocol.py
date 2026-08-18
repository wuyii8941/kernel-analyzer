#!/usr/bin/env python3
"""Freeze the machine-readable directional-bias case protocol."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = {
        "schema": "kernel-analyzer-directional-bias-protocol-v3",
        "status": "FROZEN_BEFORE_PHI_DEEPSEEK_VALUES",
        "property_induction_allowed": False,
        "tiers": {
            "T1_LOCAL": {
                "required": ["exact_forward_actual_backward_binding", "repeat_stable", "finite", "local_difference_nonzero"],
                "screen_only": True,
                "sparse_coordinates_may_assign_case": False,
            },
            "T2_CAUSAL": {
                "required": ["exact_semantic_or_arithmetic_intervention", "declared_parameter_reach", "intervention_type_matched_negative_control"],
                "negative_controls": {
                    "runtime_replacement": "candidate_restoration_sham",
                    "arithmetic_or_config": "exact_baseline_repeat_and_exact_non_target_endpoints",
                    "composite_graph": "candidate_restoration_sham_and_complete_carrier_residual",
                },
            },
            "T3_COHERENT": {
                "required": ["independent_confirmation_states", "all_declared_carrier_coordinates", "repeat_exact", "cluster_bootstrap_lower_95_gt_zero"],
                "multiple_testing": "HOLM_FWER_OR_PREDECLARED_FDR_OVER_ALL_SIMULTANEOUS_HYPOTHESES",
                "posthoc_coordinate_selection": False,
                "hypotheses": {
                    "raw": {
                        "statistic": "distinct_state_cross_inner_product_u",
                        "meaning": "common direction in complete fixed parameter coordinates",
                    },
                    "relative": {
                        "statistic": "reference_relative_gradient_scale",
                        "formula": "dot(delta_g_s, g_ref_s) / dot(g_ref_s, g_ref_s)",
                        "meaning": "persistent amplification or attenuation of the current reference update",
                    },
                    "factor": {
                        "statistic": "analytic_error_coefficient_times_reference_only_carrier",
                        "meaning": "FlashAttention-style biased coefficient transported by a mathematically derived carrier",
                        "analytic_factorization_required": True,
                    },
                },
            },
            "T4_ACCUMULATION": {
                "required": ["frozen_paired_trajectory", "same_weight_contrast_before_each_update", "nonzero_declared_gradient_contrast", "live_weight_divergence"],
                "only_after_T3": True,
            },
        },
        "verdicts": [
            "NO_LOCAL_DIFFERENCE", "LOCAL_SCREEN_ONLY", "CAUSAL_NONCOHERENT",
            "COHERENT_SINGLE_STEP", "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE",
            "NEEDS_RECONFIRMATION", "UNRESOLVED",
        ],
        "optional_stopping_after_failed_frozen_confirmation": False,
    }
    payload["protocol_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    output = ROOT / "results/coverage/directional_bias_protocol.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": payload["protocol_sha256"]}))


if __name__ == "__main__":
    main()
