#!/usr/bin/env python3
"""Re-audit retained cases with strict-op and composite counts separated."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str, compressed: bool = False):
    target = ROOT / path
    opener = gzip.open if compressed else open
    with opener(target, "rt") as handle:
        return json.load(handle)


def gate(passed, evidence, note="", failed_status="NEEDS_RECONFIRMATION"):
    return {"status": "PASS" if passed else failed_status, "evidence": evidence, "note": note}


def main() -> None:
    protocol = load("results/coverage/directional_bias_protocol.json")
    precision = load("results/final/precision.json.gz", True)["results"]
    mm_case, mm_carrier, mm_steps = precision["mm_case"], precision["mm_carrier"], precision["mm_steps"]
    mm_v2 = load("results/coverage/lmhead_t3_confirmation.json")

    liger = load("archive/nonprecision_v1/runs/liger.fused_ce.certificate.json")
    liger_mechanism = load("archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json")
    liger_trajectory = load("results/final/trajectory.json.gz", True)["results"]

    structured = load("results/final/structured_carrier_confirmation.json")
    qref = load("results/final/l23_qproj_backward_tile_reference_s8.json")
    qsham = load("results/final/l23_qproj_backward_tile_sham_s8.json")
    qtrajectory = load("results/final/l23_key_live_weight_adamw.json")

    mm_t1 = all(mm_case["causal_gates"].get(key) for key in [
        "concrete_forward_and_actual_vjp_mathematical_unit_bound",
        "natural_unscaled_local_native_edge_difference",
        "baseline_repeat_determinism_exact",
    ])
    mm_t2 = mm_case["causal_gates"].get("exact_arithmetic_intervention_replicated") and mm_steps["gates"].get("all_repair_same_weight_losses_exact")
    mm_t3_partial = mm_carrier["claim_boundary"].get("full_parameter_vectors_used") and mm_carrier["causal_gates"].get("cross_state_vector_u_positive")
    mm_t4 = mm_steps["gates"].get("live_weight_trajectory_measured") and mm_steps["gates"].get("every_step_fp32_master_pair_divergent")

    d_w = next(row for row in liger["directional_results"] if row["endpoint"] == "dW" and row["contrast"] == "default_accum")
    liger_t1 = liger["complete_unit_gates"]["default_loss_dH_dW_directional"] and liger_mechanism["flashattention_style_gates"]["closed_forward_and_actual_backward"]
    liger_t2 = liger["complete_unit_gates"]["accumulator_loss_and_dH_controls_exact"] and liger["accumulator_causal_readout"]["fraction_of_mean_new_error_removed"] > 0.9
    liger_t3 = d_w["cluster_bootstrap_95ci"][0] > 0 and d_w["bh_fdr_qvalue"] < 0.05 and d_w["nonzero_states"] == 24
    trajectory_steps = liger_trajectory["steps"]
    liger_t4 = len(trajectory_steps["default"]) == 32 and all(row["gates"]["same_weight_accumulator_delta_nonzero"] for row in trajectory_steps["default"])

    q_t1 = structured["confirmed_count"] == 1 and structured["gate"]["cluster_bootstrap_lower_95_positive"]
    ref_repeat = qref["arms"][0]["repeats"][0]["frozen_tile_causal_contrast"]
    sham_repeat = qsham["arms"][0]["repeats"][0]["frozen_tile_causal_contrast"]
    q_t2 = ref_repeat["directional_removal_fraction"] > 0 and abs(sham_repeat["directional_removal_fraction"]) < 1e-12
    q_t3 = structured["gate"]["holm_family_wise_alpha"] == 0.05 and structured["gate"]["independent_state_count"] == 32 and structured["gate"]["cluster_bootstrap_lower_95_positive"]
    q_t4 = qtrajectory["status"] == "COMPLETE" and all(
        qtrajectory["gates"].get(key) for key in [
            "same_state_order", "same_initial_fp32_master",
            "final_fp32_master_divergence_nonzero",
            "only_q_proj_live_weight_updated", "bf16_live_weight_feedback_observed",
            "same_weight_baseline_and_repair_measured_each_arm_each_step",
        ]
    ) and qtrajectory["gates"].get("tensor_values_saved") is False

    rows = [
        {
            "case": "seq128_lm_head_input_vjp_mm",
            "case_level": "HISTORICAL_REJECTED",
            "property_positive_eligible": False,
            "classification": "HISTORICAL_CASE_REJECTED_BY_V2_T3_CAUSAL_NONCOHERENT",
            "tiers": {
                "T1_LOCAL": gate(mm_t1, "precision.json.gz:mm_case"),
                "T2_CAUSAL": gate(mm_t2, "precision.json.gz:mm_case+mm_steps", "Arithmetic intervention uses exact same-weight loss and repeat controls."),
                "T3_COHERENT": gate(
                    mm_v2["frozen_success_gate_passed"],
                    "results/coverage/lmhead_t3_confirmation.json",
                    "A new 32-state, all-310-parameter, two-pass confirmation has a bootstrap interval crossing zero.",
                    failed_status="FAIL_CAUSAL_NONCOHERENT",
                ),
                "T4_ACCUMULATION": gate(mm_t4, "precision.json.gz:mm_steps"),
            },
        },
        {
            "case": "liger_fused_linear_ce_dw",
            "case_level": "STRICT_ROOT_ARITHMETIC_OP_CASE",
            "property_positive_eligible": True,
            "classification": "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE",
            "tiers": {
                "T1_LOCAL": gate(liger_t1, "liger.fused_ce.certificate.json"),
                "T2_CAUSAL": gate(liger_t2, "liger.fused_ce.certificate.json", "Accumulator intervention has exact loss/dH non-target controls."),
                "T3_COHERENT": gate(liger_t3, "liger.fused_ce.certificate.json:default_accum.dW"),
                "T4_ACCUMULATION": gate(liger_t4, "trajectory.json.gz"),
            },
        },
        {
            "case": "layer23_qproj_composite_tile",
            "case_level": "COMPOSITE_CARRIER_CASE",
            "property_positive_eligible": False,
            "classification": "COMPLETE_COMPOSITE_DIRECTIONAL_ACCUMULATION_CASE",
            "tiers": {
                "T1_LOCAL": gate(q_t1, "structured_carrier_confirmation.json"),
                "T2_CAUSAL": gate(q_t2, "l23_qproj_backward_tile_{reference,sham}_s8.json"),
                "T3_COHERENT": gate(q_t3, "structured_carrier_confirmation.json"),
                "T4_ACCUMULATION": gate(q_t4, "l23_key_live_weight_adamw.json"),
            },
            "boundary": "Passes as a composite carrier case; it is not promoted to a fully single-kernel-attributed mechanism.",
        },
    ]
    strict = sum(
        row["case_level"] == "STRICT_ROOT_ARITHMETIC_OP_CASE"
        and all(t["status"] == "PASS" for t in row["tiers"].values())
        for row in rows
    )
    composite = sum(
        row["case_level"] == "COMPOSITE_CARRIER_CASE"
        and all(t["status"] == "PASS" for t in row["tiers"].values())
        for row in rows
    )
    rejected = sum(any(t["status"].startswith("FAIL_") for t in row["tiers"].values()) for row in rows)
    payload = {
        "schema": "kernel-analyzer-existing-case-reaudit-v3",
        "status": "COMPLETE_FAIL_CLOSED_REAUDIT",
        "protocol_sha256": protocol["protocol_sha256"],
        "rows": rows,
        "counts": {
            "previous_project_cases": 3,
            "strict_root_arithmetic_op_pass": strict,
            "composite_carrier_pass": composite,
            "rejected_by_direction_gate": rejected,
            "needs_reconfirmation": 3 - strict - composite - rejected,
        },
        "claim_boundary": (
            "Strict op cases require a localized root arithmetic mechanism plus complete F+B, "
            "causal, carrier and accumulation gates. Composite carriers are retained separately "
            "and are forbidden from property-positive labels. No old case is grandfathered."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = ROOT / "results/coverage/existing_case_reaudit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
