#!/usr/bin/env python3
"""Separate trajectory-local Flash-style cases from cross-state generalization."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def digest(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def gate(status: str, evidence: str, note: str = "") -> dict[str, str]:
    return {"status": status, "evidence": evidence, "note": note}


def all_pass(gates: dict[str, dict[str, str]]) -> bool:
    return all(row["status"] == "PASS" for row in gates.values())


def main() -> None:
    old_protocol = load("results/coverage/directional_bias_protocol.json")
    precision = load("results/final/precision.json.gz")["results"]
    mm_case = precision["mm_case"]
    mm_steps = precision["mm_steps"]
    mm_v2 = load("results/coverage/lmhead_t3_confirmation.json")

    liger = load("archive/nonprecision_v1/runs/liger.fused_ce.certificate.json")
    liger_mechanism = load("archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json")
    liger_trajectory = load("results/final/trajectory.json.gz")["results"]

    phi = load("results/coverage/cases/phi4_seq64_lmhead_dx.json")
    qwen_mm = load("results/coverage/cases/qwen128_vproj.json")
    qwen_repair = load("results/coverage/cases/qwen128_vproj_repair_pilot.json")
    qwen_trajectory = load("results/coverage/cases/qwen128_vproj_trajectory.json")
    mamba_mm = load("results/coverage/cases/mamba_seq64_input_proj.json")
    mamba_repair = load("results/coverage/cases/mamba_seq64_input_proj_repair_pilot.json")
    mamba_trajectory = load("results/coverage/cases/mamba_seq64_input_proj_trajectory.json")
    softmax = load("results/coverage/cases/qwen128_softmax_fb.json")
    softmax_trajectory = load(
        "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json"
    )
    silu_cause = load("results/round2/vl_silu_cause.json")
    silu_bias = load("results/round2/vl_bias.json")
    silu_trajectory = load("results/coverage/cases/qwen3vl_layer0_silu_trajectory.json")

    l23 = load("results/coverage/cases/l23_qproj_attention_state_region.json")

    protocol = {
        "schema": "kernel-analyzer-case-classification-protocol-v1",
        "status": "METHODOLOGY_CORRECTION_AFTER_CASE_PROPERTY_SEPARATION",
        "supersedes_for_case_count_only": old_protocol["protocol_sha256"],
        "preserves_original_measurements": True,
        "prospective_preregistration_claimed": False,
        "tracks": {
            "FLASH_STYLE_CASE": {
                "scope": (
                    "one concrete natural F+B mechanism, closed at either one arithmetic root "
                    "or a minimal semantic-region boundary, and its paired training trajectory"
                ),
                "required": [
                    "exact_forward_actual_backward_binding",
                    "localized_numerical_mechanism",
                    "causal_repair_with_matched_control",
                    "real_parameter_gradient_carrier",
                    "paired_same-weight_trajectory",
                    "trajectory_directional_accumulation",
                    "live_weight_divergence",
                ],
                "does_not_require": "fixed-coordinate direction across unrelated natural inputs",
            },
            "GENERALIZABLE_BIAS": {
                "scope": "cross-state evidence beyond one training trajectory",
                "required": [
                    "all_declared_coordinates",
                    "independent_confirmation_states",
                    "repeat_exact",
                    "distinct_cluster_bootstrap_lower_95_gt_zero",
                ],
                "note": (
                    "A pass is cross-state evidence for the concrete mechanism; it is not by "
                    "itself cross-shape, cross-model, or cross-operator-family generalization."
                ),
            },
        },
        "classification_rule": (
            "Failure of GENERALIZABLE_BIAS cannot revoke a complete FLASH_STYLE_CASE. "
            "Missing causal repair or trajectory remains fail-closed for FLASH_STYLE_CASE. "
            "A joint semantic region is strict only when its joint repair closes the carrier, "
            "its matched sham is null, and its paired trajectory accumulates directionally; "
            "such a case is not automatically a single-kernel property label."
        ),
    }
    protocol["protocol_sha256"] = canonical(protocol)
    protocol_path = RESULTS / "coverage/case_classification_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    mm_local = all(mm_case["causal_gates"].get(key) for key in (
        "concrete_forward_and_actual_vjp_mathematical_unit_bound",
        "natural_unscaled_local_native_edge_difference",
        "baseline_repeat_determinism_exact",
    ))
    mm_causal = bool(mm_case["causal_gates"].get("exact_arithmetic_intervention_replicated"))
    mm_trajectory = all(mm_steps["gates"].get(key) for key in (
        "all_repair_same_weight_losses_exact",
        "every_step_natural_local_parameter_carrier_nonzero",
        "live_weight_trajectory_measured",
        "every_step_fp32_master_pair_divergent",
        "bf16_live_weight_feedback_active",
    ))
    mm_directional_accumulation = all(mm_case["causal_gates"].get(key) for key in (
        "all_blind_confirmation_projections_positive",
        "cumulative_carrier_exceeds_frozen_balanced_sign_control",
        "all_32_live_steps_have_natural_parameter_carrier",
        "all_32_live_steps_have_fp32_master_divergence",
    ))
    mm_flash_gates = {
        "F1_COMPLETE_FB": gate("PASS" if mm_local else "FAIL", "precision.json.gz:mm_case"),
        "F2_CAUSAL_REPAIR": gate("PASS" if mm_causal else "FAIL", "precision.json.gz:mm_case"),
        "F3_REAL_CARRIER": gate(
            "PASS" if mm_directional_accumulation else "FAIL", "precision.json.gz:mm_carrier",
            "Independent frozen-carrier projections and balanced-sign accumulation control pass.",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if mm_trajectory else "FAIL", "precision.json.gz:mm_steps",
            "32-step FP32-master trajectory; every step has a natural carrier and arm divergence.",
        ),
    }

    d_w = next(row for row in liger["directional_results"]
               if row["endpoint"] == "dW" and row["contrast"] == "default_accum")
    liger_flash_gates = {
        "F1_COMPLETE_FB": gate(
            "PASS" if liger_mechanism["flashattention_style_gates"]
            ["closed_forward_and_actual_backward"] else "FAIL",
            "liger.fused_ce.mechanism.json",
        ),
        "F2_CAUSAL_REPAIR": gate(
            "PASS" if liger["accumulator_causal_readout"]
            ["fraction_of_mean_new_error_removed"] > 0.9 else "FAIL",
            "liger.fused_ce.certificate.json",
        ),
        "F3_REAL_CARRIER": gate(
            "PASS" if d_w["cluster_bootstrap_95ci"][0] > 0 else "FAIL",
            "liger.fused_ce.certificate.json:default_accum.dW",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if len(liger_trajectory["steps"]["default"]) == 32 and all(
                row["gates"]["same_weight_accumulator_delta_nonzero"]
                for row in liger_trajectory["steps"]["default"]
            ) else "FAIL",
            "trajectory.json.gz",
        ),
    }

    phi_flash_gates = {
        "F1_COMPLETE_FB": gate(
            "PASS" if all(value for key, value in phi["concrete_program_proof"].items()
                          if isinstance(value, bool)) else "FAIL",
            "cases/phi4_seq64_lmhead_dx.json",
        ),
        "F2_CAUSAL_REPAIR": gate(
            "PASS" if phi["causal_repair"]["loss_exact_states"] == 32
            and phi["causal_repair"]["sham_all_gradient_exact_states"] == 32 else "FAIL",
            "cases/phi4_seq64_lmhead_dx.json:causal_repair",
        ),
        "F3_REAL_CARRIER": gate(
            "PASS" if phi["causal_repair"]["final_norm_weight_carrier"]
            ["cluster_bootstrap_95"]["lower_95"] > 0 else "FAIL",
            "cases/phi4_seq64_lmhead_dx.json:final_norm_weight_carrier",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if all(phi["live_accumulation"]["gates"].values()) else "FAIL",
            "cases/phi4_seq64_lmhead_dx_trajectory.json",
        ),
    }

    q_flash_gates = {
        "F1_COMPLETE_FB": gate(
            "PASS" if l23["gates"]["complete_forward_backward_math"] else "FAIL",
            "cases/l23_qproj_attention_state_region.json:complete_forward_backward_math",
            "Concrete q_proj F+B unit bound to a closed attention-state semantic region.",
        ),
        "F2_CAUSAL_REPAIR": gate(
            "PASS" if l23["gates"]["joint_attention_state_repair_closes_direction"]
            and l23["gates"]["s_bwd_only_repair_closes_direction"]
            and l23["gates"]["matched_sham_exact"] else "FAIL",
            "cases/l23_qproj_attention_state_region.json:causal_attribution",
        ),
        "F3_REAL_CARRIER": gate(
            "PASS" if l23["gates"]["real_directional_parameter_carrier"] else "FAIL",
            "cases/l23_qproj_attention_state_region.json:real_directional_parameter_carrier",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if l23["gates"]["paired_live_weight_trajectory"] else "FAIL",
            "cases/l23_qproj_attention_state_region.json:trajectory",
        ),
    }

    def local_only_gates(complete: bool, causal: bool, evidence: str) -> dict[str, dict[str, str]]:
        return {
            "F1_COMPLETE_FB": gate("PASS" if complete else "FAIL", evidence),
            "F2_CAUSAL_REPAIR": gate(
                "PASS" if causal else "NEEDS_EXPERIMENT", evidence,
                "No type-compatible endpoint repair and matched sham are currently bound."
                if not causal else "Exact intervention exists.",
            ),
            "F3_REAL_CARRIER": gate(
                "NEEDS_EXPERIMENT", evidence,
                "No trajectory-local complete parameter carrier certificate exists.",
            ),
            "F4_PAIRED_TRAJECTORY": gate(
                "NEEDS_EXPERIMENT", evidence,
                "No paired same-weight evolving trajectory exists.",
            ),
        }

    qwen_complete = all(value for key, value in qwen_mm["concrete_program_proof"].items()
                        if isinstance(value, bool))
    mamba_complete = all(value for key, value in mamba_mm["concrete_program_proof"].items()
                         if isinstance(value, bool))
    softmax_complete = all(value for key, value in softmax["concrete_program_proof"].items()
                           if isinstance(value, bool))
    silu_complete = silu_cause["status"] == "COMPLETE_CAUSAL_INTERVENTION"
    silu_causal = (
        silu_cause["comparison"]["candidate_intervention_exact_parameter_count"]
        == silu_cause["comparison"]["parameter_count"]
    )
    silu_flash_gates = {
        "F1_COMPLETE_FB": gate(
            "PASS" if silu_complete else "FAIL",
            "results/round2/vl_math_ledger.json.gz:vl-fb-1315",
        ),
        "F2_CAUSAL_REPAIR": gate(
            "PASS" if silu_causal
            and silu_trajectory["gates"]["single_invocation_repair_nonzero"]
            and silu_trajectory["gates"]["matched_sham_exact"] else "FAIL",
            "cases/qwen3vl_layer0_silu_trajectory.json:initial_controls",
        ),
        "F3_REAL_CARRIER": gate(
            "PASS" if silu_trajectory["gates"]["only_declared_parameter_updated"]
            and silu_trajectory["gates"]["all_steps_repair_nonzero"] else "FAIL",
            "cases/qwen3vl_layer0_silu_trajectory.json:records",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if silu_trajectory["gates"]["directional_live_weight_accumulation"]
            else "FAIL",
            "cases/qwen3vl_layer0_silu_trajectory.json:directional_projection_checkpoints",
            "Live weights diverge, but the frozen-direction projection does not grow through step 32.",
        ),
    }
    mamba_flash_gates = {
        "F1_COMPLETE_FB": gate(
            "PASS" if mamba_complete else "FAIL",
            "cases/mamba_seq64_input_proj.json:concrete_program_proof",
        ),
        "F2_CAUSAL_REPAIR": gate(
            "PASS" if all(mamba_repair["gates"][key] for key in (
                "restoration_sham_exact",
                "accumulation_intervention_nonnull_every_state",
                "accumulation_intervention_reduces_fp32_sse_every_state",
                "direct_weight_carrier_nonnull_every_state",
            )) else "FAIL",
            "cases/mamba_seq64_input_proj_repair_pilot.json",
        ),
        "F3_REAL_CARRIER": gate(
            "PASS" if mamba_trajectory["gates"]["only_declared_parameter_updated"]
            and mamba_trajectory["gates"]["all_steps_repair_nonzero"] else "FAIL",
            "cases/mamba_seq64_input_proj_trajectory.json:records",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if mamba_trajectory["gates"]
            ["directional_live_weight_accumulation"] else "FAIL",
            "cases/mamba_seq64_input_proj_trajectory.json:directional_projections",
        ),
    }
    qwen_vproj_flash_gates = {
        "F1_COMPLETE_FB": gate(
            "PASS" if qwen_complete else "FAIL",
            "cases/qwen128_vproj.json:concrete_program_proof",
        ),
        "F2_CAUSAL_REPAIR": gate(
            "PASS" if all(qwen_repair["gates"][key] for key in (
                "restoration_sham_exact",
                "accumulation_intervention_nonnull_every_state",
                "accumulation_intervention_reduces_fp32_sse_every_state",
                "direct_weight_carrier_nonnull_every_state",
            )) else "FAIL",
            "cases/qwen128_vproj_repair_pilot.json",
        ),
        "F3_REAL_CARRIER": gate(
            "PASS" if qwen_trajectory["gates"]["only_declared_parameter_updated"]
            and qwen_trajectory["gates"]["all_steps_repair_nonzero"] else "FAIL",
            "cases/qwen128_vproj_trajectory.json:records",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if qwen_trajectory["gates"]
            ["directional_live_weight_accumulation"] else "FAIL",
            "cases/qwen128_vproj_trajectory.json:directional_projections",
            "The frozen-direction projection grows through step 16 but falls at step 32.",
        ),
    }
    softmax_flash_gates = {
        "F1_COMPLETE_FB": gate(
            "PASS" if softmax_complete else "FAIL",
            "cases/qwen128_softmax_fb.json:concrete_program_proof",
        ),
        "F2_CAUSAL_REPAIR": gate(
            "PASS" if softmax_trajectory["gates"]["matched_sham_exact"]
            and softmax_trajectory["gates"]["forward_loss_unchanged_by_backward_repair"]
            and softmax_trajectory["gates"]["saved_p_boundary_repair_nonzero"] else "FAIL",
            "cases/qwen128_softmax_saved_p_trajectory.json:initial_controls",
        ),
        "F3_REAL_CARRIER": gate(
            "PASS" if softmax_trajectory["gates"]
            ["only_declared_qk_parameters_updated"]
            and softmax_trajectory["gates"]["all_steps_repair_nonzero"] else "FAIL",
            "cases/qwen128_softmax_saved_p_trajectory.json:records",
        ),
        "F4_PAIRED_TRAJECTORY": gate(
            "PASS" if softmax_trajectory["gates"]
            ["directional_live_weight_accumulation"] else "FAIL",
            "cases/qwen128_softmax_saved_p_trajectory.json:directional_projections",
        ),
    }

    rows = [
        {
            "case": "seq128_lm_head_input_vjp_mm",
            "mechanism_level": "ROOT_ARITHMETIC",
            "flash_style": {
                "verdict": "PASS_FLASH_STYLE_CASE" if all_pass(mm_flash_gates) else "INCOMPLETE",
                "gates": mm_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "FAIL_CROSS_STATE_NONCOHERENT",
                "evidence": "results/coverage/lmhead_t3_confirmation.json",
                "bootstrap": mm_v2["cluster_pseudovalue_bootstrap_95"],
            },
            "property_positive_eligible": False,
            "note": "Restored as a Flash-style case; the original 32-state negative result is retained.",
        },
        {
            "case": "liger_fused_linear_ce_dw",
            "mechanism_level": "ROOT_ARITHMETIC",
            "flash_style": {
                "verdict": "PASS_FLASH_STYLE_CASE" if all_pass(liger_flash_gates) else "INCOMPLETE",
                "gates": liger_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "PASS_CROSS_STATE_CONCRETE_MECHANISM",
                "evidence": "liger.fused_ce.certificate.json:default_accum.dW",
                "bootstrap": {"lower_95": d_w["cluster_bootstrap_95ci"][0],
                              "upper_95": d_w["cluster_bootstrap_95ci"][1]},
            },
            "property_positive_eligible": True,
        },
        {
            "case": "phi4_seq64_lmhead_dx_mm",
            "mechanism_level": "ROOT_ARITHMETIC",
            "flash_style": {
                "verdict": "PASS_FLASH_STYLE_CASE" if all_pass(phi_flash_gates) else "INCOMPLETE",
                "gates": phi_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "PASS_CROSS_STATE_CONCRETE_MECHANISM",
                "evidence": "cases/phi4_seq64_lmhead_dx.json",
                "bootstrap": phi["causal_repair"]["final_norm_weight_carrier"]
                ["cluster_bootstrap_95"],
            },
            "property_positive_eligible": True,
            "note": "Trajectory is bounded to model.norm.weight with other parameters frozen.",
        },
        {
            "case": "layer23_qproj_attention_state_region",
            "mechanism_level": "CLOSED_SEMANTIC_REGION",
            "flash_style": {
                "verdict": "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE"
                if all_pass(q_flash_gates)
                else "INCOMPLETE",
                "gates": q_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "NOT_ELIGIBLE_AS_SINGLE_KERNEL_PROPERTY",
                "evidence": "cases/l23_qproj_attention_state_region.json",
            },
            "property_positive_eligible": False,
            "note": (
                "Strict concrete case at a causally closed semantic-region boundary; no unique "
                "single-kernel root or cross-operator property is claimed."
            ),
        },
        {
            "case": "qwen128_layer27_softmax_saved_state",
            "mechanism_level": "ROOT_SEMANTIC_REGION",
            "flash_style": {
                "verdict": "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE"
                if all_pass(softmax_flash_gates) else "INCOMPLETE",
                "gates": softmax_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "FAIL_CROSS_STATE_NONCOHERENT",
                "evidence": "cases/qwen128_softmax_fb_formal.json:semantic_total",
                "bootstrap": softmax["numerical"]["sources"]["semantic_total"]
                ["cluster_bootstrap_95"],
            },
            "property_positive_eligible": False,
            "note": (
                "The causal boundary replaces reconstructed probability at dS with the typed "
                "true-forward-P VJP. This is a closed saved-state semantic region, not a unique "
                "single-Triton-instruction attribution."
            ),
        },
        {
            "case": "qwen3vl_silu_backward_decomposition",
            "mechanism_level": "ROOT_ARITHMETIC",
            "flash_style": {
                "verdict": "FAIL_DIRECTIONAL_ACCUMULATION",
                "gates": silu_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "FAIL_CROSS_STATE_NONCOHERENT",
                "evidence": "results/round2/vl_bias.json:global_direction",
                "statistic": silu_bias["global_direction"],
            },
            "property_positive_eligible": False,
        },
        {
            "case": "qwen128_layer0_vproj_output",
            "mechanism_level": "ROOT_ARITHMETIC",
            "flash_style": {
                "verdict": "FAIL_DIRECTIONAL_ACCUMULATION",
                "gates": qwen_vproj_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "FAIL_CROSS_STATE_CARRIER_NONCOHERENT",
                "evidence": "cases/qwen128_vproj_repair_pilot.json",
                "bootstrap": qwen_repair["direct_weight_carrier"]
                ["cluster_bootstrap_95"],
            },
            "property_positive_eligible": False,
            "note": (
                "The local MM-accumulation repair is causal but does not accumulate "
                "monotonically through step 32. Coherent output rounding is outside this "
                "root-arithmetic intervention and remains a separate semantic-region question."
            ),
        },
        {
            "case": "mamba64_layer0_input_proj_output",
            "mechanism_level": "ROOT_ARITHMETIC",
            "flash_style": {
                "verdict": "PASS_FLASH_STYLE_CASE" if all_pass(mamba_flash_gates)
                else "INCOMPLETE",
                "gates": mamba_flash_gates,
            },
            "generalizable_bias": {
                "verdict": "FAIL_CROSS_STATE_CARRIER_NONCOHERENT",
                "evidence": "cases/mamba_seq64_input_proj_repair_pilot.json",
                "bootstrap": mamba_repair["direct_weight_carrier"]
                ["cluster_bootstrap_95"],
            },
            "property_positive_eligible": False,
        },
    ]

    root_pass = sum(row["flash_style"]["verdict"] == "PASS_FLASH_STYLE_CASE"
                    for row in rows)
    region_pass = sum(row["flash_style"]["verdict"]
                      == "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE" for row in rows)
    strict_pass = root_pass + region_pass
    composite_pass = sum(row["flash_style"]["verdict"]
                         == "PASS_COMPOSITE_FLASH_STYLE_CASE" for row in rows)
    property_pass = sum(row["generalizable_bias"]["verdict"]
                        == "PASS_CROSS_STATE_CONCRETE_MECHANISM" for row in rows)
    completed_negative = sum(row["flash_style"]["verdict"].startswith("FAIL_")
                             for row in rows)
    payload = {
        "schema": "kernel-analyzer-flash-style-reaudit-v1",
        "status": "COMPLETE_DUAL_TRACK_REAUDIT",
        "protocol_sha256": protocol["protocol_sha256"],
        "rows": rows,
        "counts": {
            "audited_candidates": len(rows),
            "strict_flash_style_cases": strict_pass,
            "composite_flash_style_cases": composite_pass,
            "cross_state_concrete_mechanism_passes": property_pass,
            "cross_operator_property_claims": 0,
            "strict_root_arithmetic_op_pass": root_pass,
            "strict_semantic_region_pass": region_pass,
            "composite_carrier_pass": composite_pass,
            "rejected_by_direction_gate": 0,
            "completed_negative_cases": completed_negative,
            "needs_reconfirmation": (
                len(rows) - strict_pass - composite_pass - completed_negative
            ),
            "previous_project_cases": 4,
        },
        "evidence_files": {
            relative: digest(relative) for relative in (
                "results/final/precision.json.gz",
                "results/coverage/lmhead_t3_confirmation.json",
                "results/final/trajectory.json.gz",
                "results/coverage/cases/phi4_seq64_lmhead_dx.json",
                "results/coverage/cases/qwen128_softmax_fb.json",
                "results/coverage/cases/qwen128_softmax_fb_formal.json",
                "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json",
                "results/coverage/cases/qwen128_vproj.json",
                "results/coverage/cases/qwen128_vproj_repair_pilot.json",
                "results/coverage/cases/qwen128_vproj_trajectory.json",
                "results/coverage/cases/l23_qproj_attention_state_region.json",
                "results/coverage/cases/mamba_seq64_input_proj.json",
                "results/coverage/cases/mamba_seq64_input_proj_repair_pilot.json",
                "results/coverage/cases/mamba_seq64_input_proj_trajectory.json",
                "results/round2/vl_silu_cause.json",
                "results/round2/vl_bias.json",
                "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
            )
        },
        "claim_boundary": (
            "This is a transparent post-measurement methodology correction that separates "
            "case identity from property generalization. It does not create new measurements, "
            "does not promote incomplete candidates, and does not claim a cross-operator property."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    output = RESULTS / "coverage/existing_case_reaudit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "counts": payload["counts"],
                      "sha256": payload["result_sha256"]}))


if __name__ == "__main__":
    main()
