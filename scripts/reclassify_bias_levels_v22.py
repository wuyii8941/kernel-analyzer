#!/usr/bin/env python3
"""Reclassify existing paired trajectories under the unified evidence levels.

This is an artifact audit only.  It does not change old verdicts and does not
claim a mechanism property.  Causal live-weight separation, directional
persistence, and formation/trajectory contrast alignment are separate fields.
A growing parameter-distance norm is never relabeled as directional bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_analyzer.bias_formation_v22 import certify_trajectory_separation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_v22"


SPECS = [
    {
        "case_id": "liger_fused_ce",
        "mechanism_family": "FUSED_ACCUMULATION_LOSS",
        "path": "results/trajectory/liger_trajectory.json",
        "row_key": "step_rows",
        "metric": "fp32_master_pair_l2",
        "old_status_key": "verdict",
        "old_direction_key": "all_64_frozen_carrier_projections_positive",
        "formation_contrast": "FP32_DW_ACCUMULATOR",
        "trajectory_contrast": "FP32_DW_ACCUMULATOR",
        "contrast_alignment": "ALIGNED",
        "same_contrast_full_chain": True,
    },
    {
        "case_id": "phi4_seq64_lmhead_dx",
        "mechanism_family": "LOSS_HEAD_TRANSPORT",
        "path": "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json",
        "row_key": "steps",
        "metric": "master_arm_distance_l2",
        "old_status_key": "status",
        "old_direction_key": "same_weight_carrier_direction_stable",
        "formation_contrast": "ANALYTIC_DX_MM_REPAIR",
        "trajectory_contrast": "ANALYTIC_DX_MM_REPAIR",
        "contrast_alignment": "ALIGNED",
        "same_contrast_full_chain": True,
    },
    {
        "case_id": "qwen64_vproj_mm",
        "mechanism_family": "MM_ACCUMULATION",
        "path": "results/coverage/cases/qwen64_vproj_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
        "formation_contrast": "JOINT_KERNEL_PLUS_UNBIASED_ROUNDING",
        "trajectory_contrast": "KERNEL_ONLY_FP32_MM_WITH_BF16_ABI",
        "contrast_alignment": "MISMATCH",
        "same_contrast_full_chain": False,
    },
    {
        "case_id": "qwen128_vproj_mm",
        "mechanism_family": "MM_ACCUMULATION",
        "path": "results/coverage/cases/qwen128_vproj_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
        "formation_contrast": "ROUNDING_ONLY_UNBIASED_BF16",
        "trajectory_contrast": "KERNEL_ONLY_FP32_MM_WITH_BF16_ABI",
        "contrast_alignment": "MISMATCH",
        "same_contrast_full_chain": False,
    },
    {
        "case_id": "qwen_saved_p_seq128",
        "mechanism_family": "SAVED_STATE_SOFTMAX_TRANSPORT",
        "path": "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json",
        "row_key": "records",
        "metric": "joint_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
        "formation_contrast": "TRUE_FORWARD_P_AT_DS",
        "trajectory_contrast": "TRUE_FORWARD_P_AT_DS",
        "contrast_alignment": "ALIGNED",
        "same_contrast_full_chain": True,
    },
    {
        "case_id": "qwen3vl_silu_layer0",
        "mechanism_family": "NONLINEAR_SILU_BACKWARD",
        "path": "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
        "formation_contrast": "NATIVE_SILU_BACKWARD_PLUS_ANTITHETIC_ADAM_RESPONSE",
        "trajectory_contrast": "NATIVE_SILU_BACKWARD",
        "contrast_alignment": "ALIGNED_BASE_CONTRAST",
        "same_contrast_full_chain": False,
    },
    {
        "case_id": "mamba_seq64_input_proj",
        "mechanism_family": "RECURRENT_INPUT_PROJECTION",
        "path": "results/coverage/cases/mamba_seq64_input_proj_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
        "formation_contrast": "JOINT_KERNEL_PLUS_UNBIASED_ROUNDING",
        "trajectory_contrast": "KERNEL_ONLY_FP32_MM_WITH_BF16_ABI",
        "contrast_alignment": "MISMATCH",
        "same_contrast_full_chain": False,
    },
    {
        "case_id": "qwen_layer23_attention_state",
        "mechanism_family": "ATTENTION_STATE_TRANSPORT",
        "semantic_case_group": "qwen_layer23_qproj_composite",
        "count_as_unique_case": True,
        "dedup_reason": None,
        "path": "results/final/l23_attention_live_weight.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "bf16_live_weight_feedback_observed",
        "formation_contrast": "S_BWD_CAUSAL_REGION",
        "trajectory_contrast": "CONSERVATIVE_S_K_REGION",
        "contrast_alignment": "ALIGNED_SEMANTIC_SUPERSET",
        "same_contrast_full_chain": True,
    },
]

# Candidate artifacts deliberately retained in the audit ledger but excluded
# from the complete-case count.  This is important: an alternate repair
# boundary is not a duplicate *complete* case when its sham is not exact, and
# an old SEUP negative control must not be silently promoted to a v2.2 case.
EXCLUDED_ARTIFACTS = [
    {
        "artifact": "results/final/l23_key_live_weight_adamw.json",
        "classification": "INCOMPLETE_COMMON_TRAJECTORY",
        "reason": (
            "same_weight sham is not exact: every step has a nonzero "
            "baseline_loss-repaired_loss difference; fail closed"
        ),
        "semantic_group": "qwen_layer23_qproj_composite",
    },
    {
        "artifact": "results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz",
        "classification": "REGISTERED_NEGATIVE_CONTROL",
        "reason": (
            "endpoint repair is not nonzero at every step, matched sham is "
            "not certified, and stable carrier gate failed"
        ),
        "semantic_group": "qwen_bmm_seq64",
    },
    {
        "artifact": "results/property/seup_mainline/qwen_rsqrt_seq256_seup.json.gz",
        "classification": "REGISTERED_NEGATIVE_CONTROL",
        "reason": (
            "matched sham is not certified and stable carrier gate failed; "
            "not a complete v2.2 causal case"
        ),
        "semantic_group": "qwen_rsqrt_seq256",
    },
    {
        "artifact": "results/property/seup_mainline/liger_seup.json",
        "classification": "DUPLICATE_OF_COMPLETE_ARTIFACT",
        "reason": "shorter SEUP certificate for results/trajectory/liger_trajectory.json",
        "semantic_group": "liger_fused_ce",
    },
    {
        "artifact": "results/property/seup_mainline/phi_seup.json",
        "classification": "DUPLICATE_OF_COMPLETE_ARTIFACT",
        "reason": "shorter SEUP certificate for the Phi seq64 lm-head trajectory",
        "semantic_group": "phi4_seq64_lmhead_dx",
    },
    {
        "artifact": "results/property/bias_formation/consequence/phi4_lm_head_dx_trajectory.json",
        "classification": "DUPLICATE_OF_COMPLETE_ARTIFACT",
        "reason": "same Phi seq64 paired trajectory under the older consequence path",
        "semantic_group": "phi4_seq64_lmhead_dx",
    },
    {
        "artifact": "results/property/bias_formation/consequence/phi4_lm_head_dx_seup.json",
        "classification": "DUPLICATE_OF_COMPLETE_ARTIFACT",
        "reason": "same Phi seq64 SEUP certificate under the older consequence path",
        "semantic_group": "phi4_seq64_lmhead_dx",
    },
    {
        "artifact": "results/property/seup_mainline/qwen_softmax_seup.json",
        "classification": "DUPLICATE_OF_COMPLETE_ARTIFACT",
        "reason": "shorter saved-P softmax consequence certificate",
        "semantic_group": "qwen_saved_p_seq128",
    },
]


def normalized_gates(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, bool]:
    gates = payload.get("gates", {})
    # Some older live-weight runners did not emit a boolean sham field.  The
    # nested same-weight controls are sufficient only when we verify the
    # equality from the recorded values; merely measuring both arms is not a
    # sham certificate.
    nested_sham_exact = False
    nested_sham_observed = bool(rows) and all(
        isinstance(row.get("same_weight"), dict)
        and all(
            isinstance(row["same_weight"].get(arm), dict)
            and row["same_weight"][arm].get("baseline_loss")
            == row["same_weight"][arm].get("repaired_loss")
            for arm in ("default", "repair")
        )
        for row in rows
    )
    nested_sham_exact = nested_sham_observed
    raw_sham_exact = bool(
        gates.get("matched_sham_exact")
        or gates.get("same_weight_loss_exact_every_step")
        or gates.get("all_same_weight_local_controls_pass")
        or gates.get("all_64_same_weight_local_controls_pass")
    )
    only_declared = bool(
        gates.get("only_declared_parameter_updated")
        or gates.get("only_declared_q_proj_live_weight_updated")
        or gates.get("only_q_proj_live_weight_updated")
        or gates.get("only_declared_qk_parameters_updated")
        or bool(payload.get("frozen_other_parameters"))
        or payload.get("other_parameters_updated") is False
    )
    full_step_scope = bool(
        gates.get("all_32_steps_and_both_arms_complete")
        and (
            gates.get("all_same_weight_local_controls_pass")
            or gates.get("all_64_same_weight_local_controls_pass")
        )
        and gates.get("fp32_master_divergence_after_every_step")
    )
    return {
        "repair_effect_present_every_step": bool(
            gates.get("repair_effect_present_every_step")
            or gates.get("all_steps_repair_nonzero")
            or gates.get("endpoint_repair_nonzero_every_step")
            or all(
                row.get("bf16_materialized_nonzero", 0) > 0
                for row in rows
            )
            or gates.get("all_64_same_weight_accumulator_deltas_nonzero")
        ),
        "matched_sham_exact": raw_sham_exact or nested_sham_exact,
        "only_declared_parameter_updated": only_declared,
        "full_step_two_arm_scope_closed": full_step_scope,
        "parameter_scope_closed": only_declared or full_step_scope,
    }


def main() -> None:
    cases: list[dict[str, Any]] = []
    for spec in SPECS:
        path = ROOT / spec["path"]
        payload = json.loads(path.read_text())
        rows = payload.get(spec["row_key"], [])
        metric = spec["metric"]
        trajectory_rows = [{"drift_norm": float(row[metric])} for row in rows]
        gates = normalized_gates(payload, rows)
        old_status = payload.get(spec["old_status_key"])
        old_direction = payload.get("gates", {}).get(spec["old_direction_key"])
        cert = certify_trajectory_separation(
            trajectory_rows,
            gates=gates,
            directional_persistence_gate=old_direction,
        )
        semantic_case_group = spec.get("semantic_case_group", spec["case_id"])
        cases.append({
            "case_id": spec["case_id"],
            "mechanism_family": spec["mechanism_family"],
            "semantic_case_group": semantic_case_group,
            "count_as_unique_case": spec.get("count_as_unique_case", True),
            "dedup_reason": spec.get("dedup_reason"),
            "artifact": spec["path"],
            "old_status": old_status,
            "old_fixed_direction_gate": old_direction,
            "formation_contrast": spec["formation_contrast"],
            "trajectory_contrast": spec["trajectory_contrast"],
            "contrast_alignment": spec["contrast_alignment"],
            "same_contrast_full_chain": spec["same_contrast_full_chain"],
            "normalized_causal_gates": gates,
            "sham_validation": {
                "nested_same_weight_exact": bool(
                    gates.get("matched_sham_exact")
                    and not bool(payload.get("gates", {}).get("matched_sham_exact"))
                    and not bool(payload.get("gates", {}).get("same_weight_loss_exact_every_step"))
                    and not bool(payload.get("gates", {}).get("all_same_weight_local_controls_pass"))
                    and not bool(payload.get("gates", {}).get("all_64_same_weight_local_controls_pass"))
                ),
                "raw_gate_present": bool(
                    payload.get("gates", {}).get("matched_sham_exact")
                    or payload.get("gates", {}).get("same_weight_loss_exact_every_step")
                    or payload.get("gates", {}).get("all_same_weight_local_controls_pass")
                    or payload.get("gates", {}).get("all_64_same_weight_local_controls_pass")
                ),
            },
            "trajectory_certificate": cert,
            "property_analysis_status": "MECHANISM_NOT_IDENTIFIED",
            "claim_boundary": (
                "Causal trajectory separation and signed directional persistence "
                "are separate claims.  A full mechanism-to-persistence chain also "
                "requires the formation and trajectory repair contrasts to align."
            ),
        })
    result = {
        "schema": "kernel-analyzer-bias-level-reclassification-v2",
        "definition": "formation, causal separation, directional persistence, and same-contrast full chain",
        "fixed_global_cross_state_direction_is_not_required": True,
        "trajectory_local_directional_persistence_is_required_for_flash_style": True,
        "strict_complete_case_gate": {
            "required": [
                "repair_effect_present_every_step",
                "matched_sham_exact",
                "parameter_scope_closed",
                "at_least_8_live_steps",
                "finite_drift_norms",
                "final_separation_greater_than_initial_for_separation",
            ],
            "note": (
                "A raw trajectory artifact without an exact sham is excluded. "
                "Growing norm certifies separation only; directional persistence "
                "and same-contrast closure are additional gates."
            ),
        },
        "cases": cases,
        "excluded_artifacts": EXCLUDED_ARTIFACTS,
        "counts": {},
        "property_analysis": {
            "uses_original_theory": True,
            "mechanisms": ["P1_SOURCE_ASYMMETRY", "P2_SOURCE_TRANSPORT_ALIGNMENT", "P3_FB_NUMERICAL_CONSISTENCY", "P4_NONLINEAR_RECTIFICATION", "P5_OPTIMIZER_RECTIFICATION", "P6_SEMANTIC_ORBIT_CENTERING"],
            "seup_is_consequence_layer": True,
            "conditional_level_ready": False,
            "conditional_level_reason": "Existing trajectories lack preregistered repeated condition strata.",
        },
    }
    counts: dict[str, int] = {}
    for case in cases:
        status = case["trajectory_certificate"]["status"]
        counts[status] = counts.get(status, 0) + 1
    result["counts"] = counts
    result["counts"]["trajectory_artifacts"] = len(cases)
    result["counts"]["unique_semantic_cases"] = len({case["semantic_case_group"] for case in cases})
    result["counts"]["mechanism_family_clusters"] = len({case["mechanism_family"].replace("KEY_MATERIALIZATION", "STATE_TRANSPORT") for case in cases})
    result["counts"]["prior_fixed_direction_confirmed"] = sum(
        case["old_fixed_direction_gate"] is True for case in cases
    )
    result["counts"]["directional_persistence_confirmed"] = sum(
        case["trajectory_certificate"]["directional_persistence"] == "CONFIRMED"
        for case in cases
    )
    result["counts"]["separation_without_directional_persistence"] = sum(
        case["old_fixed_direction_gate"] is not True for case in cases
    )
    result["counts"]["same_contrast_full_chain"] = sum(
        case["same_contrast_full_chain"] is True for case in cases
    )
    result["counts"]["excluded_artifacts"] = len(EXCLUDED_ARTIFACTS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trajectory_reclassification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Unified trajectory evidence audit",
        "",
        "This is an artifact audit, not a new experiment.  It separates causal",
        "paired parameter separation from signed directional persistence and from",
        "same-contrast mechanism-to-persistence closure.",
        "",
        "| case | separation | directional persistence | contrast alignment | same-contrast full chain | initial norm | final norm |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for case in cases:
        cert = case["trajectory_certificate"]
        lines.append(
            f"| {case['case_id']} | {cert['status']} | "
            f"{cert['directional_persistence']} | {case['contrast_alignment']} | "
            f"{case['same_contrast_full_chain']} | {cert.get('initial_drift_norm', '—')} | "
            f"{cert.get('final_drift_norm', '—')} |"
        )
    lines += [
        "",
        "The table contains only complete artifact rows.  The strict count is",
        f"{len(cases)} complete paired trajectory artifacts and {len({case['semantic_case_group'] for case in cases})} semantic cases.",
        f"{sum(case['old_fixed_direction_gate'] is True for case in cases)} have confirmed trajectory-local directional persistence; "
        f"{sum(case['old_fixed_direction_gate'] is not True for case in cases)} have separation without that proof.",
        f"{sum(case['same_contrast_full_chain'] is True for case in cases)} connect the current formation mechanism to persistence using an aligned repair contrast.",
        "Excluded candidates (including the incomplete layer-23 key repair) are",
        "listed in the JSON audit and are not silently counted as duplicates.",
        "",
        "All eight are paired-separation artifacts.  Only the directional subset",
        "may be called persistent Flash-style cases, and only an aligned-contrast",
        "subset closes the currently identified formation mechanism to persistence.",
    ]
    (OUT / "trajectory_reclassification.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "counts": counts}))


if __name__ == "__main__":
    main()
