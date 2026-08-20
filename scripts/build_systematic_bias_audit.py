#!/usr/bin/env python3
"""Build one evidence-bounded mechanism audit for all eight trajectory cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from kernel_analyzer.systematic_bias_audit import (
    first_conditional_bias_stage,
    validate_audit,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_systematic"


def read_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {relative}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def trajectory_index() -> dict[str, dict[str, Any]]:
    source = read_json("results/property/bias_formation_v22/trajectory_reclassification.json")
    return {str(row["case_id"]): row for row in source["cases"]}


def old_formation(relative: str) -> dict[str, str]:
    source = read_json(relative)
    confirmation = source["populations"]["confirmation"]
    return {
        "local": confirmation["LOCAL_ENDPOINT"]["status"].replace(
            "UNRESOLVED_INSUFFICIENT_STATES", "UNRESOLVED"
        ),
        "gradient": confirmation["PARAMETER_GRADIENT"]["status"].replace(
            "UNRESOLVED_INSUFFICIENT_STATES", "UNRESOLVED"
        ),
        "update": confirmation["EFFECTIVE_UPDATE"]["status"].replace(
            "UNRESOLVED_INSUFFICIENT_STATES", "UNRESOLVED"
        ),
    }


def trajectory(row: dict[str, Any]) -> dict[str, Any]:
    certificate = row["trajectory_certificate"]
    return {
        "status": certificate["status"],
        "separation_status": certificate["status"],
        "directional_persistence": certificate["directional_persistence"],
        "steps": certificate["step_count"],
        "initial_drift_norm": certificate.get("initial_drift_norm"),
        "final_drift_norm": certificate.get("final_drift_norm"),
        "fixed_global_direction_passed_old_gate": row.get("old_fixed_direction_gate"),
        "formation_contrast": row["formation_contrast"],
        "trajectory_contrast": row["trajectory_contrast"],
        "contrast_alignment": row["contrast_alignment"],
        "same_contrast_full_chain": row["same_contrast_full_chain"],
        "separation_is_not_bias_by_itself": True,
        "formation_label": False,
        "artifact": row["artifact"],
    }


def symmetric_consequence(relative: str) -> dict[str, Any]:
    source = read_json(relative)
    evaluation = source["evaluation"]
    return {
        "status": source["status"],
        "evaluation_steps": evaluation["evaluation_steps"],
        "local_accumulation_l2": evaluation["local_accumulation_l2"],
        "feedback_accumulation_l2": evaluation["feedback_accumulation_l2"],
        "max_recurrence_relative_residual": evaluation["max_recurrence_relative_residual"],
        "stable_fixed_carrier": source["gates"].get("stable_calibration_carrier"),
        "signed_persistence": evaluation.get("signed_persistence"),
        "source": relative,
    }


def conditional_debias(relative: str, arm: str) -> dict[str, Any]:
    """Load a compact fixed-state candidate-versus-debiased-ensemble result."""

    source = read_json(relative)
    if source.get("status") != "COMPLETE" or source.get(
        "global_direction_used_as_gate"
    ) is not False:
        raise ValueError("conditional debias summary is incomplete or globally gated")
    value = source["arms"][arm]
    aggregate = value["aggregate"]
    roles = aggregate["roles"]

    def layer(role: str) -> str:
        row = roles[role]
        if row["all_conditions_biased"]:
            return "BIASED"
        if row["all_conditions_centered"]:
            return "CENTERED"
        return "UNRESOLVED"

    sgd = layer("candidate_sgd_update_effect_removed")
    adam = layer("candidate_adamw_zero_update_effect_removed")
    return {
        "local": layer("candidate_local_effect_removed"),
        "gradient": layer("candidate_gradient_effect_removed"),
        "update": sgd if sgd == adam else "UNRESOLVED",
        "repair_local_residual": layer("repair_local_residual"),
        "conditions": source["condition_count"],
        "all_roles": roles,
        "reference": "STOCHASTIC_SOURCE_DEBIASED_ENSEMBLE",
        "absolute_downstream_repair_bias": aggregate[
            "absolute_downstream_repair_bias"
        ],
        "update_scopes": ["STATELESS_SGD", "ADAMW_ZERO_MOMENT_STEP1"],
        "source": relative,
    }


def rectification_aggregate(document: dict[str, Any]) -> dict[str, Any]:
    """Add energy-weighted geometry derivable from complete per-step records."""

    aggregate = dict(document["aggregate"])
    records = document.get("records", [])
    if records and "response_even_l2" in records[0]:
        even_energy = sum(row["response_even_l2"] ** 2 for row in records)
        odd_energy = sum(row["response_odd_l2"] ** 2 for row in records)
        aggregate["energy_weighted_response_even_on_sign_crossings"] = sum(
            row["response_even_l2"] ** 2
            * row["response_even_energy_on_sign_crossings"]
            for row in records
        ) / max(even_energy, 1e-30)
        aggregate["response_even_energy_in_first_two_steps"] = sum(
            row["response_even_l2"] ** 2 for row in records[:2]
        ) / max(even_energy, 1e-30)
        aggregate["step_integrated_response_even_energy_fraction"] = (
            even_energy / max(even_energy + odd_energy, 1e-30)
        )
    return aggregate


def cases() -> list[dict[str, Any]]:
    trajectories = trajectory_index()
    source_repairs = {
        row["case_id"]: row for row in read_json(
            "results/coverage/cases/source_aligned_repair_summary.json"
        )["cases"]
    }
    saved_p_symmetry_path = (
        "results/property/bias_property_search/saved_p_pairing_work_v2.json"
        if (ROOT / "results/property/bias_property_search/saved_p_pairing_work_v2.json").exists()
        else "results/property/bias_property_search/saved_p_pairing_work.json"
    )
    silu_symmetry_path = (
        "results/property/bias_property_search/vl_silu_optimizer_oddness_v2.json"
        if (ROOT / "results/property/bias_property_search/vl_silu_optimizer_oddness_v2.json").exists()
        else "results/property/bias_property_search/vl_silu_optimizer_oddness.json"
    )
    saved_p_symmetry = rectification_aggregate(read_json(saved_p_symmetry_path))
    silu_symmetry = rectification_aggregate(read_json(silu_symmetry_path))
    qwen128_conditional = conditional_debias(
        "results/property/conditional_debias/qwen128_vproj.json",
        "ROUNDING_ONLY",
    )
    qwen64_conditional = conditional_debias(
        "results/property/conditional_debias/qwen64_vproj.json",
        "JOINT",
    )
    mamba_conditional = conditional_debias(
        "results/property/conditional_debias/mamba_seq64_input_proj.json",
        "JOINT",
    )
    l23_antithetic = read_json("results/final/l23_s_bwd_antithetic.json")
    l23_gradient_even = [
        row["gradient_q_proj_tile"]["balanced_mean_over_natural"]
        for row in l23_antithetic["rows"]
    ]
    l23_adam_even = [
        row["adamw_zero_moment_step1_q_proj_tile"]["balanced_mean_over_natural"]
        for row in l23_antithetic["rows"]
    ]
    not_measured = {"local": "NOT_MEASURED", "gradient": "NOT_MEASURED", "update": "NOT_MEASURED"}

    result = [
        {
            "case_id": "liger_fused_ce",
            "model": "Qwen3-1.7B + Liger fused linear CE",
            "semantic_unit": "Z=HW^T; G=(softmax(Z)-onehot)/N; dH=GW; dW=G^T H",
            "forward_backward": {"status": "CLOSED", "scope": "full fused loss F+B region"},
            "physical_source": (
                "64 two-token chunk contributions are added sequentially to a BF16 dW "
                "accumulator; chunk geometry changes the rounding schedule"
            ),
            "formation": {
                "conditional": dict(not_measured),
                "global": old_formation("results/property/bias_formation/formation/liger_fused_ce_t128.json"),
                "label_source": "OPEN_LOOP_GLOBAL_V21",
                "interpretation": "calibration was directional, disjoint confirmation unresolved; this is not a conditional null",
            },
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY", "P6_SEMANTIC_ORBIT_CENTERING"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM",
                "intervention": {
                    "description": "promote only dW accumulation to FP32 while preserving loss, dH, and all untied gradients",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "removed_candidate_added_error_fraction": 0.957,
                },
                "why_directional": (
                    "finite-precision sequential accumulation is conditionally asymmetric under the "
                    "declared chunk schedule, so E[epsilon|chunk geometry] need not vanish"
                ),
                "claim_boundary": "case-specific source mechanism; no universal P1 property and no M7",
            },
            "bias_map": {
                "channel": "EVENT_PAIRING_ASYMMETRY",
                "status": "MATCHED_SUPPORT",
                "reason": (
                    "the BF16 chunk/schedule orbit is 24/24 one-signed, while the "
                    "same semantic orbit with FP32 accumulation is 13/11 and centered"
                ),
            },
            "trajectory": {
                **trajectory(trajectories["liger_fused_ce"]),
                "local_feedback": symmetric_consequence(
                    "results/property/seup_mainline/liger_seup.json"
                ),
            },
            "next_decisive_test": "variance-matched stratum-mean removal if P1 is to become a general property",
            "evidence": [
                "archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json",
                "archive/nonprecision_v1/runs/liger.fused_ce.certificate.json",
                "archive/nonprecision_v1/runs/liger.fused_ce.chunk.certificate.json",
                "results/trajectory/liger_trajectory.json",
                "results/property/seup_mainline/liger_seup.json",
            ],
        },
        {
            "case_id": "phi4_seq64_lmhead_dx",
            "model": "Phi-4-mini",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at lm_head input VJP",
            "forward_backward": {"status": "CLOSED", "scope": "one exact backward MM invocation and both VJP edges"},
            "physical_source": "same-BF16-operand MM kernel arithmetic; final output rounding is noncoherent",
            "formation": {
                "conditional": dict(not_measured),
                "global": old_formation("results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json"),
                "label_source": "OPEN_LOOP_GLOBAL_V21",
                "interpretation": "the strongest existing population result is LOCAL_CENTERED -> GRADIENT_BIASED -> UPDATE_BIASED",
            },
            "mechanism": {
                "candidate_properties": ["P2_SOURCE_TRANSPORT_ALIGNMENT"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_TRANSPORT_MECHANISM",
                "intervention": {
                    "description": "permute residual/row transport pairing while preserving the local residual multiset and norm",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "natural_gradient": "BIASED",
                    "shuffled_gradient": "CENTERED",
                },
                "why_directional": (
                    "the local mean is small, but the real backward pairing makes Cov(T,epsilon|c) nonzero"
                ),
                "claim_boundary": "empirical composite transport mechanism; analytic transport reconstruction remains incomplete",
            },
            "bias_map": {
                "channel": "EVENT_PAIRING_ASYMMETRY",
                "status": "MATCHED_SUPPORT",
                "reason": (
                    "the natural residual/transport joint pairing is directional, while "
                    "a residual-marginal-preserving row permutation restores centering"
                ),
            },
            "trajectory": {
                **trajectory(trajectories["phi4_seq64_lmhead_dx"]),
                "local_feedback": symmetric_consequence(
                    "results/property/seup_mainline/phi_seup.json"
                ),
            },
            "next_decisive_test": "close the remaining analytic VJP factors before naming one physical transport factor",
            "evidence": [
                "results/coverage/cases/phi4_seq64_lmhead_dx.json",
                "results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json",
                "results/property/bias_formation/interventions/phi4_mm_transport_pairing.json",
                "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json",
            ],
        },
        {
            "case_id": "qwen64_vproj_mm",
            "model": "Qwen3-1.7B",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj",
            "forward_backward": {"status": "CLOSED", "scope": "one exact forward MM with actual AOT backward edges"},
            "physical_source": "same-operand MM kernel arithmetic and deterministic output rounding are both directional",
            "formation": {
                "conditional": {
                    key: qwen64_conditional[key]
                    for key in ("local", "gradient", "update")
                },
                "conditional_details": qwen64_conditional,
                "global": dict(not_measured),
                "label_source": "FIXED_STATE_STOCHASTIC_REPAIR_ENSEMBLE",
            },
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM",
                "intervention": {
                    "description": "joint FP32 MM plus coordinate-wise unbiased BF16 materialization; kernel-only and rounding-only factorial controls",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "full_observed_source_repaired_in_expectation": True,
                    "downstream_global_carrier": source_repairs["qwen64_vproj_mm"]["downstream_carrier_effect"]["status"],
                },
                "why_directional": (
                    "at each of 16 fixed states, the deterministic joint MM-kernel plus "
                    "output-rounding residual has a nonzero candidate-minus-debiased-"
                    "ensemble mean that remains directional after the actual backward "
                    "and both declared optimizer mappings"
                ),
                "claim_boundary": (
                    "this closes conditional source formation relative to the stochastic "
                    "joint-source-debiased ensemble; it does not certify absolute "
                    "downstream repair bias without an exact downstream reference"
                ),
            },
            "bias_map": {
                "channel": "EVENT_PAIRING_ASYMMETRY",
                "status": "MATCHED_CONDITIONAL_SOURCE_SUPPORT",
                "reason": (
                    "an independent 16-repeat confirmation centers the repair local "
                    "residual in 16/16 fixed conditions, while candidate-minus-repair "
                    "local, gradient, SGD-update, and zero-moment AdamW-update effects "
                    "are biased in 16/16"
                ),
            },
            "source_aligned_repair": source_repairs["qwen64_vproj_mm"],
            "trajectory": trajectory(trajectories["qwen64_vproj_mm"]),
            "next_decisive_test": "use a new JOINT-repair trajectory only if persistence of this exact identified source is required",
            "evidence": [
                "results/coverage/cases/qwen64_vproj.json",
                "results/coverage/cases/qwen64_vproj_precision_decomposition.json",
                "results/coverage/cases/qwen64_vproj_source_aligned_repair.json.gz",
                "results/coverage/cases/qwen64_vproj_conditional_debias_r16.json.gz",
                "results/property/conditional_debias/qwen64_vproj.json",
                "results/coverage/cases/qwen64_vproj_repair_pilot.json",
                "results/coverage/cases/qwen64_vproj_trajectory.json",
            ],
        },
        {
            "case_id": "qwen128_vproj_mm",
            "model": "Qwen3-1.7B",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj",
            "forward_backward": {"status": "CLOSED", "scope": "one exact forward MM with actual AOT backward edges"},
            "physical_source": "global local decomposition identifies deterministic FP32-to-BF16 output rounding, not MM kernel arithmetic",
            "formation": {
                "conditional": {
                    key: qwen128_conditional[key]
                    for key in ("local", "gradient", "update")
                },
                "conditional_details": qwen128_conditional,
                "global": dict(not_measured),
                "label_source": "FIXED_STATE_STOCHASTIC_REPAIR_ENSEMBLE",
            },
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY", "P6_SEMANTIC_ORBIT_CENTERING"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM",
                "intervention": {
                    "description": "replace deterministic nearest BF16 output rounding by coordinate-wise unbiased BF16 materialization while retaining the noncoherent kernel residual",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "full_observed_source_repaired_in_expectation": True,
                    "downstream_global_carrier": source_repairs["qwen128_vproj_mm"]["downstream_carrier_effect"]["status"],
                },
                "trajectory_repairs_declared_local_source": False,
                "why_directional": (
                    "at each of 16 fixed states, deterministic nearest rounding has a nonzero "
                    "candidate-minus-debiased-ensemble mean that remains directional after the "
                    "actual backward and both declared optimizer mappings"
                ),
                "claim_boundary": (
                    "this closes conditional source formation relative to the stochastic "
                    "source-debiased ensemble; absolute downstream repair bias remains "
                    "unidentified without an exact downstream reference, and the historical "
                    "trajectory used a different repair contrast"
                ),
            },
            "bias_map": {
                "channel": "EVENT_PAIRING_ASYMMETRY",
                "status": "MATCHED_CONDITIONAL_SOURCE_SUPPORT",
                "reason": (
                    "repair local residual is centered in 16/16 fixed conditions, while "
                    "candidate-minus-repair local, gradient, SGD-update, and zero-moment "
                    "AdamW-update effects are biased in 16/16"
                ),
            },
            "source_aligned_repair": source_repairs["qwen128_vproj_mm"],
            "trajectory": trajectory(trajectories["qwen128_vproj_mm"]),
            "next_decisive_test": "add an exact downstream reference only if claiming the repaired F+B/update itself is absolutely unbiased; use a new ROUNDING_ONLY trajectory for persistence",
            "evidence": [
                "results/coverage/cases/qwen128_vproj.json",
                "results/coverage/cases/qwen128_vproj_precision_decomposition.json",
                "results/coverage/cases/qwen128_vproj_source_aligned_repair.json.gz",
                "results/coverage/cases/qwen128_vproj_conditional_debias.json.gz",
                "results/property/conditional_debias/qwen128_vproj.json",
                "results/coverage/cases/qwen128_vproj_repair_pilot.json",
                "results/coverage/cases/qwen128_vproj_trajectory.json",
            ],
        },
        {
            "case_id": "qwen_saved_p_seq128",
            "model": "Qwen3-1.7B",
            "semantic_unit": "p=softmax(a); da=p*(q-<p,q>) at layer-27 attention",
            "forward_backward": {"status": "CLOSED", "scope": "softmax forward, saved/reconstructed P, dS, and actual q/k VJPs"},
            "physical_source": "backward reconstructs P from BF16 logits plus FP32 max/sum instead of consuming true-forward FP32 P",
            "formation": {
                "conditional": dict(not_measured),
                "global": old_formation("results/property/bias_formation/formation/qwen_saved_p_seq128.json"),
                "label_source": "OPEN_LOOP_GLOBAL_V21",
                "interpretation": "global centered does not imply conditional or trajectory variance-only",
            },
            "mechanism": {
                "candidate_properties": ["P3_FORWARD_BACKWARD_NUMERICAL_CONSISTENCY", "P2_SOURCE_TRANSPORT_ALIGNMENT", "P5_OPTIMIZER_RECTIFICATION"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_CONTRACT_MECHANISM",
                "intervention": {
                    "description": "replace reconstructed P by the exact true-forward P only at dS, retain BF16 dS ABI",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "forward_loss_unchanged": True,
                },
                "why_directional": (
                    "the implementation violates a forward/saved/backward representation contract, "
                    "and Adam maps an exact +delta_g/-delta_g pair to non-antithetic updates"
                ),
                "claim_boundary": (
                    "the head-specific transport-pairing hypothesis is rejected; the contract source "
                    "and optimizer response are supported, while unrelated-state global centering "
                    "remains compatible with trajectory-conditioned bias"
                ),
            },
            "bias_map": {
                "channel": "RESPONSE_RECTIFICATION",
                "status": "MATCHED_SUPPORT",
                "reason": (
                    "equal and opposite gradient residuals at identical weights and Adam moments "
                    "produce a nonzero response-even update component"
                ),
                "accumulated_nonoddness_ratio": saved_p_symmetry["optimizer_oddness_resultant_ratio"],
                "mean_step_nonoddness_ratio": saved_p_symmetry["mean_step_optimizer_oddness_ratio"],
                "natural_antithetic_resultant_cosine": saved_p_symmetry["natural_antithetic_update_resultant_cosine"],
                "mean_step_sign_crossing_fraction": saved_p_symmetry.get("mean_step_sign_crossing_fraction"),
                "mean_step_delta_energy_on_sign_crossings": saved_p_symmetry.get("mean_step_delta_energy_on_sign_crossings"),
                "mean_step_response_even_energy_on_sign_crossings": saved_p_symmetry.get("mean_step_response_even_energy_on_sign_crossings"),
                "energy_weighted_response_even_on_sign_crossings": saved_p_symmetry.get("energy_weighted_response_even_on_sign_crossings"),
                "response_even_energy_in_first_two_steps": saved_p_symmetry.get("response_even_energy_in_first_two_steps"),
                "step_integrated_response_even_energy_fraction": saved_p_symmetry.get("step_integrated_response_even_energy_fraction"),
                "rejected_subhypothesis": (
                    "rolling dS residuals across heads suppresses the gradient resultant by only "
                    f"{100.0 * saved_p_symmetry['gradient_pairing_suppression']:.2f}% and increases "
                    "the update resultant"
                ),
            },
            "trajectory": {
                **trajectory(trajectories["qwen_saved_p_seq128"]),
                "local_feedback": symmetric_consequence(
                    "results/property/seup_mainline/qwen_softmax_seup.json"
                ),
            },
            "next_decisive_test": "derive a coordinate/state susceptibility predictor for the measured Adam even response",
            "evidence": [
                "results/coverage/cases/qwen128_softmax_fb.json",
                "results/coverage/cases/qwen128_softmax_fb_formal.json",
                "results/property/bias_formation/formation/qwen_saved_p_seq128.json",
                "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json",
                saved_p_symmetry_path,
            ],
        },
        {
            "case_id": "qwen3vl_silu_layer0",
            "model": "Qwen3-VL-2B",
            "semantic_unit": "y=x*sigmoid(x); dx=q*sigmoid(x)*(1+x*(1-sigmoid(x)))",
            "forward_backward": {"status": "CLOSED", "scope": "same forward and one exact layer-0 SiLU backward invocation"},
            "physical_source": "AOT graph-dtype elementary backward arithmetic differs from native aten.silu_backward arithmetic",
            "formation": {"conditional": dict(not_measured), "global": dict(not_measured), "label_source": "NOT_MEASURED"},
            "mechanism": {
                "candidate_properties": ["P4_NONLINEAR_RECTIFICATION", "P3_FORWARD_BACKWARD_NUMERICAL_CONSISTENCY", "P5_OPTIMIZER_RECTIFICATION"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_OPTIMIZER_RESPONSE_MECHANISM",
                "intervention": {
                    "description": "use the exact natural delta_g and its negation around the native-SiLU repair gradient at identical Adam state",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                },
                "why_directional": (
                    "the backward implementation supplies delta_g, and Adam maps the exact "
                    "+delta_g/-delta_g pair to almost orthogonal rather than opposite update resultants"
                ),
                "claim_boundary": "optimizer response rectification is closed; the arithmetic origin inside the decomposed backward remains case-specific",
            },
            "bias_map": {
                "channel": "RESPONSE_RECTIFICATION",
                "status": "MATCHED_INDEPENDENT_REPLICATION",
                "reason": "an exact antithetic gradient pair produces a nonzero Adam response-even component",
                "accumulated_nonoddness_ratio": silu_symmetry["optimizer_oddness_resultant_ratio"],
                "mean_step_nonoddness_ratio": silu_symmetry["mean_step_optimizer_oddness_ratio"],
                "natural_antithetic_resultant_cosine": silu_symmetry["natural_antithetic_update_resultant_cosine"],
                "mean_step_sign_crossing_fraction": silu_symmetry.get("mean_step_sign_crossing_fraction"),
                "mean_step_delta_energy_on_sign_crossings": silu_symmetry.get("mean_step_delta_energy_on_sign_crossings"),
                "mean_step_response_even_energy_on_sign_crossings": silu_symmetry.get("mean_step_response_even_energy_on_sign_crossings"),
                "energy_weighted_response_even_on_sign_crossings": silu_symmetry.get("energy_weighted_response_even_on_sign_crossings"),
                "response_even_energy_in_first_two_steps": silu_symmetry.get("response_even_energy_in_first_two_steps"),
                "step_integrated_response_even_energy_fraction": silu_symmetry.get("step_integrated_response_even_energy_fraction"),
            },
            "trajectory": trajectory(trajectories["qwen3vl_silu_layer0"]),
            "next_decisive_test": "derive a coordinate/state susceptibility predictor shared with saved-P",
            "evidence": [
                "results/round2/vl_silu_cause.json",
                "results/round2/vl_silu_cause_fp32.json",
                "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
                silu_symmetry_path,
            ],
        },
        {
            "case_id": "mamba_seq64_input_proj",
            "model": "Mamba-130M",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at layer-0 in_proj",
            "forward_backward": {"status": "CLOSED", "scope": "one exact recurrent input-projection MM and actual VJP edges"},
            "physical_source": "both same-operand MM kernel arithmetic and deterministic output rounding are directional",
            "formation": {
                "conditional": {
                    key: mamba_conditional[key]
                    for key in ("local", "gradient", "update")
                },
                "conditional_details": mamba_conditional,
                "global": dict(not_measured),
                "label_source": "FIXED_STATE_STOCHASTIC_REPAIR_ENSEMBLE",
            },
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY"],
                "verdict": "PARTIAL_SOURCE_MECHANISM",
                "intervention": {
                    "description": "kernel-only, rounding-only, and joint factorial arms; joint uses FP32 MM plus coordinate-wise unbiased BF16 materialization",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "full_observed_source_repaired_in_expectation": True,
                    "evidence_scope": source_repairs["mamba_seq64_input_proj"]["evidence_scope"],
                    "downstream_carrier": source_repairs["mamba_seq64_input_proj"]["downstream_carrier_effect"]["status"],
                },
                "why_directional": (
                    "the joint repair centers the declared local source in all 16 fixed "
                    "conditions, while the natural local effect and zero-moment AdamW "
                    "effect are biased in all 16; the actual backward/SGD effect is biased "
                    "in 13 conditions and unresolved in three"
                ),
                "claim_boundary": (
                    "cross-architecture conditional local-source and bounded optimizer "
                    "evidence; the all-layer mechanism gate remains unresolved because "
                    "three real-backward conditions do not obtain a directional verdict, "
                    "and the historical trajectory closes only the KERNEL_ONLY arm"
                ),
            },
            "bias_map": {
                "channel": "EVENT_PAIRING_ASYMMETRY",
                "status": "MATCHED_LOCAL_SOURCE_MIXED_F_B_ADAM_DIRECTIONAL",
                "reason": (
                    "repair local residual is centered 16/16 and candidate local plus "
                    "zero-moment AdamW effects are biased 16/16; gradient/SGD are biased "
                    "13/16 and unresolved 3/16, so the complete conditional chain fails "
                    "closed rather than being promoted"
                ),
            },
            "source_aligned_repair": source_repairs["mamba_seq64_input_proj"],
            "trajectory": trajectory(trajectories["mamba_seq64_input_proj"]),
            "next_decisive_test": "only if this partial case is revisited, use an exact gradient antithetic control in the three unresolved conditions; do not add repeats or relax the gate",
            "evidence": [
                "results/coverage/cases/mamba_seq64_input_proj.json",
                "results/coverage/cases/mamba_seq64_input_proj_precision_decomposition.json",
                "results/coverage/cases/mamba_seq64_input_proj_source_aligned_repair.json.gz",
                "results/coverage/cases/mamba_seq64_input_proj_conditional_debias.json.gz",
                "results/property/conditional_debias/mamba_seq64_input_proj.json",
                "results/coverage/cases/mamba_seq64_input_proj_repair_pilot.json",
                "results/coverage/cases/mamba_seq64_input_proj_trajectory.json",
            ],
        },
        {
            "case_id": "qwen_layer23_attention_state",
            "model": "Qwen3-1.7B",
            "semantic_unit": "S_bwd=alpha*J_softmax(P)^T(DV^T); Gq=S_bwd*K; dWq=Gq^T H",
            "forward_backward": {"status": "CLOSED", "scope": "layer-23 q_proj attention-state semantic region and exact tile carrier"},
            "physical_source": "attention-backward state S_bwd is causal; upstream contributors overlap and include delayed key materialization",
            "formation": {"conditional": dict(not_measured), "global": dict(not_measured), "label_source": "SEMANTIC_REGION_CAUSAL_EVIDENCE"},
            "mechanism": {
                "candidate_properties": ["P2_SOURCE_TRANSPORT_ALIGNMENT", "P3_FORWARD_BACKWARD_NUMERICAL_CONSISTENCY", "P6_SEMANTIC_ORBIT_CENTERING"],
                "verdict": "SUPPORTED_SEMANTIC_REGION_TRANSPORT_CONTRACT_MECHANISM",
                "intervention": {
                    "description": "restore S_bwd at bmm_76; K-only repair is insufficient; joint S/K repair closes the direction",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                },
                "why_directional": (
                    "the changed attention state is transported through Gq=S_bwd*K into a fixed q_proj tile; "
                    "S_bwd restoration removes that carrier"
                ),
                "claim_boundary": "validated semantic-region mechanism, not a uniquely identified kernel instruction",
            },
            "bias_map": {
                "channel": "UNRESOLVED_MIXED_CHANNEL",
                "status": "PROJECTED_ANTITHETIC_RESPONSE_NATURAL_FIDELITY_FAILED",
                "reason": (
                    "all 16 projected BF16 +/-epsilon pairs and shams are exact, and "
                    f"the projected F+B response-even ratio spans "
                    f"{min(l23_gradient_even):.3f}--{max(l23_gradient_even):.3f}, while "
                    f"zero-moment AdamW spans {min(l23_adam_even):.3f}--"
                    f"{max(l23_adam_even):.3f}; "
                    "however, natural-source fidelity falls below the frozen 90% gate in "
                    "some conditions, so this cannot upgrade the natural layer-23 case to "
                    "a marginal-preserving matched mechanism"
                ),
                "antithetic_status": l23_antithetic["status"],
                "validity_gates": l23_antithetic["validity_gates"],
                "mechanism_gates": l23_antithetic["mechanism_gates"],
            },
            "trajectory": trajectory(trajectories["qwen_layer23_attention_state"]),
            "next_decisive_test": "retain as a bounded semantic-region mechanism; do not relax the failed natural-fidelity gate or force single-kernel attribution",
            "evidence": [
                "results/coverage/cases/l23_qproj_attention_state_region.json",
                "results/final/l23_attention_live_weight.json",
                "results/property/bias_formation_final/qwen_l23_attention_mechanism.json",
                "results/final/l23_s_bwd_antithetic.json",
            ],
        },
    ]
    validate_audit(result)
    for case in result:
        case["formation"]["first_conditional_bias_stage"] = first_conditional_bias_stage(case)
    return result


def report(case: dict[str, Any]) -> str:
    cond = case["formation"]["conditional"]
    glob = case["formation"]["global"]
    mechanism = case["mechanism"]
    bias_map = case["bias_map"]
    evidence = "\n".join(f"- `{path}`" for path in case["evidence"])
    local_feedback = case["trajectory"].get("local_feedback")
    consequence = ""
    if local_feedback:
        consequence = (
            "\n\n对称四反事实 recurrence 已测：local accumulation L2 "
            f"`{local_feedback['local_accumulation_l2']}`，feedback accumulation L2 "
            f"`{local_feedback['feedback_accumulation_l2']}`，最大相对闭合残差 "
            f"`{local_feedback['max_recurrence_relative_residual']}`。"
        )
    return f"""# {case['case_id']}

## 数学单位

模型：{case['model']}

F+B：{case['semantic_unit']}

闭合范围：{case['forward_backward']['scope']}（{case['forward_backward']['status']}）。

## 统一 Bias Formation Map

对预先声明的反对称操作，将事件分布写成 `p=p_s+p_a`，将真实 F+B/optimizer 响应写成 `F=F_e+F_o`。精确形成式是：

`E[F(ε)|c] = ∫p_s(ε)F_e(ε)dε + ∫p_a(ε)F_o(ε)dε`。

本例归入：`{bias_map['channel']}`（`{bias_map['status']}`）。{bias_map['reason']}。

本例的物理差异是：{case['physical_source']}。

条件化 formation（local / gradient / update）：`{cond['local']} / {cond['gradient']} / {cond['update']}`。

旧跨无关状态结果（local / gradient / update）：`{glob['local']} / {glob['gradient']} / {glob['update']}`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`{mechanism['verdict']}`。

原因：{mechanism['why_directional']}。

干预：{mechanism['intervention']['description']}。

边界：{mechanism['claim_boundary']}。

## 轨迹后果

separation：`{case['trajectory']['separation_status']}`；directional persistence：`{case['trajectory']['directional_persistence']}`。共 {case['trajectory']['steps']} steps，drift norm `{case['trajectory']['initial_drift_norm']}` → `{case['trajectory']['final_drift_norm']}`。

formation contrast：`{case['trajectory']['formation_contrast']}`；trajectory contrast：`{case['trajectory']['trajectory_contrast']}`；alignment：`{case['trajectory']['contrast_alignment']}`；same-contrast full chain：`{case['trajectory']['same_contrast_full_chain']}`。

参数距离增长只证明 causal separation，不单独证明方向性 persistence，也不提供 formation 标签。{consequence}

## 下一项决定性实验

{case['next_decisive_test']}。

## 证据

{evidence}
"""


def main() -> None:
    rows = cases()
    for case in rows:
        for relative in case["evidence"]:
            if not (ROOT / relative).is_file():
                raise FileNotFoundError(f"{case['case_id']}: {relative}")

    protocol = {
        "schema": "kernel-analyzer-systematic-bias-audit-protocol-v2",
        "status": "UNIFIED_EIGHT_CASE_EVIDENCE_AUDIT",
        "question": "Which F+B implementation differences form conditional bias, which persist directionally, and which close both claims under one repair contrast?",
        "case_denominator": 8,
        "mechanism_family_clusters": 7,
        "equations": {
            "exact_response": "F_c(epsilon) = U_s(z_repair+epsilon)-U_s(z_repair)",
            "formation": "E[F(epsilon)|c] = integral p_s(epsilon)F_e(epsilon) + integral p_a(epsilon)F_o(epsilon)",
            "event_pairing_channel": "integral p_a(epsilon)F_o(epsilon)",
            "response_rectification_channel": "integral p_s(epsilon)F_e(epsilon)",
            "optimizer": "delta_u = Opt(g+delta_g,z)-Opt(g,z)",
            "trajectory": "D_(t+1) = D_t + L_t + B_t + recurrence_residual_t",
        },
        "rules": [
            "Analyze one complete forward plus its actual backward as the unit.",
            "Use conditional, trajectory, and global bias as separate claims.",
            "A global centered result is not a conditional null.",
            "A trajectory is consequence evidence and never labels formation.",
            "A growing parameter-distance norm is causal separation, not directional persistence by itself.",
            "Flash-style persistence requires a predeclared or calibration-frozen directional gate on the live trajectory.",
            "A supported mechanism requires a causal intervention and exact matched sham.",
            "Do not join evidence produced by different repaired contrasts.",
            "SEUP describes persistence only after formation evidence exists.",
            "The semantic antithetic operation must be declared from the F+B boundary, not fitted to drift.",
        ],
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "case_audit.json", {"schema": "kernel-analyzer-systematic-bias-audit-v2", "cases": rows})

    matrix_rows = []
    for case in rows:
        cond = case["formation"]["conditional"]
        glob = case["formation"]["global"]
        matrix_rows.append({
            "case_id": case["case_id"],
            "model": case["model"],
            "fb": case["forward_backward"]["status"],
            "conditional_local": cond["local"],
            "conditional_gradient": cond["gradient"],
            "conditional_update": cond["update"],
            "global_local": glob["local"],
            "global_gradient": glob["gradient"],
            "global_update": glob["update"],
            "mechanism_verdict": case["mechanism"]["verdict"],
            "bias_map_channel": case["bias_map"]["channel"],
            "bias_map_status": case["bias_map"]["status"],
            "trajectory_separation": case["trajectory"]["separation_status"],
            "directional_persistence": case["trajectory"]["directional_persistence"],
            "contrast_alignment": case["trajectory"]["contrast_alignment"],
            "same_contrast_full_chain": case["trajectory"]["same_contrast_full_chain"],
            "next_decisive_test": case["next_decisive_test"],
        })
    matrix_path = OUT / "case_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = matrix_path.with_name(".case_matrix.csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix_rows)
    temporary.replace(matrix_path)

    reports = OUT / "case_reports"
    reports.mkdir(parents=True, exist_ok=True)
    for case in rows:
        (reports / f"{case['case_id']}.md").write_text(report(case), encoding="utf-8")

    counts: dict[str, int] = {}
    for case in rows:
        verdict = case["mechanism"]["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
    gap_order = [
        "qwen128_vproj_mm", "qwen_saved_p_seq128", "qwen3vl_silu_layer0",
        "qwen64_vproj_mm", "mamba_seq64_input_proj", "phi4_seq64_lmhead_dx",
        "liger_fused_ce", "qwen_layer23_attention_state",
    ]
    write_json(OUT / "gap_plan.json", {
        "schema": "kernel-analyzer-systematic-bias-gap-plan-v2",
        "principle": "run only experiments that resolve a specific missing link in the two-channel parity equation",
        "ordered_cases": [
            {
                "case_id": case_id,
                "experiment": next(row["next_decisive_test"] for row in rows if row["case_id"] == case_id),
            }
            for case_id in gap_order
        ],
    })

    directional_cases = [
        row["case_id"] for row in rows
        if row["trajectory"]["directional_persistence"] == "CONFIRMED"
    ]
    separation_only_cases = [
        row["case_id"] for row in rows
        if row["trajectory"]["directional_persistence"] != "CONFIRMED"
    ]
    full_chain_cases = [
        row["case_id"] for row in rows
        if row["trajectory"]["same_contrast_full_chain"] is True
    ]
    summary = f"""# 八案例统一证据审计

## 结论

“8 个案例”是审计分母，不是“8 个持久性 bias”。统一口径得到四个互不替代的计数：

- **8/8 causal paired separation artifacts**：都有闭合 F+B、repair/sham 和 live-weight 参数距离；这只证明 implementation contrast 会让两臂分开。
- **6/8 directional-persistence positives**：`{', '.join(directional_cases)}` 通过了轨迹局部的预声明/冻结方向门。
- **6/8 matched formation-mechanism positives**：Liger、Phi、Qwen64/128 `v_proj`、saved-P、Qwen3-VL SiLU；Mamba 和 layer-23 在当前 antithetic formation protocol 下保留 partial/unresolved。
- **4/8 same-contrast full chains**：`{', '.join(full_chain_cases)}` 将当前 formation mechanism 和 directional persistence 用同一个或闭合语义超集 repair 串起来。

`{', '.join(separation_only_cases)}` 只有 causal separation，没有确认的方向性 persistence；不得称为完整 Flash-style case。

这四个集合故意不相同。一个案例可以形成条件 bias 但未证明持续，也可以有旧 T1--T4 持久轨迹但当前 formation 干预使用了另一种 repair。

## Formation mechanism

- Liger、Phi：`EVENT_PAIRING_ASYMMETRY` 的 matched positives；
- saved-P、Qwen3-VL SiLU：`RESPONSE_RECTIFICATION` 的两个独立 matched positives；
- layer-23：semantic-region transport/contract mechanism，不是单 kernel root；
- Qwen128：在 16 个固定 state 中，repair local residual 全部 centered，而 candidate-minus-repair 的 local、真实 gradient、SGD update 与 zero-moment AdamW update 全部 biased；这是新的 conditional source-formation positive；
- Qwen64：独立 16-repeat fixed-state confirmation 中，repair local residual 在 16/16 conditions centered，而 candidate-minus-repair 的 local、真实 gradient、SGD update 与 zero-moment AdamW update 均在 16/16 biased；
- Mamba：16-condition joint-repair confirmation 得到 local 与 zero-moment AdamW 16/16 biased、repair local 16/16 centered，但真实 gradient/SGD 为 13/16 biased、3/16 unresolved；因此仍是 partial，而不是第七个 matched positive；
- layer-23：16-condition exact projected-antithetic control 揭示 F+B 与 Adam response-even 分量，但自然 source fidelity 在部分条件未过冻结的 90% gate，因此不升级；
- 因此严格 formation-mechanism positives 是 6/8；Qwen64/128 的 fixed-state formation、Mamba 的 JOINT formation 与各自历史 KERNEL_ONLY 轨迹不能拼接。

## 为什么会出现系统性 bias

统一解释不是“任何环节都可能有偏”，而是一个精确的奇偶分解。固定训练条件 `c` 和从 F+B 数学边界预先声明的 `ε→-ε`，令 `p_s/p_a` 是事件分布的对称/反对称部分，`F_e/F_o` 是真实 F+B+optimizer 响应的偶/奇部分：

`E[F(ε)|c] = ∫p_sF_e + ∫p_aF_o`。

- `∫p_aF_o` 是事件/配对失衡：相反 residual 没有以相同条件质量出现，或 residual 与 transport 的真实配对不具反对称闭包。Liger 与 Phi 分别给出 schedule/source 和 composite transport 的 matched evidence。
- `∫p_sF_e` 是响应整流：即使人为构造严格等范数、反号的 `+δg/-δg`，真实映射仍不满足 `F(-δg)=-F(+δg)`。saved-P 与 SiLU 在相同 Adam state 下独立复现这一项，累计 non-oddness ratio 分别为 `0.6817` 与 `0.6956`。
- 按 response-even 能量加权，saved-P 与 SiLU 分别有 `99.48%` 和 `99.87%` 的偶分量落在梯度符号穿越坐标；两者又分别有 `99.51%` 和 `>99.99%` 的偶分量能量出现在前两步。这把响应整流定位到 Adam 冷启动时的小梯度/符号边界，而不是笼统归因于“优化器非线性”。
- 若事件配对闭合且响应为奇函数，两项同时为零，variance 无论多大都不会产生条件均值。这才是可以区分“有 bias/无 bias”的安全 property。
- 进入轨迹后，`D_(t+1)=D_t+L_t+B_t+r_t`；local effect 与 feedback 决定差异持续还是抵消。固定 global carrier 不是必要条件。

## 当前可以声称什么

可以声称：在六个具有 matched formation 证据的独立 F+B 案例中，conditional training bias 均对应条件反对称消除失败；失败来自 source/event/transport 配对失衡或真实 optimizer 响应的偶分量。另有六个案例具有轨迹局部方向性 persistence，但只有四个把当前 formation 机制与 persistence 在相同 contrast 下闭合。error norm、raw tensor mean、BF16 dtype 与跨无关 state 的固定 global carrier 都不是统一判据。

不能声称：8/8 都是持久性 bias；不能把 basis-free 参数距离增长自动叫 directional bias；不能把 local source repair 自动升级成完整 F+B/optimizer 去偏，也不能把旧 accumulation trajectory 与新的 rounding/joint repair 拼成同一条轨迹因果链。

## 下一步

这一轮 fixed-state conditional audit 已完成。结果不支持通过继续增加随机重复来强行统一 Mamba 或 layer-23；只有获得 exact downstream reference，才能签发 repair 自身的 downstream zero-bias certificate。后续若继续主线，应把两个 formation channel 自动化为 event antithetic closure 与 `(g,m,v,δg)` response-even susceptibility；个案边界见 `gap_plan.json`。
"""
    (OUT / "scientific_summary.md").write_text(summary, encoding="utf-8")
    write_json(OUT / "summary.json", {
        "status": "COMPLETE_UNIFIED_EIGHT_CASE_EVIDENCE_AUDIT",
        "cases": len(rows),
        "paired_separation_cases": len(rows),
        "directional_persistence_cases": len(directional_cases),
        "directional_persistence_case_ids": directional_cases,
        "separation_only_case_ids": separation_only_cases,
        "same_contrast_full_chain_cases": len(full_chain_cases),
        "same_contrast_full_chain_case_ids": full_chain_cases,
        "mechanism_family_clusters": len({row["mechanism_family"] for row in trajectory_index().values()}),
        "mechanism_verdict_counts": counts,
        "cross_case_property": "EFFECTIVE_ANTITHETIC_SYMMETRY_WORKING_PROPERTY",
        "matched_property_cases": 6,
        "property_channels": {
            "EVENT_PAIRING_ASYMMETRY": ["liger_fused_ce", "phi4_seq64_lmhead_dx", "qwen64_vproj_mm", "qwen128_vproj_mm"],
            "RESPONSE_RECTIFICATION": ["qwen_saved_p_seq128", "qwen3vl_silu_layer0"],
        },
        "next_work": "PROPERTY_AUTOMATION_WITH_PARTIAL_CASES_PRESERVED_FAIL_CLOSED",
    })
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "cases": len(rows), "verdicts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
