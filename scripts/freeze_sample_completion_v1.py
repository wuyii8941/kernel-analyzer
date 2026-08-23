#!/usr/bin/env python3
"""Freeze the sample-completion protocol and its pre-measurement roster.

This script deliberately records *eligibility and provenance*, not scientific
labels.  Existing T1--T4/SEUP verdicts are copied only as historical evidence;
they are never used to select the new search units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/sample_completion_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def base_cases() -> list[dict[str, Any]]:
    """The eight predeclared cases from the completion plan.

    A row can be a complete historical case and still require a uniform export
    for this campaign.  That distinction is intentional.
    """
    rows = [
        ("liger_fused_ce_t128", "POSITIVE", "fused_reduction_accumulation", "qwen3_1p7b", 128,
         ["results/coverage/existing_case_reaudit.json", "results/property/joint_bias_formation_v1/three_stage_summary.json"]),
        ("phi4_lm_head_dx_seq64", "POSITIVE", "lm_head_vjp_gemm", "phi4", 64,
         ["results/coverage/cases/phi4_seq64_lmhead_dx.json", "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json"]),
        ("qwen_seq256_lm_head_dx", "POSITIVE", "lm_head_vjp_gemm", "qwen3_1p7b", 256,
         ["results/property/persistence_v1/confirmation/qwen256_lmhead.json"]),
        ("qwen128_layer0_vproj_output", "CONTROL", "projection_rounding", "qwen3_1p7b", 128,
         ["results/coverage/cases/qwen128_vproj.json", "results/coverage/cases/qwen128_vproj_trajectory.json"]),
        ("qwen_saved_p_seq128", "CONTROL", "saved_state_reconstruction", "qwen3_1p7b", 128,
         ["results/coverage/cases/qwen128_softmax_fb.json", "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json"]),
        ("qwen3vl_silu_backward", "CONTROL", "pointwise_backward", "qwen3_vl_reranker_2b", 160,
         ["results/coverage/cases/qwen3vl_layer0_silu_trajectory.json", "results/round2/vl_bias.json"]),
        ("gemma4_e2b_ple_rmsnorm", "CONTROL", "normalization_backward", "gemma4_e2b", 128,
         ["results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_formation16.json", "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_consequence32.json"]),
        ("qwen_bmm_seq64", "CONTROL", "attention_bmm", "qwen3_1p7b", 64,
         ["results/property/seup_geometry_followup/qwen_bmm_seq64_geometry.json", "results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz"]),
    ]
    result = []
    for case_id, role, family, model, seq, artifacts in rows:
        result.append({
            "case_id": case_id,
            "role": role,
            "implementation_family": family,
            "model": model,
            "sequence_length": seq,
            "selection_rule": "predeclared_base_case_from_sample_completion_plan",
            "historical_artifacts": [
                {"path": p, "exists": (ROOT / p).exists(),
                 "sha256": sha256_file(ROOT / p) if (ROOT / p).is_file() else None}
                for p in artifacts
            ],
            "uniform_measurement_status": "NOT_YET_UNIFIED",
            "scientific_label": None,
        })
    return result


def search_units() -> list[dict[str, Any]]:
    """Freeze 16 semantic units without using outcome labels for selection."""
    rows = [
        ("qwen_seq64_ce_dlogits", "LOSS_BACKWARD", "qwen3_1p7b", 64, "results/property/bias_formation/hotspot_search/qwen_seq64_matrix.json"),
        ("qwen_seq64_l27_q_norm_vjp", "NORMALIZATION_BACKWARD", "qwen3_1p7b", 64, "results/property/bias_formation/hotspot_search/qwen_seq64_matrix.json"),
        ("qwen_seq64_l27_k_norm_vjp", "NORMALIZATION_BACKWARD", "qwen3_1p7b", 64, "results/property/bias_formation/hotspot_search/qwen_seq64_matrix.json"),
        ("qwen_seq64_l23_softmax_vjp", "ATTENTION_BACKWARD", "qwen3_1p7b", 64, "results/property/bias_formation/hotspot_search/qwen_seq64_matrix.json"),
        ("deepseek8b_seq64_ce_dlogits", "LOSS_BACKWARD", "deepseek8b", 64, "results/property/bias_formation/hotspot_search/deepseek8b_seq64_rescreen/deepseek8b_seq64_ce_dlogits.json"),
        ("deepseek8b_seq64_final_norm_vjp", "NORMALIZATION_BACKWARD", "deepseek8b", 64, "results/property/bias_formation/hotspot_search/deepseek8b_seq64_rescreen/deepseek8b_seq64_final_norm_vjp.json"),
        ("deepseek8b_seq64_l35_attention_dv", "ATTENTION_BACKWARD", "deepseek8b", 64, "results/property/bias_formation/hotspot_search/deepseek8b_seq64_rescreen/deepseek8b_seq64_l35_attention_dv.json"),
        ("deepseek8b_seq64_l35_softmax_vjp", "ATTENTION_BACKWARD", "deepseek8b", 64, "results/property/bias_formation/hotspot_search/deepseek8b_seq64_rescreen/deepseek8b_seq64_l35_softmax_vjp.json"),
        ("phi4_seq64_ce_dlogits", "LOSS_BACKWARD", "phi4", 64, "results/property/bias_formation/hotspot_search/phi4_seq64_rescreen/phi4_seq64_ce_dlogits.json"),
        ("phi4_seq64_final_norm_vjp", "NORMALIZATION_BACKWARD", "phi4", 64, "results/property/bias_formation/hotspot_search/phi4_seq64_rescreen/phi4_seq64_final_norm_vjp.json"),
        ("phi4_seq64_l31_attention_dv", "ATTENTION_BACKWARD", "phi4", 64, "results/property/bias_formation/hotspot_search/phi4_seq64_rescreen/phi4_seq64_l31_attention_dv.json"),
        ("phi4_seq64_l31_softmax_vjp", "ATTENTION_BACKWARD", "phi4", 64, "results/property/bias_formation/hotspot_search/phi4_seq64_rescreen/phi4_seq64_l31_softmax_vjp.json"),
        ("mamba_seq64_ce_dlogits", "LOSS_BACKWARD", "mamba", 64, "results/property/bias_formation/hotspot_search/mamba_seq64_capture_plan.json"),
        ("mamba_seq64_final_norm_vjp", "NORMALIZATION_BACKWARD", "mamba", 64, "results/property/bias_formation/hotspot_search/mamba_seq64_capture_plan.json"),
        ("mamba_seq64_scan_recurrence", "STATE_SPACE_BACKWARD", "mamba", 64, "results/property/bias_formation/hotspot_search/mamba_seq64_capture_plan.json"),
        ("mamba_seq64_scan_reduction", "STATE_SPACE_BACKWARD", "mamba", 64, "results/property/bias_formation/hotspot_search/mamba_seq64_capture_plan.json"),
    ]
    result = []
    for case_id, family, model, seq, artifact in rows:
        result.append({
            "case_id": case_id,
            "role": "SEARCH_UNIT",
            "implementation_family": family,
            "model": model,
            "sequence_length": seq,
            "selection_rule": "semantic_bottleneck_and_model_family_before_outcome_reveal",
            "selection_inputs_excluded": ["T1", "T2", "T3", "T4", "SEUP", "trajectory_drift", "error_magnitude"],
            "binding_artifact": artifact,
            "binding_artifact_exists": (ROOT / artifact).exists(),
            "uniform_measurement_status": "NOT_YET_UNIFIED",
            "screen_status": "NOT_STARTED",
            "scientific_label": None,
        })
    return result


def build_protocol(head: str) -> dict[str, Any]:
    return {
        "schema": "kernel-analyzer-sample-completion-protocol-v1",
        "protocol_id": "sample_completion_v1",
        "status": "FROZEN_BEFORE_NEW_MEASUREMENT",
        "freeze_commit": head,
        "scientific_goal": {
            "uniform_cases_minimum": 20,
            "valid_nonzero_reachable_controls_minimum": 15,
            "independent_positive_families_minimum": 3,
            "heldout_units_minimum": 8,
            "heldout_models_minimum": 2,
            "heldout_new_implementation_classes_minimum": 2,
        },
        "valid_case_requirements": [
            "candidate_and_repair_bind_one_exact_operator_or_closed_semantic_region",
            "repair_changes_only_declared_target_implementation",
            "candidate_and_repair_share_weights_batch_rng_and_optimizer_state",
            "nonzero_candidate_repair_difference",
            "difference_reaches_declared_parameter_gradient",
            "forward_and_actual_backward_are_bound",
        ],
        "measurement_stages": {
            "operator_output_error": {"required": True, "steps": [8, 16, 32]},
            "parameter_gradient_error": {"required": True, "steps": [8, 16, 32]},
            "optimizer_update_error": {"required": True, "optimizers": ["SGD", "AdamW"], "steps": [8, 16, 32]},
            "paired_parameter_drift": {"required": True, "steps": [32]},
            "feedback_decomposition": {"required": True, "steps": [32]},
        },
        "screening_schedule": {
            "engineering": 2,
            "prescreen": 8,
            "short_prediction": 16,
            "final_label": 32,
            "final_label_rule": "only_32_step_complete_trace_can_assign_case_label",
            "random_unflagged_32_step_controls": {"min": 8, "max": 10},
        },
        "oracle": {
            "input_steps": [1, 16],
            "forbidden_inputs": ["32_step_label", "final_drift", "historical_case_name", "T4_verdict", "SEUP_verdict", "post_step_16_measurements"],
            "outputs": ["ESCALATE_TO_32_STEP", "NO_ESCALATION_UNDER_PROTOCOL", "ABSTAIN_MISSING_REQUIRED_MEASUREMENT"],
            "safe_label_forbidden": True,
            "baselines": ["local_rms", "dtype", "reduction_length", "8_or_16_step_magnitude"],
        },
        "causal_interventions": [
            "accumulator_precision_or_legal_reduction_order",
            "saved_state_or_exact_recompute_with_matched_sham",
            "SGD_AdamW_moment_precision_or_reset",
        ],
        "label_policy": {
            "historical_verdicts_are_not_reused_as_labels": True,
            "unresolved_is_retained_in_denominator": True,
            "out_of_domain_is_not_negative": True,
            "no_case_or_threshold_changes_after_measurement": True,
        },
        "completion_gate": [
            "at_least_20_uniform_complete_cases",
            "at_least_15_valid_nonzero_parameter_reachable_controls",
            "three_positive_families_or_real_kernel_causal_construct",
            "frozen_heldout_validation",
            "oracle_vs_baselines",
            "recall_false_positive_abstain_and_gpu_cost",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    head = git_head()
    protocol = build_protocol(head)
    roster = {
        "schema": "kernel-analyzer-sample-completion-roster-v1",
        "status": "FROZEN_BEFORE_NEW_MEASUREMENT",
        "freeze_commit": head,
        "base_case_count": 8,
        "search_unit_count": 16,
        "case_count": 24,
        "base_cases": base_cases(),
        "search_units": search_units(),
        "claim_boundary": "The roster is a measurement plan. It assigns no new scientific labels.",
    }
    snapshot = {
        "schema": "kernel-analyzer-sample-completion-existing-evidence-snapshot-v1",
        "status": "PRE_CAMPAIGN_AUDIT",
        "models_with_actual_artifacts": 10,
        "systematic_census_models": 4,
        "systematic_census_cells": 12,
        "systematic_census": {
            "eager_invocations": 466419,
            "candidate_invocations": 70171,
            "primary_fb_proof_units": 186807,
            "full_coordinate_t1_endpoints": 1562,
            "t1_pass": 1390,
            "t1_reject": 172,
            "t1_pending": 0,
        },
        "headline_source_persistent_cases": 3,
        "screen_negative_complete_consequence_sample": 12,
        "current_sample_completion_uniform_cases": 0,
        "current_sample_completion_uniform_controls": 0,
        "current_sample_completion_heldout_confirmation": "NOT_STARTED",
        "mamba_fresh_timing": "UNRESOLVED_BLOCKED",
        "claim_boundary": "Existing coverage is not substituted for the new uniform 20-case sample.",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "roster.json").write_text(json.dumps(roster, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "existing_evidence_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "cases": 24, "freeze_commit": head}, sort_keys=True))


if __name__ == "__main__":
    main()
