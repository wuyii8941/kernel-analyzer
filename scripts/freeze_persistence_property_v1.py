#!/usr/bin/env python3
"""Freeze the non-circular persistence-property v1 protocol and roster."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/property/persistence_v1"


def commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def write(name: str, payload: object) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    protocol = {
        "schema": "kernel-analyzer-persistence-property-protocol-v1",
        "status": "FROZEN_BEFORE_V1_MEASUREMENT",
        "protocol_freeze_parent_commit": commit(),
        "objective": (
            "Test whether low-cost source/orbit/transport and feedback probes predict "
            "implementation-induced effective-update persistence."
        ),
        "claim_tracks": {
            "source_transport": {
                "question": "Does a local implementation residual survive as a non-canceling effective update?",
                "candidate_predictors": [
                    "orbit_mean_temporal_amplification",
                    "default_schedule_residual_temporal_amplification",
                    "reference_vjp_transported_amplification",
                ],
            },
            "feedback": {
                "question": "Does closed-loop model/optimizer state amplify an otherwise diffusive local update?",
                "candidate_predictors": [
                    "optimizer_response_evenness",
                    "moment_state_dependence",
                    "matched_residual_feedback_gain",
                ],
            },
        },
        "ordered_states": 16,
        "semantic_orbit_variants": 8,
        "max_reported_lag": 8,
        "sign_flip_null": {"draws": 4000, "seed": 20260820},
        "measurement_levels": ["epsilon", "gradient", "local", "feedback", "actual"],
        "primary_continuous_measurement": (
            "norm(sum_t x_t) / sqrt(sum_t norm(x_t)^2)"
        ),
        "required_auxiliary_measurements": [
            "lag_resolved_normalized_inner_product",
            "prefix_amplification_curve",
            "resultant_over_path",
            "sign_flip_null_interval",
        ],
        "hard_rules": {
            "old_case_verdicts_remain_frozen": True,
            "formation_t4_and_seup_verdicts_are_not_predictor_inputs": True,
            "incompatible_coordinate_spaces_are_never_concatenated": True,
            "semantic_orbit_must_preserve_exact_real_arithmetic_semantics": True,
            "not_applicable_is_not_a_negative": True,
            "missing_vector_or_counterfactual_arm_fails_closed": True,
            "no_threshold_is_tuned_to_recover_a_named_case": True,
        },
        "interpretation": {
            "source_persistent": "source/orbit or transported local component has excess coherence",
            "feedback_sustained": "local component is null-like while feedback and actual components are coherent",
            "diffusive_or_canceling": "actual component remains within the frozen matched-null regime",
            "unresolved": "measurement power, recurrence, or semantic-orbit validity is insufficient",
        },
        "predictor_freeze_gate": [
            "all development cases use the same 16-state horizon",
            "each reported statistic has an empirical matched null",
            "at least one semantics-preserving intervention changes the proposed predictor",
            "predictor implementation cannot read case IDs or historical verdicts",
        ],
        "prospective_confirmation_gate": [
            "predictions are committed before consequence values are revealed",
            "at least four exact new invocations",
            "at least two model families",
            "at least two eligible source mechanisms",
            "at least two risk predictions and two safe/control predictions when available",
            "no threshold changes after reveal",
        ],
        "success_levels": {
            "MECHANISM_TAXONOMY": "five-level localization and causal intervention only",
            "CANDIDATE_PROPERTY": "development predictor beats norm/shape baselines",
            "SUPPORTED_PROPERTY": "frozen predictor succeeds on prospective confirmation",
        },
    }

    roster = {
        "schema": "kernel-analyzer-persistence-property-roster-v1",
        "status": "DEVELOPMENT_FROZEN_PROSPECTIVE_SELECTION_RULE_FROZEN",
        "development_cases": [
            {
                "case_id": "liger_fused_ce",
                "role": "SOURCE_PERSISTENT_ANCHOR",
                "endpoint": "qwen3_liger_fused_linear_ce:dW",
                "model": "Qwen3-1.7B",
                "state_source": "results/property/seup_mainline/liger_seup.json",
                "orbit": "JOINT_TOKEN_PERMUTATION_OF_G_AND_H",
                "orbit_applicability": "ELIGIBLE_REDUCTION",
            },
            {
                "case_id": "phi4_seq64_lmhead_dx",
                "role": "TRANSPORT_PERSISTENT_ANCHOR",
                "endpoint": "phi4_seq64:lm_head.input_gradient.mm",
                "model": "Phi-4-mini-instruct",
                "state_source": "results/coverage/phi4_seq64_input_bank.json",
                "orbit": "JOINT_VOCAB_PERMUTATION_OF_G_AND_W",
                "orbit_applicability": "ELIGIBLE_REDUCTION",
            },
            {
                "case_id": "qwen128_vproj_mm",
                "role": "FORMATION_POSITIVE_PERSISTENCE_BOUNDARY",
                "endpoint": "qwen_seq128_forward_8_output",
                "model": "Qwen3-1.7B",
                "state_source": "results/coverage/qwen_seq128_input_bank.json",
                "orbit": "SEPARATE_MM_REDUCTION_AND_OUTPUT_ROUNDING_ORBITS",
                "orbit_applicability": "ELIGIBLE_WITH_SOURCE_SPLIT",
            },
            {
                "case_id": "qwen_saved_p_seq128",
                "role": "FEEDBACK_CANDIDATE",
                "endpoint": "qwen_seq128_layer27_attention_softmax_saved_P",
                "model": "Qwen3-1.7B",
                "state_source": "results/property/seup_mainline/qwen_softmax_seup.json",
                "orbit": None,
                "orbit_applicability": "NOT_APPLICABLE_CONTRACT_SOURCE",
            },
            {
                "case_id": "qwen3vl_silu_layer0",
                "role": "FEEDBACK_ANCHOR",
                "endpoint": "qwen3vl_layer0_silu_backward",
                "model": "Qwen3-VL-2B",
                "state_source": "results/coverage/cases/qwen3vl_layer0_silu_persistence_recurrence.json",
                "orbit": None,
                "orbit_applicability": "NOT_APPLICABLE_POINTWISE_SOURCE",
            },
            {
                "case_id": "qwen_bmm_seq64",
                "role": "VARIANCE_ONLY_CONTROL",
                "endpoint": "qwen_bmm_seq64_seup_negative_control",
                "model": "Qwen3-1.7B",
                "state_source": "results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz",
                "orbit": "BOUND_REDUCTION_PERMUTATION_IF_EXACT_OPERANDS_ARE_AVAILABLE",
                "orbit_applicability": "BLOCKED_PENDING_OPERAND_BINDING",
            },
        ],
        "retrospective_only_not_heldout": [
            "qwen64_vproj_mm",
            "mamba_seq64_input_proj",
            "qwen_layer23_attention_state",
            "deepseek_layer35_dv",
        ],
        "prospective_selection_rule": {
            "selection_uses_candidate_values": False,
            "required": [
                "exact F+B proof unit and executable endpoint binding",
                "candidate/repair/sham availability",
                "sixteen ordered states not used by development predictors",
                "declared parameter coordinate set",
                "semantic orbit explicitly valid or feedback probe explicitly applicable",
            ],
            "exclusions": [
                "any of the six development cases",
                "any invocation already used to define predictor fields",
                "unresolved many-to-one boundary without an exact repair",
            ],
            "slots": [
                "two reduction/source invocations from at least two model families",
                "one feedback-susceptibility invocation",
                "one source/control invocation expected to remain null-like",
            ],
        },
    }
    write("protocol.json", protocol)
    write("roster.json", roster)


if __name__ == "__main__":
    main()
