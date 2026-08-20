import pytest

from kernel_analyzer.bias_formation_v22 import (
    BiasV22Status,
    aggregate_conditional_debias,
    certify_trajectory_separation,
    summarize_conditional_gram,
    summarize_conditional_vectors,
)


def test_mixed_conditions_are_not_pooled_into_a_global_null():
    result = summarize_conditional_vectors({
        "positive": [[1.0, 1.0]] * 4,
        "negative": [[-1.0, -1.0]] * 4,
    })
    assert result["status"] == BiasV22Status.CONDITIONAL_BIAS.value
    assert result["pooled_cross_condition_direction_not_tested"] is True


def test_one_observation_condition_is_unresolved_not_centered():
    result = summarize_conditional_vectors({"one_state": [[1.0, 0.0]]})
    assert result["status"] == BiasV22Status.CONDITIONAL_UNRESOLVED.value
    assert result["conditions"]["one_state"]["reason"] == "INSUFFICIENT_REPLICATES"


def test_conditional_gram_keeps_repair_residual_and_removed_effect_distinct():
    gram = [[2.0] * 4 for _ in range(4)]
    repair = summarize_conditional_gram(
        gram,
        condition_id="fixed-state",
        coordinate_count=2,
        replicate_ids=[f"r{i}" for i in range(4)],
        vector_digests=[f"d{i}" for i in range(4)],
        estimand="REPAIR_RESIDUAL",
        reference="EXACT_DECLARED_LOCAL_SOURCE_ZERO_SAME_OPERANDS",
    )
    effect = summarize_conditional_gram(
        gram,
        condition_id="fixed-state",
        coordinate_count=2,
        replicate_ids=[f"r{i}" for i in range(4)],
        vector_digests=[f"d{i}" for i in range(4)],
        estimand="CANDIDATE_MINUS_REPAIR_ENSEMBLE",
        reference="STOCHASTIC_SOURCE_DEBIASED_ENSEMBLE",
    )
    assert repair["identifies_repair_residual_bias"] is True
    assert effect["identifies_candidate_effect_removed"] is True
    assert repair["certifies_downstream_repair_is_unbiased"] is False
    assert effect["certifies_downstream_repair_is_unbiased"] is False


def test_conditional_gram_fails_closed_with_too_few_repeats():
    result = summarize_conditional_gram(
        [[1.0]], condition_id="fixed-state", coordinate_count=1,
        replicate_ids=["r0"], vector_digests=["d0"],
        estimand="REPAIR_RESIDUAL",
        reference="EXACT_DECLARED_LOCAL_SOURCE_ZERO_SAME_OPERANDS",
    )
    assert result["status"] == BiasV22Status.CONDITIONAL_UNRESOLVED.value


def test_conditional_debias_aggregate_does_not_require_global_direction():
    conditions = {
        "state-a": {
            "repair_local_residual": {"status": "CONDITIONAL_CENTERED"},
            "candidate_gradient_effect_removed": {"status": "CONDITIONAL_BIAS"},
        },
        "state-b": {
            "repair_local_residual": {"status": "CONDITIONAL_CENTERED"},
            "candidate_gradient_effect_removed": {"status": "CONDITIONAL_BIAS"},
        },
    }
    result = aggregate_conditional_debias(conditions)
    assert result["status"] == (
        "LOCAL_SOURCE_CONDITIONALLY_DEBIASED_WITH_SYSTEMATIC_CANDIDATE_F_B_EFFECT"
    )
    assert result["global_direction_required"] is False
    assert result["absolute_downstream_repair_bias"].startswith("NOT_IDENTIFIABLE")


def test_conditional_debias_aggregate_fails_closed_on_one_unresolved_condition():
    conditions = {
        "state-a": {
            "repair_local_residual": {"status": "CONDITIONAL_CENTERED"},
            "candidate_gradient_effect_removed": {"status": "CONDITIONAL_BIAS"},
        },
        "state-b": {
            "repair_local_residual": {"status": "CONDITIONAL_UNRESOLVED"},
            "candidate_gradient_effect_removed": {"status": "CONDITIONAL_BIAS"},
        },
    }
    assert aggregate_conditional_debias(conditions)["status"] == (
        "CONDITIONAL_DEBIAS_UNRESOLVED"
    )


def test_trajectory_does_not_require_fixed_global_direction():
    rows = [{"drift_norm": value} for value in [1.0, 0.7, 1.4, 1.1, 1.8, 1.5, 2.0, 2.2]]
    result = certify_trajectory_separation(
        rows,
        gates={
            "repair_effect_present_every_step": True,
            "matched_sham_exact": True,
            "only_declared_parameter_updated": True,
            "directional_live_weight_accumulation": False,
        },
    )
    assert result["status"] == BiasV22Status.TRAJECTORY_BIAS.value
    assert result["fixed_global_carrier_required"] is False


def test_missing_sham_fails_closed():
    rows = [{"drift_norm": 1.0 + 0.1 * i} for i in range(8)]
    result = certify_trajectory_separation(
        rows,
        gates={
            "repair_effect_present_every_step": True,
            "matched_sham_exact": False,
            "only_declared_parameter_updated": True,
        },
    )
    assert result["status"] == BiasV22Status.TRAJECTORY_UNRESOLVED.value
    assert "MISSING_CAUSAL_OR_SHAM_GATE" in result["reason"]


def test_closed_full_step_scope_can_replace_single_parameter_scope():
    rows = [{"drift_norm": 1.0 + 0.1 * i} for i in range(8)]
    result = certify_trajectory_separation(
        rows,
        gates={
            "repair_effect_present_every_step": True,
            "matched_sham_exact": True,
            "full_step_two_arm_scope_closed": True,
        },
    )
    assert result["status"] == BiasV22Status.TRAJECTORY_BIAS.value


def test_scope_missing_fails_closed_even_when_drift_grows():
    rows = [{"drift_norm": 1.0 + 0.1 * i} for i in range(8)]
    result = certify_trajectory_separation(
        rows,
        gates={
            "repair_effect_present_every_step": True,
            "matched_sham_exact": True,
        },
    )
    assert result["status"] == BiasV22Status.TRAJECTORY_UNRESOLVED.value
    assert "MISSING_CAUSAL_OR_SHAM_GATE" in result["reason"]
