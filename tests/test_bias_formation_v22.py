import pytest

from kernel_analyzer.bias_formation_v22 import (
    BiasV22Status,
    certify_trajectory_separation,
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
