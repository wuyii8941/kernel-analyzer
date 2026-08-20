import copy

import pytest

from scripts.build_systematic_bias_audit import cases
from kernel_analyzer.systematic_bias_audit import validate_audit, validate_case


def test_systematic_audit_has_exactly_eight_unique_cases():
    rows = cases()
    validate_audit(rows)
    assert len(rows) == 8
    assert len({row["case_id"] for row in rows}) == 8
    assert "qwen_layer23_key_materialization" not in {row["case_id"] for row in rows}


def test_eight_cases_are_not_relabeled_as_eight_persistent_biases():
    rows = cases()
    assert all(
        row["trajectory"]["separation_status"] == "TRAJECTORY_SEPARATION"
        for row in rows
    )
    persistent = {
        row["case_id"] for row in rows
        if row["trajectory"]["directional_persistence"] == "CONFIRMED"
    }
    assert persistent == {
        "liger_fused_ce",
        "phi4_seq64_lmhead_dx",
        "qwen64_vproj_mm",
        "qwen_saved_p_seq128",
        "qwen3vl_silu_layer0",
        "mamba_seq64_input_proj",
        "qwen_layer23_attention_state",
    }
    assert {
        row["case_id"] for row in rows
        if row["trajectory"]["directional_persistence"] == "NOT_CONFIRMED"
    } == {"qwen128_vproj_mm"}


def test_same_contrast_full_chain_is_stricter_than_either_layer():
    rows = cases()
    full_chain = {
        row["case_id"] for row in rows
        if row["trajectory"]["same_contrast_full_chain"] is True
    }
    assert full_chain == {
        "liger_fused_ce",
        "phi4_seq64_lmhead_dx",
        "qwen_saved_p_seq128",
        "qwen_layer23_attention_state",
    }
    for row in rows:
        if row["trajectory"]["same_contrast_full_chain"]:
            assert row["trajectory"]["directional_persistence"] == "CONFIRMED"
            assert row["trajectory"]["contrast_alignment"].startswith("ALIGNED")


def test_separation_cannot_be_named_trajectory_bias():
    row = copy.deepcopy(cases()[0])
    row["trajectory"]["status"] = "TRAJECTORY_BIAS"
    row["trajectory"]["separation_status"] = "TRAJECTORY_BIAS"
    with pytest.raises(ValueError, match="invalid trajectory separation"):
        validate_case(row)


def test_global_centered_saved_p_is_not_variance_only():
    row = next(row for row in cases() if row["case_id"] == "qwen_saved_p_seq128")
    assert set(row["formation"]["global"].values()) == {"CENTERED"}
    assert row["mechanism"]["verdict"] != "VARIANCE_ONLY_UNDER_DECLARED_CONDITION"
    assert set(row["formation"]["conditional"].values()) == {"NOT_MEASURED"}


def test_trajectory_cannot_supply_formation_label():
    row = copy.deepcopy(cases()[0])
    row["formation"]["label_source"] = "TRAJECTORY"
    with pytest.raises(ValueError, match="trajectory cannot label formation"):
        validate_case(row)


def test_supported_mechanism_requires_intervention_and_sham():
    row = copy.deepcopy(cases()[0])
    row["mechanism"]["intervention"]["matched_sham_exact"] = False
    with pytest.raises(ValueError, match="lacks intervention/sham"):
        validate_case(row)


def test_qwen128_uses_aligned_rounding_trajectory_but_does_not_persist():
    row = next(row for row in cases() if row["case_id"] == "qwen128_vproj_mm")
    assert row["mechanism"]["trajectory_repairs_declared_local_source"] is True
    assert row["mechanism"]["verdict"] == "SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM"
    assert row["mechanism"]["intervention"]["full_observed_source_repaired_in_expectation"]
    assert row["bias_map"]["status"] == "MATCHED_CONDITIONAL_SOURCE_SUPPORT"
    assert row["formation"]["conditional"] == {
        "local": "BIASED", "gradient": "BIASED", "update": "BIASED",
    }
    assert row["formation"]["conditional_details"]["repair_local_residual"] == "CENTERED"
    assert row["trajectory"]["contrast_alignment"] == "ALIGNED"
    assert row["trajectory"]["directional_persistence"] == "NOT_CONFIRMED"
    assert row["trajectory"]["ordered_recurrence"]["verdict"] == "DIFFUSIVE_OR_CANCELING_SEPARATION"


def test_silu_persistence_is_feedback_sustained_not_local_persistent():
    row = next(row for row in cases() if row["case_id"] == "qwen3vl_silu_layer0")
    recurrence = row["trajectory"]["ordered_recurrence"]
    assert row["trajectory"]["directional_persistence"] == "CONFIRMED"
    assert row["trajectory"]["persistence_regime"] == "FEEDBACK_SUSTAINED"
    assert recurrence["verdict"] == "FEEDBACK_SUSTAINED_SEPARATION"
    assert recurrence["local_coherence_amplification"] < 2.0
    assert recurrence["feedback_coherence_amplification"] >= 2.0


def test_qwen64_joint_source_has_independent_fixed_state_confirmation():
    row = next(row for row in cases() if row["case_id"] == "qwen64_vproj_mm")
    assert row["mechanism"]["verdict"] == "SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM"
    assert row["formation"]["conditional"] == {
        "local": "BIASED", "gradient": "BIASED", "update": "BIASED",
    }
    details = row["formation"]["conditional_details"]
    assert details["conditions"] == 16
    assert details["repair_local_residual"] == "CENTERED"
    assert row["bias_map"]["status"] == "MATCHED_CONDITIONAL_SOURCE_SUPPORT"


def test_layer23_antithetic_followup_fails_natural_fidelity_closed():
    row = next(
        row for row in cases() if row["case_id"] == "qwen_layer23_attention_state"
    )
    assert row["bias_map"]["antithetic_status"] == "ANTITHETIC_INTERVENTION_UNRESOLVED"
    assert row["bias_map"]["validity_gates"]["sixteen_fixed_conditions"] is True
    assert row["bias_map"]["validity_gates"]["all_local_antithetic_exact"] is True
    assert row["bias_map"]["validity_gates"]["natural_source_fidelity_every_condition"] is False


def test_mamba_preserves_mixed_conditional_result_without_promotion():
    row = next(row for row in cases() if row["case_id"] == "mamba_seq64_input_proj")
    assert row["mechanism"]["verdict"] == "PARTIAL_SOURCE_MECHANISM"
    assert row["formation"]["conditional"] == {
        "local": "BIASED", "gradient": "UNRESOLVED", "update": "UNRESOLVED",
    }
    roles = row["formation"]["conditional_details"]["all_roles"]
    assert roles["repair_local_residual"]["status_counts"] == {
        "CONDITIONAL_CENTERED": 16,
    }
    assert roles["candidate_gradient_effect_removed"]["status_counts"] == {
        "CONDITIONAL_BIAS": 13,
        "CONDITIONAL_UNRESOLVED": 3,
    }
    assert roles["candidate_adamw_zero_update_effect_removed"]["status_counts"] == {
        "CONDITIONAL_BIAS": 16,
    }


def test_variance_only_needs_all_three_conditional_nulls():
    row = copy.deepcopy(cases()[0])
    row["mechanism"]["verdict"] = "VARIANCE_ONLY_UNDER_DECLARED_CONDITION"
    with pytest.raises(ValueError, match="three conditional nulls"):
        validate_case(row)
