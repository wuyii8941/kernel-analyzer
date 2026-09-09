import pytest

from kernel_analyzer.training_candidate_selection import select_candidates


def candidate(case_id="new-triton", *, rms=.02, novelty="NEW_OPERATOR_FAMILY",
              kind="TRITON"):
    return {
        "case_id": case_id,
        "operator_family_id": "GROUPED_CAUSAL_SOFTMAX_FORWARD",
        "implementation_kind": kind,
        "family_novelty": novelty,
        "measurement": {
            "status": "VALID", "contrast_id": "NATURAL_IMPLEMENTATION",
            "primary_endpoint": "PARAMETER_WRITE",
            "reference_scope": {
                "same_local_operands": True,
                "includes_possible_upstream_differences": False,
                "source_or_call_identity_verified": True,
            },
        },
        "warm_state": {
            "status": "VALID", "natural_optimizer_state": True,
            "actual_parameter_write": True, "fixed_suite_total_rms": rms,
        },
        "mechanism": {
            "prediction_status": "FROZEN_BEFORE_CONFIRMATION",
            "implementation_location_verified": True,
            "legal_modification_ids": ["FP32_ACCUMULATION"],
        },
    }


def manifest(*cases, fixed=True, floor=.01):
    return {"schema": "training-validation-candidate-manifest-v1",
            "policy": {"fixed_before_candidate_measurement": fixed,
                       "minimum_warm_update_rms": floor}, "cases": list(cases)}


def test_new_triton_family_is_first_review_queue_without_auto_launch():
    old = candidate("old", novelty="ALREADY_DEEPLY_STUDIED")
    ordinary = candidate("ordinary", kind="ORDINARY")
    new = candidate()
    result = select_candidates(manifest(old, ordinary, new))
    assert [row["queue"] for row in result["rows"]] == [
        "NEW_TRITON_OPERATOR_FAMILY", "NEW_NONTRITON_OPERATOR_FAMILY",
        "EXISTING_FAMILY_MECHANISM_FOLLOWUP",
    ]
    assert result["eligible_cases"] == 3
    assert not result["automatic_training_launch"]
    assert not result["selection_uses_effect_size_as_only_criterion"]


@pytest.mark.parametrize("change,reason", [
    (("warm_state", "fixed_suite_total_rms", .001), "WARM_EFFECT_BELOW_PREDECLARED_TRAINING_RELEVANCE_FLOOR"),
    (("warm_state", "natural_optimizer_state", False), "NATURAL_WARM_STATE_EVIDENCE_MISSING"),
    (("measurement", "primary_endpoint", "ADAMW_UPDATE"), "ACTUAL_PARAMETER_WRITE_NOT_PRIMARY"),
    (("measurement", "reference_scope", {"same_local_operands": False,
      "includes_possible_upstream_differences": True,
      "source_or_call_identity_verified": False}), "REFERENCE_DOES_NOT_USE_SAME_LOCAL_OPERANDS"),
    (("mechanism", "prediction_status", "POST_HOC"), "MECHANISM_PREDICTION_NOT_FROZEN"),
])
def test_each_non_magnitude_gate_fails_closed(change, reason):
    case = candidate()
    parent, key, value = change
    case[parent][key] = value
    result = select_candidates(manifest(case))["rows"][0]
    assert result["status"] == "NOT_ELIGIBLE"
    assert reason in result["reasons"]


def test_policy_cannot_be_added_after_results_or_use_zero_floor():
    with pytest.raises(ValueError, match="fixed before"):
        select_candidates(manifest(candidate(), fixed=False))
    with pytest.raises(ValueError, match="positive finite"):
        select_candidates(manifest(candidate(), floor=0))


def test_duplicate_cases_rejected():
    with pytest.raises(ValueError, match="Unique"):
        select_candidates(manifest(candidate(), candidate()))
