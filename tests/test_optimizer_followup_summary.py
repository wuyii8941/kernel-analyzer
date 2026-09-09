import pytest

from scripts.summarize_optimizer_followup import summarize


def baseline(case, local, gradient, update):
    return {"rows": [
        {"case_id": case, "stage": stage, "status": "VALID", "relative_rms": value}
        for stage, value in zip(
            ("LOCAL", "PARAMETER_GRADIENT", "PARAMETER_WRITE"),
            (local, gradient, update),
        )
    ]}


def test_directional_prediction_keeps_missing_threshold_explicit():
    plan = {"schema": "family-optimizer-state-followup-v1", "cases": [{"case_id": "x"}],
            "prediction_fixed_before_followup": "update falls more"}
    result = summarize(
        plan, baseline("x", 1, 1, 10), baseline("x", 2, .8, 1), baseline("x", 3, .9, .5)
    )
    assert result["directional_prediction_result"].startswith("DIRECTIONALLY_CONSISTENT")
    assert result["formal_prediction_test"].startswith("NOT_ASSESSED")
    assert result["training_outcome_established"] is False


def test_missing_stage_is_rejected():
    plan = {"schema": "family-optimizer-state-followup-v1", "cases": [{"case_id": "x"}],
            "prediction_fixed_before_followup": "prediction"}
    broken = baseline("x", 1, 1, 1)
    broken["rows"].pop()
    with pytest.raises(ValueError, match="all three stages"):
        summarize(plan, broken, baseline("x", 1, 1, 1), baseline("x", 1, 1, 1))
