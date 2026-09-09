import pytest

from scripts.run_optimizer_training_pilot import summarize


def protocol():
    return {"purpose": "FEASIBILITY_AND_COST_ONLY"}


def row(condition, final):
    return {"condition": condition, "status": "COMPLETE", "training_loss": [3.0, 2.0],
            "initial_evaluation_loss": [3.0, 3.0], "final_evaluation_loss": [final, final],
            "steps_per_second": 2.0, "peak_allocated_bytes": 10,
            "final_parameter_sha256": condition}


def test_pilot_never_claims_a_scientific_training_outcome():
    raw = {"records": [row("FP32_ADAMW", 2.9), row("ADAMW8BIT_BLOCK256", 2.0),
                       row("ADAMW8BIT_BLOCK64", 2.5)]}
    result = summarize(protocol(), raw)
    assert result["all_conditions_finite_and_complete"]
    assert result["scientific_training_outcome"] == "NOT_ASSESSED_BY_PILOT"
    assert result["confirmation_design_status"].startswith("MAY_FREEZE")


def test_incomplete_pilot_fails_closed():
    with pytest.raises(ValueError, match="incomplete"):
        summarize(protocol(), {"records": [row("FP32_ADAMW", 2.9)]})
