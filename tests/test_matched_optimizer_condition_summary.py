import copy

import pytest

from scripts.summarize_matched_optimizer_conditions import summarize


def profile(effect: float, *, moments: str = "ZERO") -> dict:
    rows = [{
        "effect_energy": effect,
        "repair_energy": 1.0,
        "effect_repair_inner_product": -effect / 2,
        "nonzero_effect_coordinates": int(effect > 0),
    } for _ in range(32)]
    return {
        "status": "COMPLETE",
        "case_id": "case",
        "state_ids": [str(index) for index in range(32)],
        "optimizer": {"moments": moments},
        "original_coordinate_statistics": {
            "LOCAL": copy.deepcopy(rows),
            "PARAMETER_GRADIENT": copy.deepcopy(rows),
            "PARAMETER_WRITE": copy.deepcopy(rows),
        },
    }


def test_summary_uses_original_coordinates_and_checks_reset_gradient() -> None:
    cold = profile(0.04)
    warm = profile(0.0004, moments="WARM")
    reset = profile(0.04, moments="RESET")
    reset["original_coordinate_statistics"]["PARAMETER_GRADIENT"] = copy.deepcopy(
        warm["original_coordinate_statistics"]["PARAMETER_GRADIENT"]
    )
    result = summarize(cold, warm, reset)
    assert result["conditions"]["COLD_ZERO_MOMENTS"]["PARAMETER_WRITE"][
        "confirmation_total_rms"
    ] == pytest.approx(0.2)
    assert result["comparisons"]["warm_to_cold_parameter_write_rms_ratio"] == (
        pytest.approx(0.1)
    )
    assert result["comparisons"]["reset_to_warm_parameter_write_rms_ratio"] == (
        pytest.approx(10.0)
    )
    assert result["comparisons"]["warm_and_reset_gradient_statistics_exactly_equal"]


def test_summary_rejects_different_state_order() -> None:
    cold = profile(0.04)
    warm = profile(0.01)
    reset = profile(0.04)
    warm["state_ids"] = list(reversed(warm["state_ids"]))
    with pytest.raises(ValueError, match="same 32 ordered states"):
        summarize(cold, warm, reset)
