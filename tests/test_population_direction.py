import pytest

from kernel_analyzer.population_direction import population_positive_direction_prevalence


def test_all_positive_units_confirm_direction_prevalence():
    result = population_positive_direction_prevalence([1.0] * 16)
    assert result["decision"] == "DIRECTION_PREVALENCE_CONFIRMED"
    assert result["one_sided_probability_bounds"][0] > 0.5


def test_balanced_signs_are_inconclusive_and_zero_is_not_positive():
    result = population_positive_direction_prevalence(
        [-1.0] * 7 + [0.0, 0.0] + [1.0] * 7
    )
    assert result["decision"] == "INCONCLUSIVE"
    assert result["positive_count"] == 7
    assert result["zero_count"] == 2


@pytest.mark.parametrize("values", [[], [1.0, float("nan")]])
def test_invalid_samples_are_rejected(values):
    with pytest.raises(ValueError):
        population_positive_direction_prevalence(values)
