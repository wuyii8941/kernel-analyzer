from kernel_analyzer.direction_interpretation import positive_direction_frequency


def test_zero_values_do_not_prove_opposite_direction():
    result = positive_direction_frequency([0.0] * 16)
    assert result['decision'] == 'POSITIVE_DIRECTION_FREQUENCY_BELOW_NULL'
    assert not result['negative_direction_prevalence_assessed']


def test_positive_results_unchanged():
    result = positive_direction_frequency([1.0] * 16)
    assert result['decision'] == 'DIRECTION_PREVALENCE_CONFIRMED'
    assert result['one_sided_probability_bounds'][0] > .5
