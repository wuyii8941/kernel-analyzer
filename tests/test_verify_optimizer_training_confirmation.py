import math

from scripts.verify_optimizer_training_confirmation import close, mean, t_interval


def test_mean_and_interval_use_independent_values():
    values = [0.01, 0.02, 0.03, 0.04]
    assert mean(values) == 0.025
    low, high = t_interval(values)
    assert low < 0.025 < high


def test_perplexity_conversion_is_monotone():
    interval = [0.012, 0.042]
    converted = [math.exp(value) for value in interval]
    assert 1 < converted[0] < converted[1]


def test_nested_comparison_handles_float_roundoff():
    assert close([0.1 + 0.2, 1.0], [0.3, 1.0])
    assert not close([0.3], [0.4])
