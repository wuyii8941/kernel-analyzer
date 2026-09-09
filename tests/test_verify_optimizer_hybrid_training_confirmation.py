import math

import pytest

from scripts.verify_optimizer_hybrid_training_confirmation import close, mean, t_interval


def test_mean_and_interval_use_independent_values():
    values = [0.01, 0.02, 0.03, 0.04]
    assert mean(values) == 0.025
    low, high = t_interval(values)
    assert low < 0.025 < high


def test_interval_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        t_interval([0.01, math.nan])


def test_nested_comparison_handles_float_roundoff():
    assert close({"x": [0.1 + 0.2]}, {"x": [0.3]})
    assert not close({"x": [0.3]}, {"x": [0.4]})


def test_positive_interval_requires_stable_paired_effect():
    low, high = t_interval([0.02, 0.021, 0.019, 0.022, 0.018, 0.02, 0.021, 0.019])
    assert 0 < low < high
