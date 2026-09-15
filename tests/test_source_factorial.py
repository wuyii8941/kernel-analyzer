import numpy as np
import pytest

from kernel_analyzer.source_factorial import two_factor_source_decomposition


def test_two_factor_decomposition_closes_and_preserves_interaction():
    # f(a,b) = 2a + 3b + 5ab, with candidate=(1,1), joint repair=(0,0).
    candidate = np.array([10.0, 20.0])
    repair_a = np.array([3.0, 6.0])       # f(0,1)
    repair_b = np.array([2.0, 4.0])       # f(1,0)
    repair_both = np.zeros(2)
    result = two_factor_source_decomposition(
        candidate, repair_a, repair_b, repair_both,
        factor_a="A", factor_b="B",
    )
    assert result["closure_max_abs"] == 0.0
    assert result["factorial_interaction"]["energy"] > 0
    assert result["factor_a_contribution"]["projection_on_total"] == pytest.approx(.45)
    assert result["factor_b_contribution"]["projection_on_total"] == pytest.approx(.55)


def test_two_factor_decomposition_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match="identical coordinates"):
        two_factor_source_decomposition([1], [1, 2], [1], [0],
                                        factor_a="A", factor_b="B")
