"""check._within_input_mean_total: the repeated-launch mean of K - f per unit.

Found by path A2 (FlagGems embedding_backward with atomic adds, 2026-10-21): repeated launches differed, the call
reached this function and it raised AttributeError (intervals has no sum_bounds). The mean must enclose the exact mean
of K_j - f over the launches."""
from fractions import Fraction

import numpy as np

from kernel_analyzer import check


def test_mean_total_encloses_the_exact_mean():
    f = np.array([1.0, 2.0, 3.0])
    rows = [{"k": np.array([1.5, 2.0, 2.0]), "k_reps": [np.array([1.25, 2.5, 2.0])],
             "f_lo": f, "f_hi": f, "ok_total": np.array([True, True, False])}]
    lo, hi = check._within_input_mean_total(rows)
    exact = [(Fraction(1.5) - 1 + Fraction(1.25) - 1) / 2, (Fraction(0) + Fraction(1, 2)) / 2]
    for i, e in enumerate(exact):
        assert Fraction(float(lo[0, i])) <= e <= Fraction(float(hi[0, i]))
    assert lo[0, 2] <= 0 <= hi[0, 2]
