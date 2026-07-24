from __future__ import annotations

import math

from scripts.phase2_logsoftmax_bound import gamma


def test_gamma_matches_standard_roundoff_factor() -> None:
    u = 2.0**-24
    k = 151_935
    expected = (k * u) / (1.0 - k * u)
    assert math.isclose(gamma(k, u), expected, rel_tol=0.0, abs_tol=0.0)


def test_gamma_is_infinite_outside_valid_regime() -> None:
    assert math.isinf(gamma(2, 0.5))
