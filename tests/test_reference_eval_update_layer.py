"""Update-layer propagation (CPU): enclosures against high-precision evaluation."""

from fractions import Fraction

import numpy as np
import pytest

from kernel_analyzer.reference_eval.update_layer import adamw_delta, ema_state, sgd_delta, update_difference

mpmath = pytest.importorskip("mpmath")


def test_sgd_update_difference_is_minus_lr_times_the_residual():
    rng = np.random.default_rng(0)
    k = rng.standard_normal(200).astype(np.float32).astype(np.float64)
    kr = k + 1e-8 * rng.standard_normal(200)
    lo, hi = kr - 1e-15, kr + 1e-15
    u_lo, u_hi, r = update_difference(k, lo, hi, lambda a, b: sgd_delta(a, b, Fraction(1, 1024)))
    exact_lo = [-(Fraction(float(a)) - Fraction(float(c))) / 1024 for a, c in zip(k, lo)]
    exact_hi = [-(Fraction(float(a)) - Fraction(float(c))) / 1024 for a, c in zip(k, hi)]
    for ul, uh, el, eh in zip(u_lo, u_hi, exact_hi, exact_lo):
        assert Fraction(float(ul)) <= el and eh <= Fraction(float(uh))
    assert np.allclose(r, -kr / 1024)


def _mp(x):
    if isinstance(x, Fraction):
        return mpmath.mpf(x.numerator) / x.denominator
    return mpmath.mpf(x)


def _adamw_exact(g, m, v, t, lr, b1, b2, eps):
    mp = _mp
    b1, b2, eps, lr = mp(b1), mp(b2), mp(eps), mp(lr)
    m1 = b1 * mp(m) + (1 - b1) * mp(g)
    v1 = b2 * mp(v) + (1 - b2) * mp(g) ** 2
    return -lr * (m1 / (1 - b1 ** t)) / (mpmath.sqrt(v1 / (1 - b2 ** t)) + eps)


@pytest.mark.parametrize("zero_state", [True, False])
def test_adamw_step_encloses_the_exact_real_step(zero_state):
    mpmath.mp.dps = 60
    rng = np.random.default_rng(1)
    hist = [rng.standard_normal(64) for _ in range(16)]
    m, v = (np.zeros(64), np.zeros(64)) if zero_state else ema_state(hist, "0.9", "0.999")
    t = 1 if zero_state else 17
    g = rng.standard_normal(64)
    lo, hi = adamw_delta(g, g, m, v, t, Fraction(1, 1024), "0.9", "0.999", Fraction(1, 2 ** 27))
    for i in range(64):
        x = _adamw_exact(g[i], m[i], v[i], t, Fraction(1, 1024), "0.9", "0.999", mpmath.mpf(2) ** -27)
        assert mpmath.mpf(lo[i]) <= x <= mpmath.mpf(hi[i])
        assert hi[i] - lo[i] <= 1e-14 * abs(x) + 1e-300  # a few ulps over a dozen directed operations
