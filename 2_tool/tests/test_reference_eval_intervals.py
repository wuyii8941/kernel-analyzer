"""Directed-rounding float64 intervals checked against exact rationals."""

import math

import numpy as np
import pytest

gmpy2 = pytest.importorskip("gmpy2")
from gmpy2 import mpq  # noqa: E402

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402


def _samples(rng, n):
    scales = 10.0 ** rng.integers(-30, 30, size=n)
    x = rng.standard_normal(n) * scales
    x[: n // 10] = rng.integers(-2 ** 24, 2 ** 24, size=n // 10).astype(np.float64)
    return x.astype(np.float32).astype(np.float64)


@pytest.mark.parametrize("op", ["add", "mul", "div"])
def test_directed_bounds_enclose_exact_and_are_tight(op):
    rng = np.random.default_rng(1)
    a, b = _samples(rng, 3000), _samples(rng, 3000)
    if op == "div":
        b = np.where(b == 0, 1.0, b)
    fn = {"add": iv.add_bounds, "mul": iv.mul_bounds, "div": iv.div_bounds}[op]
    lo, hi = fn(a, b)
    for x, y, l, h in zip(a, b, lo, hi):
        exact = {"add": mpq(x) + mpq(y), "mul": mpq(x) * mpq(y), "div": mpq(x) / mpq(y)}[op]
        assert mpq(l) <= exact <= mpq(h)
        if mpq(l) == exact:
            assert l == h  # exact results keep width zero
        else:
            assert math.nextafter(l, math.inf) == h  # one ulp otherwise


def test_sqrt_bounds_enclose_exact():
    rng = np.random.default_rng(2)
    a = np.abs(_samples(rng, 2000))
    lo, hi = iv.sqrt_bounds(a)
    for x, l, h in zip(a, lo, hi):
        assert mpq(l) ** 2 <= mpq(x) <= mpq(h) ** 2
    lo, hi = iv.sqrt_bounds(np.array([4.0, 2.0]))
    assert lo[0] == hi[0] == 2.0 and lo[1] < hi[1]


@pytest.mark.parametrize("name", ["exp", "log", "tanh", "log1p", "rsqrt", "sin", "cos", "erf"])
def test_elementary_bounds_are_rigorous(name):
    rng = np.random.default_rng(3)
    x = rng.uniform(0.01, 8.0, 300)
    lo, hi, ok = iv.elementary_bounds(name, x, x)
    assert ok.all()
    fn = {"exp": gmpy2.exp, "log": gmpy2.log, "tanh": gmpy2.tanh, "log1p": gmpy2.log1p,
          "rsqrt": gmpy2.rec_sqrt, "sin": gmpy2.sin, "cos": gmpy2.cos, "erf": gmpy2.erf}[name]
    with gmpy2.context(precision=300):
        for v, l, h in zip(x, lo, hi):
            truth = mpq(fn(gmpy2.mpfr(v)))
            assert mpq(l) <= truth <= mpq(h)
            assert h - l <= 4 * math.ulp(abs(float(truth))) + 1e-300


def test_sum_and_dot_enclose_exact():
    rng = np.random.default_rng(4)
    x = _samples(rng, 512).reshape(4, 128)
    lo, hi = iv.isum(x, x, axis=1)
    for row, l, h in zip(x, lo, hi):
        exact = sum(mpq(v) for v in row)
        assert mpq(l) <= exact <= mpq(h)
    a = rng.standard_normal((8, 16)).astype(np.float32).astype(np.float64)
    b = rng.standard_normal((16, 4)).astype(np.float32).astype(np.float64)
    lo, hi = iv.idot(a, a, b, b)
    for i in range(8):
        for j in range(4):
            exact = sum(mpq(a[i, k]) * mpq(b[k, j]) for k in range(16))
            assert mpq(lo[i, j]) <= exact <= mpq(hi[i, j])


def test_round_nearest_even_formats():
    vals = np.array([257.0, 1 + 17 / 4096, 2.0 ** -149, 2.0 ** -151, 3.5e38, 65520.0])
    bf16, _ = iv.round_nearest_even(vals[:2], "bf16")
    assert bf16[0] == 256.0 and bf16[1] - vals[1] == 15 / 4096
    f16, of = iv.round_nearest_even(vals[1:2], "f16")
    assert f16[0] - vals[1] == -1 / 4096
    f32, of32 = iv.round_nearest_even(vals[2:5], "f32")
    assert f32[0] == 2.0 ** -149 and f32[1] == 0.0 and of32[2]
    _, of16 = iv.round_nearest_even(vals[5:], "f16")
    assert of16[0]  # 65520 rounds to infinity in fp16
    rng = np.random.default_rng(5)
    x = rng.standard_normal(10000) * 10.0 ** rng.integers(-40, 38, 10000)
    ours, _ = iv.round_nearest_even(x, "f32")
    with np.errstate(over="ignore"):
        assert np.array_equal(ours, x.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("name", ["exp", "log", "tanh", "sqrt", "rsqrt", "log1p", "erf", "atan"])
def test_single_call_point_enclosure_equals_two_directed_calls(name):
    fn, direction = iv._MONOTONE[name][0], iv._MONOTONE[name][1]
    rng = np.random.default_rng(6)
    xs = np.concatenate([rng.uniform(0.01, 8.0, 300), [1.0, 4.0, 0.25]])
    for x in xs:
        x = float(x)
        lo1, hi1 = iv._mpfr_point(fn, x)
        if direction > 0:
            lo2, hi2 = iv._mpfr_scalar(fn, x, iv._CTX_DOWN, -1), iv._mpfr_scalar(fn, x, iv._CTX_UP, +1)
        else:
            lo2, hi2 = iv._mpfr_scalar(fn, x, iv._CTX_DOWN, -1), iv._mpfr_scalar(fn, x, iv._CTX_UP, +1)
        if lo1 == hi1:  # exact result: the single call keeps width 0 (the two-call path widens an exact 0)
            assert lo2 <= lo1 <= hi2, (name, x)
        else:
            assert (lo1, hi1) == (lo2, hi2), (name, x)


@pytest.mark.parametrize("mode", ["rtz", "rd", "ru", "rtne"])
def test_round_directed_matches_exact_rounding(mode):
    from fractions import Fraction

    rng = np.random.default_rng(7)
    x = rng.standard_normal(2000) * 10.0 ** rng.integers(-40, 39, 2000)
    vals, pinf, ninf = iv.round_directed(x, "f32", mode)
    for xi, vi, pi, ni in zip(x, vals, pinf, ninf):
        if pi or ni:
            assert abs(xi) > 3.4e38
            continue
        lo = float(np.float32(xi))  # RN to f32, then step to the directed neighbour
        cands = [lo, float(np.nextafter(np.float32(lo), np.float32(-np.inf))),
                 float(np.nextafter(np.float32(lo), np.float32(np.inf)))]
        cands = [c for c in cands if np.isfinite(c)]
        fx = Fraction(xi)
        if mode == "rd":
            want = max(c for c in cands if Fraction(c) <= fx) if any(Fraction(c) <= fx for c in cands) else None
        elif mode == "ru":
            want = min(c for c in cands if Fraction(c) >= fx) if any(Fraction(c) >= fx for c in cands) else None
        elif mode == "rtz":
            want = (max(c for c in cands if Fraction(c) <= fx) if xi > 0 else min(c for c in cands if Fraction(c) >= fx))
        else:
            want = lo
        if want is not None:
            assert vi == want, (mode, xi, vi, want)


def test_project_bounds_encloses_the_exact_projection_with_cancellation():
    from fractions import Fraction

    rng = np.random.default_rng(11)
    for trial in range(40):
        d = int(rng.integers(2, 300))
        mid = rng.standard_normal(d) * 10.0 ** rng.integers(-8, 8, d)
        if trial % 2:  # heavy cancellation: the projection is tiny compared with its terms
            mid[d // 2:] = -mid[: d - d // 2][: d - d // 2]
        rad = np.abs(mid) * 10.0 ** rng.integers(-16, -3, d) * rng.random(d)
        lo, hi = mid - rad, mid + rad
        w = rng.standard_normal(d)
        w /= np.linalg.norm(w)
        pl, ph = iv.project_bounds(lo[None], hi[None], w)
        exact_lo = sum(min(Fraction(float(a)) * Fraction(float(c)), Fraction(float(b)) * Fraction(float(c)))
                       for a, b, c in zip(lo, hi, w))
        exact_hi = sum(max(Fraction(float(a)) * Fraction(float(c)), Fraction(float(b)) * Fraction(float(c)))
                       for a, b, c in zip(lo, hi, w))
        assert Fraction(float(pl[0])) <= exact_lo and exact_hi <= Fraction(float(ph[0])), trial


def test_fsum_bounds_is_outward_even_when_ordinary_sums_are_not():
    from fractions import Fraction

    # left-to-right summation loses the small terms; the exact sum is 2**-60 * 1000 above 1
    x = np.concatenate([[1.0], np.full(1000, 2.0 ** -60)])
    lo, hi = iv.fsum_bounds(x[None], x[None])
    exact = Fraction(1) + 1000 * Fraction(2) ** -60
    assert Fraction(float(lo[0])) <= exact <= Fraction(float(hi[0]))
    assert sum(x.tolist()) == 1.0 and Fraction(1.0) < exact  # the ordinary sum sits below the truth
