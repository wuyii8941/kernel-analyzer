"""Rounding models of the narrow formats (FP8 / FP4) and the non-RTNE modes, against exact enumeration."""

from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

from kernel_analyzer.reference_eval import intervals as iv
from kernel_analyzer.reference_eval import numbers as nb

# fmt -> (exponent bits, mantissa bits, bias, encodings that are not finite values)
LAYOUT = {
    "f8E4M3FN": (4, 3, 7, "s1111111"),       # only S.1111.111 is NaN
    "f8E4M3FNUZ": (4, 3, 8, "10000000"),     # 0x80 is NaN, no negative zero
    "f8E5M2FNUZ": (5, 2, 16, "10000000"),
    "f8E4M3B11FNUZ": (4, 3, 11, "10000000"),
    "f4E2M1FN": (2, 1, 1, None),
    "f8E5M2": (5, 2, 15, "s11111xx"),        # IEEE-like: inf and NaN in the top exponent
}
EXACT_NAME = {"f8E4M3FN": "fp8e4m3fn", "f8E4M3FNUZ": "fp8e4m3fnuz", "f8E5M2FNUZ": "fp8e5m2fnuz",
              "f4E2M1FN": "fp4e2m1fn", "f8E5M2": "fp8e5m2"}


@lru_cache(maxsize=None)
def finite_values(fmt):
    """All finite non-negative values of the format, by decoding every bit pattern."""

    ebits, mbits, bias, _ = LAYOUT[fmt]
    vals = set()
    for e in range(1 << ebits):
        for m in range(1 << mbits):
            if fmt == "f8E4M3FN" and e == (1 << ebits) - 1 and m == (1 << mbits) - 1:
                continue
            if fmt == "f8E5M2" and e == (1 << ebits) - 1:
                continue
            if e == 0:
                v = Fraction(m, 1 << mbits) * Fraction(2) ** (1 - bias)
            else:
                v = (1 + Fraction(m, 1 << mbits)) * Fraction(2) ** (e - bias)
            vals.add(v)
    return tuple(sorted(vals))


@lru_cache(maxsize=None)
def signed_extended_grid(fmt):
    """The finite grid continued past the largest value with an unbounded exponent range (two binades)."""

    grid = finite_values(fmt)
    ebits, mbits, bias, _ = LAYOUT[fmt]
    top = grid[-1]
    ext = list(grid)
    emax_field = (1 << ebits) - 1 - bias
    for e in range(emax_field, emax_field + 3):
        for m in range(1 << mbits):
            v = (1 + Fraction(m, 1 << mbits)) * Fraction(2) ** e
            if v > top:
                ext.append(v)
    return tuple(sorted(set([-g for g in ext] + ext)))


def reference_round(x: Fraction, fmt, grid, mode, has_inf):
    """Exact rounding: round onto the unbounded-exponent grid, then overflow iff beyond the largest finite
    value.  IEEE formats saturate in the modes that round toward zero; formats without infinity report
    every overflow.  Returns a Fraction, "overflow" or "tie" (exact RN ties are skipped)."""

    import bisect

    full = signed_extended_grid(fmt)
    i = bisect.bisect_left(full, x)
    above = full[i]
    below = above if above == x else full[i - 1]
    if below == above:
        r = below
    elif mode == "rd":
        r = below
    elif mode == "ru":
        r = above
    elif mode == "rtz":
        r = below if x > 0 else above
    else:
        d1, d2 = x - below, above - x
        if d1 == d2:
            return "tie"
        r = below if d1 < d2 else above
    top = grid[-1]
    if abs(r) <= top:
        return r
    if not has_inf:
        return "overflow"
    away = mode == "rtne" or (mode == "ru" and x > 0) or (mode == "rd" and x < 0)
    return "overflow" if away else (top if x > 0 else -top)


@pytest.mark.parametrize("fmt", list(LAYOUT))
def test_max_finite_matches_the_bit_layout(fmt):
    assert iv.max_finite(fmt) == float(finite_values(fmt)[-1])
    if fmt in EXACT_NAME:
        assert nb.max_finite(EXACT_NAME[fmt]) == finite_values(fmt)[-1]


@pytest.mark.parametrize("fmt", list(LAYOUT))
@pytest.mark.parametrize("mode", ["rtne", "rtz", "rd", "ru"])
def test_rounding_matches_enumeration(fmt, mode):
    grid = finite_values(fmt)
    has_inf = iv.FLOAT_FORMATS[fmt][3]
    rng = np.random.default_rng(3)
    top = float(grid[-1])
    xs = np.concatenate([rng.uniform(-1.2 * top, 1.2 * top, 800),
                         rng.uniform(-4 * float(grid[1]), 4 * float(grid[1]), 300),
                         [float(g) for g in grid], [-float(g) for g in grid]])
    vals, pinf, ninf = iv.round_directed(xs, fmt, mode)
    for x, v, p, n in zip(xs, vals, pinf, ninf):
        want = reference_round(Fraction(float(x)), fmt, grid, mode, has_inf)
        if want == "tie":  # exact ties: RN-even; check against neighbours' parity through rint
            continue
        if want == "overflow":
            assert p or n, (fmt, mode, x, v)
            continue
        assert not (p or n), (fmt, mode, x, v)
        assert Fraction(float(v)) == want, (fmt, mode, x, v, want)
        if fmt in EXACT_NAME:
            assert nb.round_rational(Fraction(float(x)), EXACT_NAME[fmt], mode) == want


def test_e4m3fn_ties_and_overflow_edge():
    # 464 is halfway between 448 (mantissa 110, even) and the NaN pattern position 480: RN-even gives 448
    v, of = iv.round_nearest_even(np.array([464.0, 464.5, 447.0]), "f8E4M3FN")
    assert v[0] == 448.0 and not of[0]
    assert of[1]
    assert v[2] == 448.0 and not of[2]
    assert nb.round_rational(Fraction(479), "fp8e4m3fn", "rtz") == 448  # toward zero lands on 448
    with pytest.raises(nb.Overflow) as exc:
        nb.round_rational(Fraction(481), "fp8e4m3fn", "rtz")  # 480 would be the NaN encoding
    assert exc.value.implementation_defined
