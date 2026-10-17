"""Shared helpers of the optimizer group (E1_optim): strict input reading for explicit optimizer pre-states, and small
exact-arithmetic building blocks (binary floating-point rounding of a rational, used by mutants that model storage
rounding and by the stochastic-rounding contract).

Nothing here is a reference of a contract by itself; each adapter states which clause a helper serves.
"""
from __future__ import annotations

import math
from fractions import Fraction

from .num import Invalid, rational, vector


def flag(inp: dict, key: str, default=None) -> bool:
    """A boolean option; a missing key takes `default` (None: required). Non-booleans are invalid, never coerced."""
    if key not in inp:
        if default is None:
            raise Invalid(f"missing option {key}")
        return default
    v = inp[key]
    if not isinstance(v, bool):
        raise Invalid(f"{key} must be a boolean, got {v!r}")
    return v


def state_dict(inp: dict, keys: set) -> dict:
    st = inp.get("state")
    if not isinstance(st, dict) or set(st) != set(keys):
        raise Invalid(f"state must have exactly the keys {sorted(keys)}")
    return st


def vec(xs, what, n=None):
    """A non-empty vector of exact rationals (finite; NaN / inf strings are invalid)."""
    v = vector(xs, what, n)
    if not v:
        raise Invalid(f"{what}: empty")
    return v


def grad_or_none(inp: dict, n: int):
    g = inp.get("gradient", "missing")
    if g == "missing":
        raise Invalid("missing gradient (use null for a parameter without gradient)")
    return None if g is None else vec(g, "gradient", n)


def hyper(inp: dict, keys, nonneg=()) -> dict:
    hp = {k: rational(inp[k], k) if k in inp else _missing(k) for k in keys}
    for k in nonneg:
        if hp[k] < 0:
            raise Invalid(f"{k} must be >= 0")
    return hp


def _missing(k):
    raise Invalid(f"missing hyperparameter {k}")


# ---------------------------------------------------------------- binary floating point
def round_nearest_even(x: Fraction, p: int) -> Fraction:
    """x rounded to the nearest binary number with p significant bits (ties to even); unbounded exponent range
    (the values used here are far from overflow and underflow)."""
    x = Fraction(x)
    if x == 0:
        return Fraction(0)
    s = -1 if x < 0 else 1
    a = abs(x)
    e = math.floor(math.log2(a.numerator) - math.log2(a.denominator))
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    ulp = Fraction(2) ** (e - p + 1)
    q, r = divmod(a, ulp)
    q = int(q)
    if r * 2 > ulp or (r * 2 == ulp and q % 2 == 1):
        q += 1
    return s * q * ulp


def bf16_rn(x: Fraction) -> Fraction:
    return round_nearest_even(x, 8)


def bf16_toward_zero(x: Fraction) -> tuple[Fraction, Fraction]:
    """(t, u): t = the bfloat16 value of largest magnitude with |t| <= |x| and the sign of x, u = the bfloat16 spacing
    above |t| (normal range). For x = 0: (0, 0)."""
    x = Fraction(x)
    if x == 0:
        return Fraction(0), Fraction(0)
    a = abs(x)
    e = 0
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    u = Fraction(2) ** (e - 7)
    t = (a // u) * u
    return (t if x > 0 else -t), u
