"""Rigorous enclosures of MATHFN elementary functions with python-flint Arb ball arithmetic (package D, 2026-10-20).

Why not kernel_spec.real for these functions: with mpmath 1.3.0, `mpmath.iv.exp` returns a zero-width interval that
does not contain exp(x) for small arguments (observed for |x| between about 2^-55 and 2^-271 at 256 bits), and
`real.tanh`, `real.sigmoid`, `real.expm1` and `real.log1p` inherit the error (see
evidence/arith_mathfn_index_domains.md). Arb bounds its own error, including the argument reduction of sin and cos
for arguments up to the binary64 maximum, so these enclosures stay correct over the whole finite range used here.

Input: an exact finite real (R with a Fraction value). Output: R holding an outward-rounded mpmath interval built from
the exact endpoints of the Arb ball, or an exact R at the points where the value is rational (x = 0 for all functions,
x = 1 for log). Values whose endpoints would need astronomically large exponents (exp of huge arguments) are not
requested by MATHFN, whose adapter keeps the |x| <= 100 bound for exp, expm1 and sigmoid.
"""
from __future__ import annotations

from fractions import Fraction

import flint

from .num import Undefined, from_interval
from .real import R

WORK_PREC = 256 + 64      # bits; the result is re-enclosed at mpmath's 256 bits

_EXACT = {"exp": {0: 1}, "expm1": {0: 0}, "log": {1: 0}, "log1p": {0: 0}, "sin": {0: 0}, "cos": {0: 1},
          "tanh": {0: 0}, "sigmoid": {0: Fraction(1, 2)}}
_RANGE = {"sin": (-1, 1), "cos": (-1, 1), "tanh": (-1, 1), "sigmoid": (0, 1)}


def _arb(q: Fraction):
    return flint.arb(flint.fmpq(int(q.numerator), int(q.denominator)))


def _exact(arf) -> Fraction:
    mantissa, exponent = arf.man_exp()
    return Fraction(int(mantissa)) * Fraction(2) ** int(exponent)


def _ball(name, a):
    if name == "exp":
        return a.exp()
    if name == "expm1":
        return a.expm1()
    if name == "log":
        return a.log()
    if name == "log1p":
        return a.log1p()
    if name == "sin":
        return a.sin()
    if name == "cos":
        return a.cos()
    if name == "tanh":
        return a.tanh()
    if name == "sigmoid":
        return 1 / (1 + (-a).exp())
    raise KeyError(name)


def evaluate(name: str, x: R) -> R:
    """Enclosure of name(x) for an exact finite x inside the function's real domain (checked by the caller)."""
    x = x if isinstance(x, R) else R(x)
    if not x.exact or not x.finite:
        raise ValueError("arb evaluation takes an exact finite real")
    q = x.v
    if q in _EXACT[name]:
        return R(Fraction(_EXACT[name][q]))
    saved = flint.ctx.prec
    flint.ctx.prec = WORK_PREC
    try:
        ball = _ball(name, _arb(q))
        if not ball.is_finite():
            raise Undefined(f"{name} has no finite enclosure at this point")
        lo, hi = _exact(ball.lower()), _exact(ball.upper())
    finally:
        flint.ctx.prec = saved
    if name in _RANGE:
        lo, hi = max(lo, Fraction(_RANGE[name][0])), min(hi, Fraction(_RANGE[name][1]))
    return R(from_interval(lo, hi))
