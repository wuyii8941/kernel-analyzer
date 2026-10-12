"""Declared values of every catalogued operation (the answers of RQ1).

``REAL[name]`` maps finite exact inputs (Fractions) to the exact real result: a Fraction where the operation is
algebraic, an mpmath number otherwise (160 bits), or a float nan / +-inf where the declared result is not finite.
``NP[name]`` gives the IEEE special-value behaviour in float64 (numpy / scipy) and is used only when an input is
nan / inf or the result is zero (for the sign of zero).  Integer operations are in ``INTOP`` (Python integers, with
``UB`` for results the declared semantics leaves undefined: poison, division by zero, out-of-range conversion).
Names are TTIR operation names or libdevice symbols without the ``__nv_`` prefix and type suffix.
"""
from __future__ import annotations

import math
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import scipy.special as sp

from exact import floor_log2 as _floor_log2, mpf_to_frac

NAN = float("nan")
INF = float("inf")


class UB:  # undefined in the declared semantics
    def __init__(self, why):
        self.why = why

    def __repr__(self):
        return f"UB({self.why})"


def _m(x):
    return mp.mpf(x.numerator) / x.denominator if isinstance(x, F) else mp.mpf(x)


def _pos(fn):
    def g(x):
        if x < 0:
            return NAN
        if x == 0:
            return None  # special path (pole or zero)
        return fn(_m(x))
    return g


def _unit(fn, closed=True):
    def g(x):
        if abs(x) > 1 or (not closed and abs(x) == 1):
            return NAN if abs(x) > 1 else None
        return fn(_m(x))
    return g


def _pow(x, y):
    if x == 0:
        return None
    if x < 0:
        if y.denominator != 1:
            return NAN
        v = (-x) ** int(y) if abs(int(y)) < 4096 and abs(x.numerator).bit_length() < 64 else None
        if v is not None:
            return v if int(y) % 2 == 0 else -v
        r = mp.power(_m(-x), _m(y))
        return r if int(y) % 2 == 0 else -r
    if y.denominator == 1 and abs(int(y)) <= 64 and x.numerator.bit_length() + x.denominator.bit_length() < 200:
        return x ** int(y)
    return mp.power(_m(x), _m(y))


def _fmod(x, y):
    if y == 0:
        return NAN
    q = abs(x / y)
    n = q.numerator // q.denominator
    r = abs(x) - n * abs(y)
    return r if x >= 0 else -r


def _remainder(x, y):
    if y == 0:
        return NAN
    q = x / y
    n = math.floor(q)
    d = q - n
    if d > F(1, 2) or (d == F(1, 2) and n % 2 == 1):
        n += 1
    return x - n * y


def _round_half_away(x):
    n = math.floor(abs(x) + F(1, 2))
    return F(n if x >= 0 else -n)


def _round_half_even(x):
    n = math.floor(x)
    d = x - n
    if d > F(1, 2) or (d == F(1, 2) and n % 2 == 1):
        n += 1
    return F(n)


def _lgamma(x):
    if x <= 0 and x.denominator == 1:
        return INF
    return mp.log(abs(mp.gamma(_m(x))))


def _tgamma(x):
    if x <= 0 and x.denominator == 1:
        return NAN if x < 0 else None
    return mp.gamma(_m(x))


REAL = {
    # exact algebraic
    "addf": lambda a, b: a + b, "subf": lambda a, b: a - b, "mulf": lambda a, b: a * b,
    "divf": lambda a, b: (a / b) if b != 0 else None, "precise_divf": lambda a, b: (a / b) if b != 0 else None,
    "fma": lambda a, b, c: a * b + c, "remf": _fmod, "fmod": _fmod, "remainder": _remainder,
    "maximumf": lambda a, b: max(a, b), "minimumf": lambda a, b: min(a, b),
    "maxnumf": lambda a, b: max(a, b), "minnumf": lambda a, b: min(a, b),
    "fmax": lambda a, b: max(a, b), "fmin": lambda a, b: min(a, b),
    "fdim": lambda a, b: max(a - b, F(0)),
    "copysign": lambda a, b: abs(a) if b > 0 else -abs(a), "negf": lambda a: -a, "absf": abs, "fabs": abs, "abs_f": abs,
    "ceil": lambda a: F(math.ceil(a)), "floor": lambda a: F(math.floor(a)), "trunc": lambda a: F(math.trunc(a)),
    "round": _round_half_away, "roundeven": _round_half_even, "rint": _round_half_even,
    "nearbyint": _round_half_even,
    "sqrt": _pos(mp.sqrt), "precise_sqrt": _pos(mp.sqrt),
    "rsqrt": _pos(lambda x: 1 / mp.sqrt(x)), "rcp": lambda a: (1 / a) if a != 0 else None,
    "cbrt": lambda a: mp.cbrt(_m(a)) if a >= 0 else -mp.cbrt(_m(-a)),
    "rcbrt": lambda a: (1 / (mp.cbrt(_m(a)) if a >= 0 else -mp.cbrt(_m(-a)))) if a != 0 else None,
    "exp": lambda a: mp.exp(_m(a)), "exp2": lambda a: mp.power(2, _m(a)), "exp10": lambda a: mp.power(10, _m(a)),
    "expm1": lambda a: mp.expm1(_m(a)),
    "log": _pos(mp.log), "log2": _pos(lambda x: mp.log(x, 2)), "log10": _pos(lambda x: mp.log10(x)),
    "log1p": lambda a: (NAN if a < -1 else (None if a == -1 else mp.log1p(_m(a)))),
    "sin": lambda a: mp.sin(_m(a)), "cos": lambda a: mp.cos(_m(a)), "tan": lambda a: mp.tan(_m(a)),
    "sinpi": lambda a: mp.sinpi(_m(a)), "cospi": lambda a: mp.cospi(_m(a)),
    "asin": _unit(mp.asin), "acos": _unit(mp.acos), "atan": lambda a: mp.atan(_m(a)),
    "sinh": lambda a: mp.sinh(_m(a)), "cosh": lambda a: mp.cosh(_m(a)), "tanh": lambda a: mp.tanh(_m(a)),
    "asinh": lambda a: mp.asinh(_m(a)), "acosh": lambda a: (NAN if a < 1 else mp.acosh(_m(a))),
    "atanh": _unit(mp.atanh, closed=False),
    "erf": lambda a: mp.erf(_m(a)), "erfc": lambda a: mp.erfc(_m(a)),
    "erfcx": lambda a: mp.exp(_m(a) ** 2) * mp.erfc(_m(a)),
    "erfinv": _unit(mp.erfinv, closed=False),
    "erfcinv": lambda a: (NAN if (a < 0 or a > 2) else (None if a in (0, 2) else mp.erfinv(1 - _m(a)))),
    "normcdf": lambda a: mp.ncdf(_m(a)),
    "normcdfinv": lambda a: (NAN if (a < 0 or a > 1) else (None if a in (0, 1) else mp.sqrt(2) * mp.erfinv(2 * _m(a) - 1))),
    "lgamma": _lgamma, "tgamma": _tgamma,
    "j0": lambda a: mp.besselj(0, _m(a)), "j1": lambda a: mp.besselj(1, _m(a)),
    "y0": _pos(lambda x: mp.bessely(0, x)), "y1": _pos(lambda x: mp.bessely(1, x)),
    "cyl_bessel_i0": lambda a: mp.besseli(0, _m(a)), "cyl_bessel_i1": lambda a: mp.besseli(1, _m(a)),
    "powf": _pow, "pow": _pow,
    "atan2": lambda y, x: (None if (x == 0 and y == 0) else mp.atan2(_m(y), _m(x))),
    "hypot": lambda a, b: mp.sqrt(_m(a) ** 2 + _m(b) ** 2),
    "rhypot": lambda a, b: (1 / mp.sqrt(_m(a) ** 2 + _m(b) ** 2)) if (a, b) != (0, 0) else None,
    "norm3d": lambda a, b, c: mp.sqrt(_m(a) ** 2 + _m(b) ** 2 + _m(c) ** 2),
    "rnorm3d": lambda a, b, c: (1 / mp.sqrt(_m(a) ** 2 + _m(b) ** 2 + _m(c) ** 2)) if (a, b, c) != (0, 0, 0) else None,
    "norm4d": lambda a, b, c, d: mp.sqrt(_m(a) ** 2 + _m(b) ** 2 + _m(c) ** 2 + _m(d) ** 2),
    "rnorm4d": lambda a, b, c, d: ((1 / mp.sqrt(_m(a) ** 2 + _m(b) ** 2 + _m(c) ** 2 + _m(d) ** 2))
                                    if (a, b, c, d) != (0, 0, 0, 0) else None),
    "saturatef": lambda a: min(max(a, F(0)), F(1)),
    "logb": lambda a: F(_floor_log2(abs(a))) if a != 0 else None,
    "clampf": lambda x, lo, hi: min(max(x, lo), hi), "clampf_nan": lambda x, lo, hi: min(max(x, lo), hi),
    "rcp64h": lambda a: (1 / a) if a != 0 else None,
    "fast_dividef": lambda a, b: (a / b) if b != 0 else None,
}
REAL["fast_sinf"] = REAL["sin"]
REAL["fast_cosf"] = REAL["cos"]
REAL["fast_tanf"] = REAL["tan"]
REAL["fast_expf"] = REAL["exp"]
REAL["fast_exp10f"] = REAL["exp10"]
REAL["fast_logf"] = REAL["log"]
REAL["fast_log2f"] = REAL["log2"]
REAL["fast_log10f"] = REAL["log10"]
REAL["fast_powf"] = REAL["pow"]
for _m_ in ("rn", "rz", "rd", "ru"):
    REAL[f"add_{_m_}"] = REAL["addf"]
    REAL[f"sub_{_m_}"] = REAL["subf"]
    REAL[f"mul_{_m_}"] = REAL["mulf"]
    REAL[f"div_{_m_}"] = REAL["divf"]
    REAL[f"fma_{_m_}"] = REAL["fma"]
    REAL[f"rcp_{_m_}"] = REAL["rcp"]
    REAL[f"sqrt_{_m_}"] = REAL["sqrt"]
REAL["rsqrt_rn"] = REAL["rsqrt"]

# numpy float64 special-value behaviour (only consulted for nan / inf inputs, pole / zero results)
NP = {
    "addf": np.add, "subf": np.subtract, "mulf": np.multiply, "divf": np.divide, "precise_divf": np.divide,
    "fma": lambda a, b, c: a * b + c, "remf": np.fmod, "fmod": np.fmod, "remainder": np.remainder,
    "maximumf": np.maximum, "minimumf": np.minimum, "maxnumf": np.fmax, "minnumf": np.fmin, "fmax": np.fmax,
    "fmin": np.fmin, "fdim": lambda a, b: np.where(np.isnan(a) | np.isnan(b), np.nan, np.maximum(a - b, 0.0)),
    "copysign": np.copysign, "negf": np.negative, "absf": np.abs, "fabs": np.abs,
    "ceil": np.ceil, "floor": np.floor, "trunc": np.trunc, "round": lambda a: np.copysign(np.floor(np.abs(a) + 0.5), a),
    "roundeven": np.rint, "rint": np.rint, "nearbyint": np.rint, "sqrt": np.sqrt, "precise_sqrt": np.sqrt,
    "rsqrt": lambda a: 1.0 / np.sqrt(a), "rcp": lambda a: 1.0 / a, "cbrt": np.cbrt, "rcbrt": lambda a: 1.0 / np.cbrt(a),
    "exp": np.exp, "exp2": np.exp2, "exp10": lambda a: np.power(10.0, a), "expm1": np.expm1, "log": np.log,
    "log2": np.log2, "log10": np.log10, "log1p": np.log1p, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinpi": lambda a: np.sin(np.pi * a), "cospi": lambda a: np.cos(np.pi * a), "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "asinh": np.arcsinh, "acosh": np.arccosh,
    "atanh": np.arctanh, "erf": sp.erf, "erfc": sp.erfc, "erfcx": sp.erfcx, "erfinv": sp.erfinv, "erfcinv": sp.erfcinv,
    "normcdf": sp.ndtr, "normcdfinv": sp.ndtri, "lgamma": sp.gammaln, "tgamma": sp.gamma, "j0": sp.j0, "j1": sp.j1,
    "y0": sp.y0, "y1": sp.y1, "cyl_bessel_i0": sp.i0, "cyl_bessel_i1": sp.i1, "powf": np.power, "pow": np.power,
    "atan2": np.arctan2, "hypot": np.hypot, "rhypot": lambda a, b: 1.0 / np.hypot(a, b),
    "norm3d": lambda a, b, c: np.sqrt(a * a + b * b + c * c), "rnorm3d": lambda a, b, c: 1.0 / np.sqrt(a * a + b * b + c * c),
    "norm4d": lambda a, b, c, d: np.sqrt(a * a + b * b + c * c + d * d),
    "rnorm4d": lambda a, b, c, d: 1.0 / np.sqrt(a * a + b * b + c * c + d * d),
    "saturatef": lambda a: np.where(np.isnan(a), 0.0, np.clip(a, 0.0, 1.0)), "fast_dividef": np.divide,
    "logb": lambda a: np.where(a == 0, -np.inf, np.floor(np.log2(np.abs(a)))), "rcp64h": lambda a: 1.0 / a,
    "clampf": lambda x, lo, hi: np.fmin(np.fmax(x, lo), hi),
    "clampf_nan": lambda x, lo, hi: np.minimum(np.maximum(x, lo), hi),
}
for _k in list(REAL):
    if _k not in NP:
        base = {"fast_sinf": "sin", "fast_cosf": "cos", "fast_tanf": "tan", "fast_expf": "exp", "fast_exp10f": "exp10",
                "fast_logf": "log", "fast_log2f": "log2", "fast_log10f": "log10", "fast_powf": "pow",
                "rsqrt_rn": "rsqrt"}.get(_k)
        if base is None:
            for pre in ("add", "sub", "mul", "div", "fma", "rcp", "sqrt"):
                if _k.startswith(pre + "_"):
                    base = {"add": "addf", "sub": "subf", "mul": "mulf", "div": "divf"}.get(pre, pre)
        if base is not None:
            NP[_k] = NP[base]


def real_value(name, args):
    """Exact real result for finite Fraction args: Fraction, mpf, float nan / inf, or None (use the special path)."""
    fn = REAL[name]
    try:
        v = fn(*args)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None
    if isinstance(v, mp.mpc):
        return NAN
    return v


def as_frac(v):
    """mpf / Fraction -> Fraction (exact for the binary mpf value)."""
    if isinstance(v, F):
        return v
    if isinstance(v, int):
        return F(v)
    if isinstance(v, mp.mpf):
        if not mp.isfinite(v):
            return float(v)
        return mpf_to_frac(v)
    return v
