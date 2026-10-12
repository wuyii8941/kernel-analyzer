"""Exact arithmetic for the answers: binary formats, rounding of exact rationals, and declared values of operations.

Independent of the tool and of the device: rounding is done on Python ``fractions.Fraction`` from the IEEE 754
definitions (ported from the three_plans_v3 oracle, with directed modes, round-to-nearest-away and the fp8 / tf32
formats); transcendental values come from mpmath at 160 bits and are rounded through the same rational path.
"""
from __future__ import annotations

import math
from fractions import Fraction as F

import mpmath
import numpy as np

mpmath.mp.prec = 160

# precision p (with the hidden bit), emin, emax, has_inf
FORMATS = {"f16": (11, -14, 15, True), "bf16": (8, -126, 127, True), "f32": (24, -126, 127, True),
           "f64": (53, -1022, 1023, True), "tf32": (11, -126, 127, True), "f8e4m3": (4, -6, 8, False),
           "f8e5m2": (3, -14, 15, True)}
INF = float("inf")


def floor_log2(x: F) -> int:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while F(2) ** e > x:
        e -= 1
    while F(2) ** (e + 1) <= x:
        e += 1
    return e


def max_finite(fmt: str) -> F:
    p, emin, emax, has_inf = FORMATS[fmt]
    if fmt == "f8e4m3":
        return F(448)
    return (2 - F(2) ** (1 - p)) * F(2) ** emax


def round_frac(x: F, fmt: str, mode: str = "rne"):
    """Round an exact rational into ``fmt``: a Fraction, or +-inf (float) on overflow, or nan for e4m3 overflow.
    Modes: rne, rna (nearest, ties away), rz, ru, rd."""
    p, emin, emax, has_inf = FORMATS[fmt]
    if x == 0:
        return F(0)
    s = -1 if x < 0 else 1
    a = abs(x)
    e = max(floor_log2(a), emin)
    ulp = F(2) ** (e - p + 1)
    q, r = divmod(a, ulp)
    if mode == "rne":
        if r * 2 > ulp or (r * 2 == ulp and q % 2 == 1):
            q += 1
    elif mode == "rna":
        if r * 2 >= ulp:
            q += 1
    elif mode == "rz":
        pass
    elif mode == "ru":
        q += 1 if (r > 0 and s > 0) else 0
    elif mode == "rd":
        q += 1 if (r > 0 and s < 0) else 0
    else:
        raise ValueError(mode)
    v = q * ulp
    m = max_finite(fmt)
    if v > m:
        directed_down = mode == "rz" or (mode == "ru" and s < 0) or (mode == "rd" and s > 0)
        if directed_down:
            return s * m
        if not has_inf:
            return float("nan")
        return s * INF
    return s * v


def ulp_at(x: F, fmt: str) -> F:
    """Spacing of ``fmt`` at |x| (the subnormal spacing below the normal range)."""
    p, emin, emax, _ = FORMATS[fmt]
    if x == 0:
        return F(2) ** (emin - p + 1)
    return F(2) ** (max(floor_log2(abs(x)), emin) - p + 1)


def to_frac(v: float) -> F:
    return F(v)


def mpf_to_frac(v) -> F:
    """Exact rational value of a finite mpf (``man_exp`` drops the sign; the raw tuple keeps it)."""
    sign, man, exp, _ = mpmath.mpf(v)._mpf_
    m = -int(man) if sign else int(man)
    return F(m) * (F(2) ** int(exp)) if exp >= 0 else F(m, 2 ** (-int(exp)))


def is_subnormal(v: float, fmt: str) -> bool:
    p, emin, emax, _ = FORMATS[fmt]
    return v != 0 and math.isfinite(v) and abs(v) < 2.0 ** emin


# ----------------------------------------------------------------------------- numpy views of the formats
NP = {"f16": np.float16, "f32": np.float32, "f64": np.float64}


def bf16_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def f64_to_bf16_bits_rne(x: np.ndarray) -> np.ndarray:
    """Exact RNE of float64 values that are themselves bf16-representable-or-not, through the rational path (slow,
    used only to construct inputs)."""
    out = np.empty(x.shape, np.uint16)
    for i, v in enumerate(x.ravel()):
        out.ravel()[i] = frac_to_bits(round_frac(F(float(v)), "bf16") if math.isfinite(v) else v, "bf16")
    return out


def frac_to_bits(v, fmt: str) -> int:
    """Bit pattern of a value representable in fmt (Fraction or +-inf / nan float)."""
    if fmt == "bf16":
        f = np.array([float(v)], np.float32)
        return int(f.view(np.uint32)[0] >> 16)
    raise NotImplementedError(fmt)
