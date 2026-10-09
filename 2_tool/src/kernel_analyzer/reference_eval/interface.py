"""Interface and compile-time constant inventory of a captured launch.

Two kinds of rounding happen before the kernel computes anything.  Reports list them apart from the
kernel's own arithmetic, so that an interface conversion is not read as a kernel deviation and a
specification is never changed silently to make a difference disappear:

* runtime scalars: a Python value passed to the launch is converted to the kernel parameter type; the
  record keeps the value passed, the parameter type and the value the kernel receives.  A specification
  evaluated on the received values (f_interface) differs from one on the passed values (f_published) by
  the interface term f_interface - f_published, an upstream node with a designated producer (the launch);
* compile-time constants: constexpr arithmetic and literals become typed constants in the TTIR.  Every
  float constant is listed with its exact value; a constant that is the correctly rounded value of a
  simple rational or of a common irrational, without being equal to it, is listed with that candidate and
  the relative rounding.  Rationals p/q with q <= Q lie about 1/Q**2 apart at any magnitude, so a fixed
  bound would "read back" almost any value; the bound is Q(x) = sqrt(2 eps / ulp(x)) with eps = 1e-3, which
  keeps the chance that an arbitrary value matches some candidate below about 1e-3 (3072/7 at 438.86 and 1/7
  at 0.14 qualify; a random float32 near 500 rarely does).
  The candidate is a reading of a value that the TTIR
  stores only after rounding (the intended value is not in the IR), and the record says so.
"""

from __future__ import annotations

import math
from fractions import Fraction

import gmpy2
import numpy as np
from gmpy2 import mpq

from . import intervals as iv
from . import numbers as nb
from .ttir_eval import _constant

# kernel parameter type -> (exact-rational format name, interval format name)
_PARAM_FORMATS = {"fp32": ("fp32", "f32"), "fp16": ("fp16", "f16"), "bf16": ("bf16", "bf16"), "fp64": ("fp64", "f64")}
_ELEM_FORMATS = {"f32": "fp32", "f16": "fp16", "bf16": "bf16", "f64": "fp64"}
_INT_RANGES = {"i1": (0, 1), "i8": (-2 ** 7, 2 ** 7 - 1), "i16": (-2 ** 15, 2 ** 15 - 1),
               "i32": (-2 ** 31, 2 ** 31 - 1), "i64": (-2 ** 63, 2 ** 63 - 1), "u32": (0, 2 ** 32 - 1),
               "u64": (0, 2 ** 64 - 1)}
_PRECISION = {"fp32": 24, "fp16": 11, "bf16": 8, "fp64": 53}
CHANCE = 1e-3


def max_denominator(value: float, fmt: str) -> int:
    """Largest denominator whose rationals are sparse enough near ``value`` (see the module docstring)."""

    _, e = math.frexp(abs(value))  # value = m 2**e, 0.5 <= m < 1
    ulp = 2.0 ** (e - _PRECISION[fmt])
    return max(1, int(math.sqrt(2 * CHANCE / ulp)))


def _irrationals():
    gmpy2.get_context().precision = 256
    two, pi = gmpy2.mpfr(2), gmpy2.const_pi()
    e, ln2 = gmpy2.exp(1), gmpy2.log(two)
    return {"sqrt(2)": gmpy2.sqrt(two), "1/sqrt(2)": 1 / gmpy2.sqrt(two), "pi": pi, "1/pi": 1 / pi,
            "2*pi": 2 * pi, "e": e, "ln(2)": ln2, "1/ln(2)": 1 / ln2, "log10(2)": ln2 / gmpy2.log(gmpy2.mpfr(10)),
            "ln(10)": gmpy2.log(gmpy2.mpfr(10)), "sqrt(pi)": gmpy2.sqrt(pi), "2/sqrt(pi)": 2 / gmpy2.sqrt(pi),
            "1/sqrt(2*pi)": 1 / gmpy2.sqrt(2 * pi), "sqrt(2/pi)": gmpy2.sqrt(2 / pi)}


_IRRATIONALS = None


def _rounds_to(candidate, value: float, fmt: str) -> bool:
    try:
        return float(nb.round_rational(mpq(candidate), fmt, "rtne")) == value
    except nb.Overflow:
        return False


def _is_power_of_two(value: float) -> bool:
    if value == 0 or not math.isfinite(value):
        return False
    m, _ = math.frexp(abs(value))
    return m == 0.5


def constant_record(value: float, elem: str) -> dict:
    """Exact value of a typed constant and, when it is a rounded simple value, the candidate."""

    global _IRRATIONALS
    rec = {"type": elem, "value": repr(value), "hex": float(value).hex(), "power_of_two": _is_power_of_two(value)}
    fmt = _ELEM_FORMATS.get(elem)
    if fmt is None or not math.isfinite(value) or value == 0:
        return rec
    q = Fraction(value)
    rec["dyadic_denominator_bits"] = q.denominator.bit_length() - 1
    cand = q.limit_denominator(max_denominator(value, fmt))
    if cand != q and _rounds_to(cand, value, fmt):
        rec["rounded_from"] = {"candidate": f"{cand.numerator}/{cand.denominator}",
                               "relative_rounding": float((q - cand) / cand),
                               "note": "candidate read from the stored value; the intended value is not in the IR"}
        return rec
    if _IRRATIONALS is None:
        _IRRATIONALS = _irrationals()
    for name, c in _IRRATIONALS.items():
        for sign in (1, -1):
            exact = sign * c
            if _rounds_to(exact, value, fmt):
                rec["rounded_from"] = {"candidate": ("-" if sign < 0 else "") + name,
                                       "relative_rounding": float((gmpy2.mpfr(value) - exact) / exact),
                                       "note": "candidate read from the stored value; the intended value is not "
                                               "in the IR"}
                return rec
    return rec


def scalar_interface(launch) -> list:
    """Runtime scalars of a launch: value passed, kernel parameter type, value received, rounding."""

    rows = []
    for a in launch.args:
        if a.kind not in ("float", "int", "bool"):
            continue
        row = {"name": a.name, "passed": repr(a.value), "parameter_type": a.signature_type,
               "compile_time": bool(a.constexpr)}
        if a.constexpr:
            row["received"] = repr(a.value)
            row["note"] = "constexpr: specialized into the kernel; derived constants appear in the TTIR"
        elif a.signature_type in _PARAM_FORMATS:
            fmt, ifmt = _PARAM_FORMATS[a.signature_type]
            received = float(iv.round_nearest_even(np.array([float(a.value)]), ifmt)[0][0])
            row.update(received=repr(received), exact=received == float(a.value))
            if received != float(a.value) and a.value != 0:
                row["relative_rounding"] = float((Fraction(received) - Fraction(float(a.value))) / Fraction(float(a.value)))
        elif a.signature_type in _INT_RANGES:
            lo, hi = _INT_RANGES[a.signature_type]
            row.update(received=repr(int(a.value)), exact=lo <= int(a.value) <= hi)
        else:
            row["received"] = None
            row["note"] = "parameter type without a conversion model"
        rows.append(row)
    return rows


def float_constants(module) -> list:
    """Every float constant of the kernel's TTIR (splat constants once per value and type)."""

    seen, rows = set(), []
    for op in module.entry().walk():
        if op.name != "arith.constant" or not op.result_types:
            continue
        elem = op.result_types[0].elem
        if elem not in _ELEM_FORMATS:
            continue
        try:
            value = float(_constant(op).lo.reshape(-1)[0])
        except Exception:  # noqa: BLE001 -- non-splat constants are reported as such
            rows.append({"node": op.node_id, "type": elem, "value": None, "note": "non-splat constant"})
            continue
        key = (elem, value)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"node": op.node_id, **constant_record(value, elem)})
    return rows


def inventory(launch, module) -> dict:
    scalars = scalar_interface(launch)
    constants = float_constants(module)
    return {"runtime_scalars": scalars, "float_constants": constants,
            "rounded_runtime_scalars": [s["name"] for s in scalars if s.get("exact") is False],
            "rounded_compile_time_constants": [c for c in constants if "rounded_from" in c],
            "note": "interface rounding (runtime scalars) is an upstream node of the launch; compile-time "
                    "constants are part of the declared semantics of the TTIR and enter K_R as stored"}
