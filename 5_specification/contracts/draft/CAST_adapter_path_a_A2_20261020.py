"""Adapter for the finite_rne scope in contracts/CAST.yaml (path A A2 draft, 2026-10-20).

Draft: when staged as <stage>/adapters/CAST.py it reads <stage>/contracts/CAST.yaml.
Sources: finite f16/bf16/f32/f64 values, integers of i64/i32/i16/i8/u8 within their range, and booleans.
Targets: f16/bf16/f32/f64 only.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R

_CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "CAST.yaml"
COVERS = yaml.safe_load(_CONTRACT.read_text())["covers"]
PROFILE = ("finite_rne",)
DTYPES = {"f16", "bf16", "f32", "f64"}
INTS = {"i64", "i32", "i16", "i8", "u8"}
BOOLS = {"b8"}
SOURCES = DTYPES | INTS | BOOLS
FMT = {"f16": (11, -14, 15), "bf16": (8, -126, 127), "f32": (24, -126, 127), "f64": (53, -1022, 1023)}
INT_RANGE = {"i64": (-2 ** 63, 2 ** 63 - 1), "i32": (-2 ** 31, 2 ** 31 - 1), "i16": (-2 ** 15, 2 ** 15 - 1),
             "i8": (-2 ** 7, 2 ** 7 - 1), "u8": (0, 2 ** 8 - 1)}


def _dtype(value, name="dtype", allowed=DTYPES):
    try:
        dtype = A.dt(decode(value))
    except Invalid:
        raise Unsupported(f"{name} {value!r} is outside the finite_rne CAST profile")
    if dtype not in allowed:
        raise Unsupported(f"{name} {dtype!r} is outside the finite_rne CAST profile")
    return dtype


def _target(arguments):
    value = arguments.get("dtype")
    if value is None:
        raise Unsupported("aten._to_copy requires an explicit dtype in the finite_rne CAST profile")
    return _dtype(value, "target dtype")


def _leaves(v):
    if isinstance(v, list):
        for x in v:
            yield from _leaves(x)
    else:
        yield v


def _raw_integer(x, source):
    """An integer record value: a Python int (not bool), an integral finite float, or an integral 'p/q' string."""
    if isinstance(x, bool):
        raise Invalid(f"a boolean is not a value of {source}")
    if isinstance(x, int):
        n = x
    elif isinstance(x, float):
        if not math.isfinite(x) or not x.is_integer():
            raise Invalid(f"{x!r} is not an integer value of {source}")
        n = int(x)
    elif isinstance(x, str):
        q = decode(x)
        if not isinstance(q, Fraction) or q.denominator != 1:
            raise Invalid(f"{x!r} is not an integer value of {source}")
        n = int(q)
    else:
        raise Invalid(f"{x!r} is not an integer value of {source}")
    lo, hi = INT_RANGE[source]
    if not lo <= n <= hi:
        raise Invalid(f"{n} is outside the range of {source}")
    return n


def _checked_tensor(value, target):
    if not (isinstance(value, dict) and "t" in value):
        raise Invalid("a must be a tensor record")
    source = _dtype(value["t"], "source dtype", SOURCES)
    if source in DTYPES:
        tensor = decode(value)
        for item in tensor.a.flat:
            if not isinstance(item, R):
                raise Invalid("a is not a real tensor")
            if not item.finite:
                raise Unsupported("infinity is outside the finite_rne CAST profile")
            if not item.exact:
                raise Invalid("source tensor values must be exact")
            source_value = ieee.round_to(item, source)
            if not source_value.exact or source_value.v != item.v:
                raise Invalid(f"{item!r} is not exactly representable as {source}")
            if not ieee.round_to(item, target).finite:
                raise Unsupported(f"conversion of {item!r} overflows {target}")
        return tensor
    raw = value.get("v")
    for x in _leaves(raw):
        if source in BOOLS:
            if not isinstance(x, bool):
                raise Invalid(f"{x!r} is not a boolean value")
        else:
            n = _raw_integer(x, source)
            if not ieee.round_to(R(n), target).finite:
                raise Unsupported(f"conversion of {n} overflows {target}")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten) or tensor.dtype != source:
        raise Invalid("a must be a tensor")
    return tensor


def _decode_argument(op, name, value, arguments):
    if name == "a":
        return _checked_tensor(value, _target(arguments))
    if name == "dtype":
        return _dtype(value)
    return decode(value)


def _convert(a, dtype):
    return A.convert_element_type(a, dtype)


def _to_copy(a, *, dtype=None, layout=None, device=None, pin_memory=None,
             non_blocking=False, memory_format=None):
    if dtype is None:
        raise Unsupported("aten._to_copy requires an explicit dtype in the finite_rne CAST profile")
    if layout is not None or device is not None or pin_memory is not None or memory_format is not None:
        raise Unsupported("device, layout, pin_memory and memory_format are outside the finite_rne CAST profile")
    if non_blocking is not False:
        raise Unsupported("non_blocking copies are outside the finite_rne CAST profile")
    return A.convert_element_type(a, dtype)


OPS = {"prims.convert_element_type": _convert, "aten._to_copy": _to_copy}
reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)


# ---------------------------------------------------------------- diagnostic mutants (independent code)
def _round(q: Fraction, dtype: str, mode="rne", subnormals=True):
    """Separate integer-arithmetic implementation used only by diagnostic mutants."""
    p, emin, emax = FMT[dtype]
    if q == 0:
        return Fraction(0)
    sign, value = (-1, -q) if q < 0 else (1, q)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    while Fraction(2) ** exponent > value:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= value:
        exponent += 1
    if exponent < emin and not subnormals:
        return Fraction(0)
    exponent = max(exponent, emin)
    step = Fraction(2) ** (exponent - p + 1)
    quotient, remainder = divmod(value, step)
    if mode == "rne" and (2 * remainder > step or (2 * remainder == step and quotient % 2)):
        quotient += 1
    elif mode == "away" and 2 * remainder >= step:
        quotient += 1
    result = quotient * step
    maximum = (2 - Fraction(2) ** (1 - p)) * Fraction(2) ** emax
    if result > maximum:
        return math.inf if sign > 0 else -math.inf
    return sign * result


def _exact(x):
    """Exact rational of a stored element: R for floating sources, int or bool otherwise."""
    if isinstance(x, R):
        return Fraction(x.v)
    return Fraction(int(x))


def _elementwise(fn, a, target):
    tensor = A.as_ten(a)
    out = np.empty(tensor.a.shape, dtype=object)
    for i in np.ndindex(tensor.a.shape):
        out[i] = R(fn(tensor.a[i], tensor.dtype))
    return A.Ten(out, target)


def _wrong_convert(*, mode="rne", subnormals=True, identity=False):
    def convert(a, dtype):
        tensor, target = A.as_ten(a), A.dt(dtype)
        if identity:
            return _elementwise(lambda x, s: _exact(x), tensor, target)
        return _elementwise(lambda x, s: _round(_exact(x), target, mode, subnormals), tensor, target)
    return convert


def _integer_truncating(a, dtype):
    """CA.M5: inexact integer sources are truncated toward zero (floating sources keep RNE)."""
    target = A.dt(dtype)

    def one(x, source):
        if source in DTYPES:
            return _round(_exact(x), target)
        return _round(_exact(x), target, mode="trunc")
    return _elementwise(one, a, target)


def _bool_minus_one(a, dtype):
    """CA.M6: True is read as the all-ones signed bit pattern, i.e. -1."""
    target = A.dt(dtype)

    def one(x, source):
        if source in BOOLS:
            return Fraction(-1) if x else Fraction(0)
        return _round(_exact(x), target)
    return _elementwise(one, a, target)


def _u8_as_signed(a, dtype):
    """CA.M7: uint8 storage is reinterpreted as int8 (signed/unsigned confusion)."""
    target = A.dt(dtype)

    def one(x, source):
        n = _exact(x)
        if source == "u8" and n >= 128:
            n -= 256
        return _round(n, target)
    return _elementwise(one, a, target)


def _via_f32(a, dtype):
    """CA.M8: integer sources are first converted to f32 and then to the target (double rounding)."""
    target = A.dt(dtype)

    def one(x, source):
        q = _exact(x)
        if source not in DTYPES:
            q = _round(q, "f32")
        return _round(q, target)
    return _elementwise(one, a, target)


def _mutant(fn):
    return make_reference(COVERS, extra={
        "prims.convert_element_type": fn,
        "aten._to_copy": lambda a, *, dtype=None, **kwargs: fn(a, dtype),
    }, profiles=PROFILE, argument_decoder=_decode_argument)


MUTANTS = {
    "CA.M1": _mutant(_wrong_convert(mode="trunc")),
    "CA.M2": _mutant(_wrong_convert(mode="away")),
    "CA.M3": _mutant(_wrong_convert(subnormals=False)),
    "CA.M4": _mutant(_wrong_convert(identity=True)),
    "CA.M5": _mutant(_integer_truncating),
    "CA.M6": _mutant(_bool_minus_one),
    "CA.M7": _mutant(_u8_as_signed),
    "CA.M8": _mutant(_via_f32),
}
READINGS = {}
