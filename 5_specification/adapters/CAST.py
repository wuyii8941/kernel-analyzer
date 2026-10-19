"""Adapter for the reviewed finite_rne scope in contracts/CAST.yaml."""
from __future__ import annotations

import math
from fractions import Fraction

import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "CAST.yaml").read_text())["covers"]
PROFILE = ("finite_rne",)
DTYPES = {"f16", "bf16", "f32", "f64"}
FMT = {"f16": (11, -14, 15), "bf16": (8, -126, 127), "f32": (24, -126, 127), "f64": (53, -1022, 1023)}


def _dtype(value, name="dtype"):
    try:
        dtype = A.dt(decode(value))
    except Invalid:
        raise Unsupported(f"{name} {value!r} is outside the finite_rne CAST profile")
    if dtype not in DTYPES:
        raise Unsupported(f"{name} {dtype!r} is outside the finite_rne CAST profile")
    return dtype


def _target(arguments):
    value = arguments.get("dtype")
    if value is None:
        raise Unsupported("aten._to_copy requires an explicit dtype in the finite_rne CAST profile")
    return _dtype(value, "target dtype")


def _checked_tensor(value, target):
    source = _dtype(value["t"], "source dtype") if isinstance(value, dict) and "t" in value else None
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid("a must be a tensor")
    source = source or _dtype(tensor.dtype, "source dtype")
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
        target_value = ieee.round_to(item, target)
        if not target_value.finite:
            raise Unsupported(f"conversion of {item!r} overflows {target}")
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


def _wrong_convert(*, mode="rne", subnormals=True, identity=False):
    def convert(a, dtype):
        tensor, target = A.as_ten(a), A.dt(dtype)
        if identity:
            return A.Ten(tensor.a.copy(), target)
        return A.unary(lambda x: R(_round(Fraction(x.v), target, mode, subnormals)), tensor, out_dtype=target)
    return convert


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
}
READINGS = {}
