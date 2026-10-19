"""Reviewed finite clip_grad_value_ post-state from contracts/CLIP.yaml."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Undefined, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "CLIP.yaml").read_text())["covers"]
PROFILE = ("finite_value_single_gradient",)
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex gradients are outside the reviewed CLIP profile")
        raise


def _raw_leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _gradient(value, name="gradient"):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating CLIP profile")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (bool, np.bool_)):
                raise Invalid(f"{name} floating values cannot be Boolean literals")
            if isinstance(raw, str):
                spelling = raw.strip().lower()
                if spelling in {"nan", "inf", "+inf", "-inf"}:
                    raise Unsupported(f"{name} contains a special value outside the reviewed profile")
                try:
                    if spelling.startswith("-") and Fraction(spelling) == 0:
                        raise Unsupported("signed-zero distinctions are outside the reviewed CLIP profile")
                except ValueError:
                    pass
            if isinstance(raw, (float, np.floating)):
                if not math.isfinite(float(raw)):
                    raise Unsupported(f"{name} contains a special value outside the reviewed profile")
                if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                    raise Unsupported("signed-zero distinctions are outside the reviewed CLIP profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} must use f16, bf16, f32 or f64")
    if tensor.a.ndim not in (1, 2, 3, 4) or any(size == 0 for size in tensor.shape):
        raise Unsupported(f"{name} must be a positive rank-1 through rank-4 tensor")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be exact real numbers")
        if not item.finite:
            raise Unsupported(f"{name} contains a special value outside the reviewed profile")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _clip_value(value, dtype):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid("clip_value must be a real scalar")
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
        raise Unsupported("nonfinite clip_value is outside the reviewed CLIP profile")
    try:
        result = R(value)
    except Undefined:
        raise Unsupported("nonfinite clip_value is outside the reviewed CLIP profile")
    except (TypeError, ValueError):
        raise Invalid("clip_value must be a real scalar")
    if not result.exact:
        raise Invalid("clip_value must be exact")
    if not result.finite:
        raise Unsupported("nonfinite clip_value is outside the reviewed CLIP profile")
    if result.sign() <= 0:
        raise Unsupported("clip_value must be positive in the reviewed CLIP profile")
    stored = ieee.round_to(result, dtype)
    if not stored.exact or stored.v != result.v:
        raise Unsupported(f"clip_value is not exactly representable as {dtype}")
    return result


def _clip_grad_value(grad, clip_value, *, foreach=None, mode="correct"):
    if foreach is not None and type(foreach) is not bool:
        raise Invalid("foreach must be true, false or null")
    bound = _clip_value(clip_value, grad.dtype)
    if mode == "half_bound":
        bound = bound / 2
    elif mode == "double_bound":
        bound = bound * 2

    flat = list(grad.a.flat)
    out = []
    for index, value in enumerate(flat):
        if mode == "skip_last" and index == len(flat) - 1:
            result = value
        elif mode == "upper_only":
            result = bound if value > bound else value
        elif mode == "lower_only":
            result = -bound if value < -bound else value
        elif mode == "positive_lower" and value < -bound:
            result = bound
        else:
            result = -bound if value < -bound else bound if value > bound else value
        out.append(result)
    return A.Ten(np.asarray(out, dtype=object).reshape(grad.shape), grad.dtype)


def _decode_argument(op, name, value, arguments):
    if name == "grad":
        return _gradient(value)
    return decode(value)


def _correct(grad, clip_value, *, foreach=None):
    return _clip_grad_value(grad, clip_value, foreach=foreach, mode="correct")


OPS = {"torch.nn.utils.clip_grad_value_": _correct}
reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)


def _mutant(mode):
    def operation(grad, clip_value, *, foreach=None):
        return _clip_grad_value(grad, clip_value, foreach=foreach, mode=mode)
    return make_reference(COVERS, extra={"torch.nn.utils.clip_grad_value_": operation},
                          profiles=PROFILE, argument_decoder=_decode_argument)


MUTANTS = {
    "CLV.M1": _mutant("upper_only"),
    "CLV.M2": _mutant("lower_only"),
    "CLV.M3": _mutant("positive_lower"),
    "CLV.M4": _mutant("half_bound"),
    "CLV.M5": _mutant("double_bound"),
    "CLV.M6": _mutant("skip_last"),
}
READINGS = {}
