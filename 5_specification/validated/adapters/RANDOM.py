"""Reviewed deterministic dropout scopes from contracts/RANDOM.yaml."""
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

COVERS = yaml.safe_load((CONTRACTS / "RANDOM.yaml").read_text())["covers"]
FORWARD_COVERS = ["aten.dropout", "torch.nn.functional.dropout"]
BACKWARD_COVERS = ["aten.native_dropout_backward"]
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed RANDOM profiles")
        raise


def _raw_leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _floating_tensor(value, name):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating RANDOM profiles")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (bool, np.bool_)):
                raise Invalid(f"{name} floating values cannot be Boolean literals")
            if isinstance(raw, str):
                spelling = raw.strip().lower()
                if spelling in {"nan", "inf", "+inf", "-inf"}:
                    raise Unsupported(f"{name} contains a special value outside the reviewed profiles")
                try:
                    if spelling.startswith("-") and Fraction(spelling) == 0:
                        raise Unsupported("signed-zero distinctions are outside the reviewed RANDOM profiles")
                except ValueError:
                    pass
            if isinstance(raw, (float, np.floating)):
                if not math.isfinite(float(raw)):
                    raise Unsupported(f"{name} contains a special value outside the reviewed profiles")
                if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                    raise Unsupported("signed-zero distinctions are outside the reviewed RANDOM profiles")
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
            raise Unsupported(f"{name} contains a special value outside the reviewed profiles")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _boolean_mask(value, name):
    if not isinstance(value, dict) or value.get("t") not in {"b8", "bool", "torch.bool"}:
        raise Invalid(f"{name} must be a Boolean tensor")
    if "v" in value and any(not isinstance(raw, (bool, np.bool_)) for raw in _raw_leaves(value["v"])):
        raise Invalid(f"{name} must contain only Boolean values")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten) or tensor.dtype != "b8":
        raise Invalid(f"{name} must be a Boolean tensor")
    if tensor.a.ndim not in (1, 2, 3, 4) or any(size == 0 for size in tensor.shape):
        raise Unsupported(f"{name} must be a positive rank-1 through rank-4 tensor")
    return tensor


def _decode_argument(op, name, value, arguments):
    if name in {"input", "grad_output"}:
        return _floating_tensor(value, name)
    if name == "mask":
        return _boolean_mask(value, name)
    return decode(value)


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be boolean")
    return bool(value)


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be a real scalar")
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
        raise Unsupported(f"nonfinite {name} is outside the reviewed RANDOM profiles")
    try:
        result = R(value)
    except Undefined:
        raise Unsupported(f"nonfinite {name} is outside the reviewed RANDOM profiles")
    except (TypeError, ValueError):
        raise Invalid(f"{name} must be a real scalar")
    if not result.exact:
        raise Invalid(f"{name} must be an exact scalar")
    if not result.finite:
        raise Unsupported(f"nonfinite {name} is outside the reviewed RANDOM profiles")
    return result


def _dropout_values(input, p=0.5, training=True, inplace=False, *, mode="correct"):
    probability = _finite_scalar(p, "p")
    if probability.sign() < 0 or (probability - R(1)).sign() > 0:
        raise Invalid("dropout probability must lie in [0, 1]")
    training = _flag(training, "training")
    if _flag(inplace, "inplace"):
        raise Unsupported("in-place dropout is outside the reviewed RANDOM profile")
    if not training:
        if mode == "eval_scaled":
            factor = R(1) / (R(1) - probability)
            return A.Ten(np.vectorize(lambda value: value * factor, otypes=[object])(input.a), input.dtype)
        return input
    if probability.same(0):
        return input
    if probability.same(1):
        return input if mode == "p1_identity" else A.Ten(np.full(input.shape, R(0), dtype=object), input.dtype)
    raise Unsupported("training dropout with 0 < p < 1 needs a random-law test, not a unique value reference")


def _aten_dropout(input, p, train, *, mode="correct"):
    return _dropout_values(input, p, train, False, mode=mode)


def _functional_dropout(input, p=0.5, training=True, inplace=False, *, mode="correct"):
    return _dropout_values(input, p, training, inplace, mode=mode)


def _dropout_backward(grad_output, mask, scale, *, mode="correct"):
    if grad_output.shape != mask.shape:
        raise Invalid("mask must have the same shape as grad_output")
    scale = _finite_scalar(scale, "scale")
    out = np.empty(grad_output.shape, dtype=object)
    for index in np.ndindex(grad_output.shape):
        grad = grad_output.a[index]
        keep = bool(mask.a[index])
        if mode == "no_scale":
            value = grad if keep else R(0)
        elif mode == "invert_mask":
            value = grad * scale if not keep else R(0)
        elif mode == "add_scale":
            value = grad + scale if keep else R(0)
        else:
            value = grad * scale if keep else R(0)
        out[index] = value
    return A.Ten(out, grad_output.dtype)


FORWARD_OPS = {
    "aten.dropout": _aten_dropout,
    "torch.nn.functional.dropout": _functional_dropout,
}
BACKWARD_OPS = {"aten.native_dropout_backward": _dropout_backward}
_FORWARD_REFERENCE = make_reference(FORWARD_COVERS, extra=FORWARD_OPS,
                                    profiles=("deterministic_forward",), argument_decoder=_decode_argument)
_BACKWARD_REFERENCE = make_reference(BACKWARD_COVERS, extra=BACKWARD_OPS,
                                     profiles=("finite_backward",), argument_decoder=_decode_argument)
_FORWARD_CHECK = make_output_checker(FORWARD_COVERS, extra=FORWARD_OPS,
                                     profiles=("deterministic_forward",), argument_decoder=_decode_argument)
_BACKWARD_CHECK = make_output_checker(BACKWARD_COVERS, extra=BACKWARD_OPS,
                                      profiles=("finite_backward",), argument_decoder=_decode_argument)


def reference(profile, inp):
    if profile == "deterministic_forward":
        return _FORWARD_REFERENCE(profile, inp)
    if profile == "finite_backward":
        return _BACKWARD_REFERENCE(profile, inp)
    raise Unsupported(f"profile {profile!r} is not covered by this contract")


def check_output(profile, inp, observed):
    if profile == "deterministic_forward":
        return _FORWARD_CHECK(profile, inp, observed)
    if profile == "finite_backward":
        return _BACKWARD_CHECK(profile, inp, observed)
    raise Unsupported(f"profile {profile!r} is not covered by this contract")


def _forward_mutant(mode):
    def aten(input, p, train):
        return _aten_dropout(input, p, train, mode=mode)

    def functional(input, p=0.5, training=True, inplace=False):
        return _functional_dropout(input, p, training, inplace, mode=mode)

    return make_reference(FORWARD_COVERS,
                          extra={"aten.dropout": aten, "torch.nn.functional.dropout": functional},
                          profiles=("deterministic_forward",), argument_decoder=_decode_argument)


def _backward_mutant(mode):
    def operation(grad_output, mask, scale):
        return _dropout_backward(grad_output, mask, scale, mode=mode)
    return make_reference(BACKWARD_COVERS, extra={"aten.native_dropout_backward": operation},
                          profiles=("finite_backward",), argument_decoder=_decode_argument)


MUTANTS = {
    "RN2.M1": _forward_mutant("eval_scaled"),
    "RN2.M2": _forward_mutant("p1_identity"),
    "RN2.M3": _backward_mutant("no_scale"),
    "RN2.M4": _backward_mutant("invert_mask"),
    "RN2.M5": _backward_mutant("add_scale"),
}
READINGS = {}
