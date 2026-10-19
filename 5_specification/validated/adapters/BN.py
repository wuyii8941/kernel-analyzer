"""Adapter for reviewed evaluation-mode BatchNorm values in contracts/BN.yaml."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec import real as rl
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Undefined, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "BN.yaml").read_text())["covers"]
PROFILE = ("finite_eval_values",)
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}
TENSOR_ARGUMENTS = {"input", "weight", "bias", "running_mean", "running_var"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed BN profile")
        raise


def _raw_leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _source_tensor(value, name):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating BN profile")
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
                        raise Unsupported("signed-zero distinctions are outside the reviewed BN profile")
                except ValueError:
                    pass
            if isinstance(raw, (float, np.floating)):
                if not math.isfinite(float(raw)):
                    raise Unsupported(f"{name} contains a special value outside the reviewed profile")
                if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                    raise Unsupported("signed-zero distinctions are outside the reviewed BN profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} must use f16, bf16, f32 or f64")
    if any(size == 0 for size in tensor.shape):
        raise Unsupported(f"{name} must have positive dimensions")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be exact real numbers")
        if not item.finite:
            raise Unsupported(f"{name} contains a special value outside the reviewed profile")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _decode_argument(op, name, value, arguments):
    if name in TENSOR_ARGUMENTS and value is not None:
        return _source_tensor(value, name)
    return decode(value)


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be boolean")
    return bool(value)


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be a real scalar")
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
        raise Unsupported(f"nonfinite {name} is outside the reviewed BN profile")
    try:
        result = R(value)
    except Undefined:
        raise Unsupported(f"nonfinite {name} is outside the reviewed BN profile")
    except (TypeError, ValueError):
        raise Invalid(f"{name} must be a real scalar")
    if not result.exact:
        raise Invalid(f"{name} must be an exact scalar")
    if not result.finite:
        raise Unsupported(f"nonfinite {name} is outside the reviewed BN profile")
    return result


def _vector(value, name, source, channels, optional=False):
    if value is None:
        if optional:
            return None
        raise Unsupported(f"{name} is required in evaluation mode")
    if not isinstance(value, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if value.shape != (channels,):
        raise Invalid(f"{name} must have shape [{channels}]")
    if value.dtype != source.dtype:
        raise Unsupported("input, running statistics and affine tensors must use the same dtype")
    return value


def _batch_norm_values(input, running_mean, running_var, weight=None, bias=None,
                       training=False, momentum=0.1, eps=1e-5, *, mode="correct"):
    if not isinstance(input, A.Ten):
        raise Invalid("input must be a tensor")
    if input.a.ndim not in (2, 3, 4):
        raise Unsupported("BN requires a rank-2, rank-3 or rank-4 input")
    if _flag(training, "training"):
        raise Unsupported("training mode is outside the reviewed BN profile")
    _finite_scalar(momentum, "momentum")
    epsilon = _finite_scalar(eps, "eps")
    if epsilon.sign() <= 0:
        raise Unsupported("eps must be positive in the reviewed BN profile")

    channels = input.shape[1]
    running_mean = _vector(running_mean, "running_mean", input, channels)
    running_var = _vector(running_var, "running_var", input, channels)
    weight = _vector(weight, "weight", input, channels, optional=True)
    bias = _vector(bias, "bias", input, channels, optional=True)
    for value in running_var.a:
        if value.sign() < 0:
            raise Unsupported("running variance must be nonnegative")

    batch_mean = batch_var = None
    if mode == "batch_stats":
        batch_mean, batch_var = [], []
        for channel in range(channels):
            values = [input.a[index] for index in np.ndindex(input.shape) if index[1] == channel]
            mean = sum(values, R(0)) / len(values)
            variance = sum(((value - mean) * (value - mean) for value in values), R(0)) / len(values)
            batch_mean.append(mean)
            batch_var.append(variance)

    out = np.empty(input.shape, dtype=object)
    for index in np.ndindex(input.shape):
        channel = 0 if mode == "channel_zero" else index[1]
        mean = batch_mean[index[1]] if mode == "batch_stats" else running_mean.a[channel]
        variance = batch_var[index[1]] if mode == "batch_stats" else running_var.a[channel]
        if mode == "omit_eps":
            denominator = rl.sqrt(variance)
        elif mode == "eps_outside":
            denominator = rl.sqrt(variance) + epsilon
        else:
            denominator = rl.sqrt(variance + epsilon)
        value = (input.a[index] - mean) / denominator
        if weight is not None and mode != "omit_weight":
            value = value * weight.a[channel]
        if bias is not None and mode != "omit_bias":
            value = value + bias.a[channel]
        out[index] = value
    return A.Ten(out, input.dtype)


def _functional_batch_norm(input, running_mean, running_var, weight=None, bias=None,
                           training=False, momentum=0.1, eps=1e-5):
    return _batch_norm_values(input, running_mean, running_var, weight, bias, training, momentum, eps)


def _aten_batch_norm(input, weight, bias, running_mean, running_var, training,
                     momentum, eps, cudnn_enabled):
    _flag(cudnn_enabled, "cudnn_enabled")
    return _batch_norm_values(input, running_mean, running_var, weight, bias, training, momentum, eps)


OPS = {
    "torch.nn.functional.batch_norm": _functional_batch_norm,
    "aten.batch_norm": _aten_batch_norm,
}
reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)


def _mutant_functional(mode):
    def operation(input, running_mean, running_var, weight=None, bias=None,
                  training=False, momentum=0.1, eps=1e-5):
        return _batch_norm_values(input, running_mean, running_var, weight, bias,
                                  training, momentum, eps, mode=mode)
    return make_reference(COVERS, extra={**OPS, "torch.nn.functional.batch_norm": operation},
                          profiles=PROFILE, argument_decoder=_decode_argument)


MUTANTS = {
    "BN2.M1": _mutant_functional("batch_stats"),
    "BN2.M2": _mutant_functional("omit_eps"),
    "BN2.M3": _mutant_functional("eps_outside"),
    "BN2.M4": _mutant_functional("omit_weight"),
    "BN2.M5": _mutant_functional("omit_bias"),
    "BN2.M6": _mutant_functional("channel_zero"),
}
READINGS = {}
