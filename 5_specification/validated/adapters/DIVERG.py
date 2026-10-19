"""Reviewed PyTorch 2.10 probability-target KL batchmean task and complete input VJP."""
from __future__ import annotations

import math

import numpy as np

from kernel_spec import d1_attention as A
from kernel_spec import ieee
from kernel_spec import real as rl
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, rational, require_fields
from kernel_spec.real import R


PROFILE = "torch210_probability_batchmean_f32"
REQUIRED = {"input", "target", "input_dtype", "target_dtype", "reduction", "log_target", "upstream"}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _reject_specials(value, name):
    for item in _walk(value):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name}: Boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported(f"{name}: special values are outside the reviewed KL profile")
        if isinstance(item, str) and item.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported(f"{name}: special values are outside the reviewed KL profile")


def _physical_scalar(value, name):
    _reject_specials(value, name)
    if isinstance(value, (list, tuple, dict)):
        raise Invalid(f"{name} must be a scalar")
    value = rational(value, name)
    rounded = ieee.round_to(R(value), "f32")
    if not rounded.exact or rounded.v != value:
        raise Invalid(f"{name}: value {value!r} is not exactly representable as float32")
    return value


def _physical_matrix(value, name):
    _reject_specials(value, name)
    tensor, shape = A.tensor(value, 2, name)
    if min(shape) <= 0:
        raise Invalid(f"{name} must have positive rank-2 shape")
    for item in _walk(tensor):
        rounded = ieee.round_to(R(item), "f32")
        if not rounded.exact or rounded.v != item:
            raise Invalid(f"{name}: value {item!r} is not exactly representable as float32")
    return tensor, shape


def _parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, REQUIRED, "DIVERG")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"fields outside the selected PyTorch KL task: {sorted(extra)}")
    if inp["input_dtype"] != "f32" or inp["target_dtype"] != "f32":
        raise Unsupported("the reviewed KL profile requires float32 input and target")
    if inp["reduction"] != "batchmean":
        raise Unsupported("the reviewed KL profile fixes reduction=batchmean")
    if type(inp["log_target"]) is not bool:
        raise Invalid("log_target must be a Boolean")
    if inp["log_target"]:
        raise Unsupported("log_target=true is outside the selected probability-target task")
    values, shape = _physical_matrix(inp["input"], "input")
    target, target_shape = _physical_matrix(inp["target"], "target")
    if target_shape != shape:
        raise Invalid(f"input and target shapes differ: {shape} versus {target_shape}")
    if any(item <= 0 for item in _walk(target)):
        raise Unsupported("zero or negative target is outside the reviewed positive-target profile")
    upstream = _physical_scalar(inp["upstream"], "upstream")
    return {"input": values, "target": target, "shape": shape, "upstream": upstream}


def _plain(value):
    return value.v if isinstance(value, R) else value


def _evaluate(profile, inp, mode=None):
    parsed = _parse(profile, inp)
    batch, support = parsed["shape"]
    targets = parsed["target"]
    if mode == "renormalise_target":
        targets = [[item / sum(row) for item in row] for row in targets]
    divisor = batch * support if mode == "mean" else batch
    loss = R(0)
    grad = []
    for input_row, target_row in zip(parsed["input"], targets):
        grad_row = []
        for value, target in zip(input_row, target_row):
            if mode == "log_target_true":
                weight = rl.exp(R(target))
                loss = loss + weight * R(target - value)
                derivative = -weight
            else:
                term = -R(target * value) if mode == "omit_target_log" else R(target) * (rl.log(R(target)) - R(value))
                loss = loss + term
                derivative = R(target) if mode == "grad_sign" else R(-target)
            grad_divisor = batch * support if mode in {"mean", "grad_mean"} else batch
            grad_row.append(_plain(derivative * R(parsed["upstream"]) / grad_divisor))
        grad.append(grad_row)
    return Result("value", {"loss": _plain(loss / divisor), "grad": grad})


def reference(profile, inp):
    return _evaluate(profile, inp)


def _observed_scalar(value, name):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != "f32" or value.get("shape") != []:
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    if isinstance(value, (list, tuple, dict)):
        raise Invalid(f"observed {name} must be scalar")
    return rational(value, f"observed {name}")


def _observed_matrix(value, shape, name):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != "f32" or value.get("shape") != list(shape):
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    tensor, actual = A.tensor(value, 2, f"observed {name}")
    if actual != shape:
        raise Invalid(f"observed {name} shape {actual}, expected {shape}")
    return tensor


def check_output(profile, inp, observed):
    try:
        parsed = _parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"loss", "grad"}:
            return False
        candidate = {
            "loss": _observed_scalar(observed["loss"], "loss"),
            "grad": _observed_matrix(observed["grad"], parsed["shape"], "grad"),
        }
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    return lambda profile, inp: _evaluate(profile, inp, mode)


MUTANTS = {
    "DV.M1": _mutant("mean"),
    "DV.M2": _mutant("omit_target_log"),
    "DV.M3": _mutant("grad_sign"),
    "DV.M4": _mutant("grad_mean"),
    "DV.M5": _mutant("renormalise_target"),
    "DV.M6": _mutant("log_target_true"),
}
READINGS = {}
