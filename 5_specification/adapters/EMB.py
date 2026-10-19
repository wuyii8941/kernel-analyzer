"""Reviewed public PyTorch 2.10 dense embedding forward and weight VJP."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from kernel_spec import ieee
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational
from kernel_spec.real import R


PROFILE = "torch210_dense_real"
REQUIRED = {
    "weight", "indices", "upstream", "dtype", "index_dtype", "padding_idx",
    "max_norm", "norm_type", "scale_grad_by_freq", "sparse",
}
DTYPES = {"f16", "bf16", "f32", "f64"}


def _shape(value, what):
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    children = [_shape(item, what) for item in value]
    if any(child != children[0] for child in children):
        raise Invalid(f"{what}: ragged tensor")
    return (len(value),) + children[0]


def _walk(value):
    if isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _map(function, value):
    return [_map(function, item) for item in value] if isinstance(value, list) else function(value)


def _rebuild(values, shape):
    iterator = iter(values)

    def build(sizes):
        if len(sizes) == 1:
            return [next(iterator) for _ in range(sizes[0])]
        return [build(sizes[1:]) for _ in range(sizes[0])]

    return build(tuple(shape))


def _physical(value, shape, name, dtype):
    actual = _shape(value, name)
    if actual != tuple(shape):
        raise Invalid(f"{name}: shape {actual}, expected {tuple(shape)}")

    def convert(item):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name}: boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported(f"{name}: special floating values are outside the reviewed profile")
        if isinstance(item, str) and item.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported(f"{name}: special floating values are outside the reviewed profile")
        value = rational(item, name)
        rounded = ieee.round_to(R(value), dtype)
        if not rounded.exact or rounded.v != value:
            raise Invalid(f"{name}: {item!r} is not exactly representable as {dtype}")
        return value

    return _map(convert, value)


def _parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    if not isinstance(inp, dict) or any(not isinstance(key, str) for key in inp):
        raise Invalid("embedding input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"embedding input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"embedding fields outside the selected task: {sorted(extra)}")
    dtype = inp["dtype"]
    if dtype not in DTYPES:
        raise Unsupported("the reviewed embedding profile requires dtype f16, bf16, f32 or f64")
    if inp["index_dtype"] != "i64":
        raise Unsupported("the reviewed embedding profile requires int64 indices")
    if inp["max_norm"] is not None:
        raise Unsupported("max_norm renormalisation is outside the reviewed embedding profile")
    if isinstance(inp["norm_type"], (bool, np.bool_)):
        raise Invalid("norm_type must be a number")
    if rational(inp["norm_type"], "norm_type") != 2:
        raise Unsupported("nondefault norm_type is outside the reviewed embedding profile")
    frequency = inp["scale_grad_by_freq"]
    sparse = inp["sparse"]
    if type(frequency) is not bool or type(sparse) is not bool:
        raise Invalid("scale_grad_by_freq and sparse must be booleans")
    if sparse:
        raise Unsupported("sparse embedding gradients are outside the reviewed profile")

    weight_shape = _shape(inp["weight"], "weight")
    if len(weight_shape) != 2 or any(size <= 0 for size in weight_shape):
        raise Invalid("weight must be a nonempty two-dimensional tensor")
    weight = _physical(inp["weight"], weight_shape, "weight", dtype)
    index_shape = _shape(inp["indices"], "indices")
    if not index_shape or any(size <= 0 for size in index_shape):
        raise Unsupported("the reviewed embedding profile requires nonempty indices of positive rank")
    indices = []
    for item in _walk(inp["indices"]):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise Invalid("indices must contain literal integers")
        value = integer(item, "indices")
        if not 0 <= value < weight_shape[0]:
            raise Invalid(f"index {value} outside [0, {weight_shape[0]})")
        indices.append(value)

    upstream_shape = index_shape + (weight_shape[1],)
    upstream_tensor = _physical(inp["upstream"], upstream_shape, "upstream", dtype)
    upstream_values = list(_walk(upstream_tensor))
    upstream = [upstream_values[start:start + weight_shape[1]]
                for start in range(0, len(upstream_values), weight_shape[1])]
    padding = inp["padding_idx"]
    if padding is not None:
        if isinstance(padding, (bool, np.bool_)) or not isinstance(padding, (int, np.integer)):
            raise Invalid("padding_idx must be an integer or null")
        padding = integer(padding, "padding_idx")
        if not -weight_shape[0] <= padding < weight_shape[0]:
            raise Invalid("padding_idx must be within num_embeddings")
        if padding < 0:
            padding += weight_shape[0]
    return {
        "weight": weight, "indices": indices, "index_shape": index_shape,
        "upstream": upstream, "weight_shape": weight_shape,
        "padding": padding, "frequency": frequency, "dtype": dtype,
    }


def _evaluate(profile, inp, mode=None):
    parsed = _parse(profile, inp)
    weight = parsed["weight"]
    indices = parsed["indices"]
    rows, width = parsed["weight_shape"]
    padding = parsed["padding"]
    upstream = parsed["upstream"]
    forward_rows = []
    for index in indices:
        selected = (index + 1) % rows if mode == "following_row" else index
        if mode == "padding_forward_zero" and index == padding:
            forward_rows.append([Fraction(0)] * width)
        else:
            forward_rows.append(list(weight[selected]))
    out = _rebuild(forward_rows, parsed["index_shape"])
    if mode == "flatten_output" and len(parsed["index_shape"]) > 1:
        out = forward_rows

    dweight = [[Fraction(0)] * width for _ in range(rows)]
    counts = {index: indices.count(index) for index in set(indices)}
    for position, index in enumerate(indices):
        if index == padding and mode != "padding_gets_gradient":
            continue
        denominator = counts[index] if parsed["frequency"] else 1
        if mode == "frequency_plus_one" and parsed["frequency"]:
            denominator += 1
        if mode == "global_frequency" and parsed["frequency"]:
            denominator = len(indices)
        contribution = [value / denominator for value in upstream[position]]
        if mode == "last_gradient_only":
            dweight[index] = contribution
        else:
            dweight[index] = [old + value for old, value in zip(dweight[index], contribution)]
    return Result("value", {"out": out, "dweight": dweight})


def reference(profile, inp):
    return _evaluate(profile, inp)


def _observed(value, shape, dtype, name):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != dtype or value.get("shape") != list(shape):
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    if _shape(value, f"observed {name}") != tuple(shape):
        raise Invalid(f"observed {name} has the wrong shape")
    return _map(lambda item: rational(item, f"observed {name}"), value)


def check_output(profile, inp, observed):
    try:
        parsed = _parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"out", "dweight"}:
            return False
        output_shape = parsed["index_shape"] + (parsed["weight_shape"][1],)
        candidate = {
            "out": _observed(observed["out"], output_shape, parsed["dtype"], "out"),
            "dweight": _observed(observed["dweight"], parsed["weight_shape"], parsed["dtype"], "dweight"),
        }
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    return lambda profile, inp: _evaluate(profile, inp, mode)


MUTANTS = {
    "EM.M1": _mutant("padding_gets_gradient"),
    "EM.M2": _mutant("padding_forward_zero"),
    "EM.M3": _mutant("frequency_plus_one"),
    "EM.M4": _mutant("global_frequency"),
    "EM.M5": _mutant("flatten_output"),
    "EM.M6": _mutant("following_row"),
    "EM.M7": _mutant("last_gradient_only"),
}
READINGS = {}
