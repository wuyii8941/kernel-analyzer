"""Adapter for the reviewed finite unique-row sorting scope in contracts/SORT.yaml."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee, sorting
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "SORT.yaml").read_text())["covers"]
PROFILE = ("finite_unique",)
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed SORT profile")
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
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating SORT profile")
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
                        raise Unsupported("signed-zero distinctions are outside the reviewed SORT profile")
                except ValueError:
                    pass  # decode reports malformed numeric strings as invalid input.
            if isinstance(raw, (float, np.floating)):
                if not math.isfinite(float(raw)):
                    raise Unsupported(f"{name} contains a special value outside the reviewed profile")
                if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                    raise Unsupported("signed-zero distinctions are outside the reviewed SORT profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} must use f16, bf16, f32 or f64")
    if tensor.a.ndim == 0:
        raise Unsupported("rank-zero sorting is outside the reviewed SORT profile")
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
    return _source_tensor(value, name) if name == "a" else decode(value)


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be boolean")
    return bool(value)


def _axis_and_unique(a, dim):
    d = sorting._dimension(a, dim)
    moved = np.moveaxis(a.a, d, -1)
    for row in np.ndindex(moved.shape[:-1]):
        keys = [value.v for value in moved[row]]
        if len(set(keys)) != len(keys):
            raise Unsupported("equal values in one selected row are outside the finite_unique profile")
    return d


def _sort(a, dim=-1, descending=False, stable=False):
    descending = _flag(descending, "sort descending")
    stable = _flag(stable, "sort stable")
    _axis_and_unique(a, dim)
    return sorting.sort(a, dim, descending, stable)


def _argsort(a, dim=-1, descending=False, stable=False):
    descending = _flag(descending, "argsort descending")
    stable = _flag(stable, "argsort stable")
    _axis_and_unique(a, dim)
    return sorting.argsort(a, dim, descending, stable)


def _topk(a, k, dim=-1, largest=True, sorted=True):
    largest = _flag(largest, "topk largest")
    ordered = _flag(sorted, "topk sorted")
    if not ordered:
        raise Unsupported("topk sorted=false is outside the reviewed SORT profile")
    _axis_and_unique(a, dim)
    return sorting.topk(a, k, dim, largest, True)


OPS = {"aten.sort": _sort, "aten.argsort": _argsort, "aten.topk": _topk}
reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)


def _mut(**replacements):
    return make_reference(COVERS, extra={**OPS, **replacements}, profiles=PROFILE,
                          argument_decoder=_decode_argument)


def _sort_descending_ignored(a, dim=-1, descending=False, stable=False):
    return _sort(a, dim, False, stable)


def _sort_last_dimension(a, dim=-1, descending=False, stable=False):
    return _sort(a, -1, descending, stable)


def _argsort_reversed(a, dim=-1, descending=False, stable=False):
    return _argsort(a, dim, not _flag(descending, "argsort descending"), stable)


def _argsort_descending_ignored(a, dim=-1, descending=False, stable=False):
    return _argsort(a, dim, False, stable)


def _topk_reversed(a, k, dim=-1, largest=True, sorted=True):
    values, indices = _topk(a, k, dim, largest, sorted)
    d = sorting._dimension(a, dim)
    return [A.Ten(np.flip(values.a, axis=d).copy(), values.dtype),
            A.Ten(np.flip(indices.a, axis=d).copy(), indices.dtype)]


def _topk_largest_ignored(a, k, dim=-1, largest=True, sorted=True):
    return _topk(a, k, dim, True, sorted)


MUTANTS = {
    "SO.M1": _mut(**{"aten.sort": _sort_descending_ignored}),
    "SO.M2": _mut(**{"aten.sort": _sort_last_dimension}),
    "SO.M3": _mut(**{"aten.argsort": _argsort_reversed}),
    "SO.M4": _mut(**{"aten.argsort": _argsort_descending_ignored}),
    "SO.M5": _mut(**{"aten.topk": _topk_reversed}),
    "SO.M6": _mut(**{"aten.topk": _topk_largest_ignored}),
}
READINGS = {}
