"""Reviewed finite out-of-place index_add values from contracts/INDEXREDUCE.yaml."""
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

COVERS = yaml.safe_load((CONTRACTS / "INDEXREDUCE.yaml").read_text())["covers"]
PROFILE = ("finite_index_add_values",)
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed INDEXREDUCE profile")
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
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating INDEXREDUCE profile")
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
                        raise Unsupported("signed-zero distinctions are outside the reviewed INDEXREDUCE profile")
                except ValueError:
                    pass
            if isinstance(raw, (float, np.floating)):
                if not math.isfinite(float(raw)):
                    raise Unsupported(f"{name} contains a special value outside the reviewed profile")
                if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                    raise Unsupported("signed-zero distinctions are outside the reviewed INDEXREDUCE profile")
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


def _index_tensor(value, name):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype != "i64":
        raise Unsupported(f"{name} must use int64 in the reviewed INDEXREDUCE profile")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
                raise Invalid(f"{name} values must be literal integers")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype != "i64":
        raise Unsupported(f"{name} must use int64 in the reviewed INDEXREDUCE profile")
    if tensor.a.ndim != 1 or tensor.a.size == 0:
        raise Invalid(f"{name} must be a nonempty one-dimensional tensor")
    return tensor


def _decode_argument(op, name, value, arguments):
    if name in {"self", "source"}:
        return _floating_tensor(value, name)
    if name == "index":
        return _index_tensor(value, name)
    return decode(value)


def _axis(dim, rank):
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise Invalid("index_add: dim must be one integer")
    dim = int(dim)
    if not -rank <= dim < rank:
        raise Invalid("index_add: dim is out of range")
    return dim % rank


def _alpha(value, dtype):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid("alpha must be a real scalar")
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
        raise Unsupported("nonfinite alpha is outside the reviewed INDEXREDUCE profile")
    try:
        result = R(value)
    except Undefined:
        raise Unsupported("nonfinite alpha is outside the reviewed INDEXREDUCE profile")
    except (TypeError, ValueError):
        raise Invalid("alpha must be a real scalar")
    if not result.exact:
        raise Invalid("alpha must be exact")
    if not result.finite:
        raise Unsupported("nonfinite alpha is outside the reviewed INDEXREDUCE profile")
    stored = ieee.round_to(result, dtype)
    if not stored.exact or stored.v != result.v:
        raise Unsupported(f"alpha is not exactly representable as {dtype}")
    return result


def _index_add(self, dim, index, source, *, alpha=1, mode="correct"):
    if self.dtype != source.dtype:
        raise Unsupported("self and source must use the same floating dtype")
    if len(self.shape) != len(source.shape):
        raise Invalid("self and source must have the same rank")
    axis = _axis(dim, len(self.shape))
    factor = R(1) if mode == "ignore_alpha" else _alpha(alpha, self.dtype)
    indices = [int(item) for item in index.a]
    if any(item < 0 or item >= self.shape[axis] for item in indices):
        raise Invalid("index_add: index is out of range")
    expected = list(self.shape)
    expected[axis] = len(indices)
    if source.shape != tuple(expected):
        raise Invalid("source shape must equal self except for index length along dim")

    out = self.a.copy()
    if mode == "dim_zero":
        axis = 0
        expected = list(self.shape)
        expected[axis] = len(indices)
        if source.shape != tuple(expected):
            raise Invalid("source shape does not match the misread dimension")

    contributions = list(enumerate(indices))
    if mode == "last_only":
        last = {target: position for position, target in contributions}
        contributions = [(position, target) for target, position in sorted(last.items(), key=lambda item: item[1])]

    if mode == "index_selects_source":
        for target, selected in enumerate(indices):
            if target >= self.shape[axis] or selected >= source.shape[axis]:
                raise Invalid("misread index selection is out of range")
            target_slice = [slice(None)] * out.ndim
            source_slice = [slice(None)] * source.a.ndim
            target_slice[axis] = target
            source_slice[axis] = selected
            old = out[tuple(target_slice)]
            add = source.a[tuple(source_slice)]
            out[tuple(target_slice)] = np.vectorize(lambda x, y: x + factor * y, otypes=[object])(old, add)
        return A.Ten(out, self.dtype)

    for position, target in contributions:
        if mode == "shift_index":
            target = (target + 1) % self.shape[axis]
        target_slice = [slice(None)] * out.ndim
        source_slice = [slice(None)] * source.a.ndim
        target_slice[axis] = target
        source_slice[axis] = position
        add = np.vectorize(lambda value: factor * value, otypes=[object])(source.a[tuple(source_slice)])
        if mode == "overwrite":
            out[tuple(target_slice)] = add
        else:
            old = out[tuple(target_slice)]
            out[tuple(target_slice)] = np.vectorize(lambda x, y: x + y, otypes=[object])(old, add)
    return A.Ten(out, self.dtype)


OPS = {"aten.index_add": _index_add, "torch.index_add": _index_add}
reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)


def _mutant(mode):
    def operation(self, dim, index, source, *, alpha=1):
        return _index_add(self, dim, index, source, alpha=alpha, mode=mode)
    return make_reference(COVERS, extra={"aten.index_add": operation, "torch.index_add": operation},
                          profiles=PROFILE, argument_decoder=_decode_argument)


MUTANTS = {
    "IR2.M1": _mutant("ignore_alpha"),
    "IR2.M2": _mutant("overwrite"),
    "IR2.M3": _mutant("index_selects_source"),
    "IR2.M4": _mutant("dim_zero"),
    "IR2.M5": _mutant("last_only"),
    "IR2.M6": _mutant("shift_index"),
}
READINGS = {}
