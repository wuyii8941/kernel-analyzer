"""Adapter for the reviewed read-only lookup scope in contracts/INDEX.yaml."""
from __future__ import annotations

import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported, rational
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "INDEX.yaml").read_text())["covers"]
PROFILE = ("real",)
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed INDEX profile")
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
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating INDEX profile")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (float, np.floating)) and raw == 0 and math.copysign(1.0, float(raw)) < 0:
                raise Invalid(f"{name} contains negative zero outside the logical-value profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} must use f16, bf16, f32 or f64")
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
        raise Unsupported(f"{name} must use int64 in the reviewed INDEX profile")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
                raise Invalid(f"{name} values must be literal integers")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype != "i64":
        raise Unsupported(f"{name} must use int64 in the reviewed INDEX profile")
    return tensor


def _decode_argument(op, name, value, arguments):
    if op == "torch.nn.functional.embedding" and name == "input":
        return _index_tensor(value, name)
    if name in ("a", "weight"):
        return _source_tensor(value, name)
    if name in ("idx", "indices"):
        return _index_tensor(value, name)
    return decode(value)


def _axis(dim, rank, op):
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise Invalid(f"{op}: dim must be one integer")
    dim = int(dim)
    if not -rank <= dim < rank:
        raise Invalid(f"{op}: dim is out of range")
    return dim % rank


def _gather(a, dim, idx, *, sparse_grad=False):
    if not isinstance(sparse_grad, (bool, np.bool_)):
        raise Invalid("gather: sparse_grad must be boolean")
    if sparse_grad:
        raise Unsupported("sparse gather gradients are outside this forward profile")
    if not isinstance(a, A.Ten) or not isinstance(idx, A.Ten):
        raise Invalid("gather requires tensor input and index")
    if len(a.shape) == 0:
        raise Unsupported("scalar gather is outside the reviewed profile")
    dim = _axis(dim, len(a.shape), "gather")
    if len(idx.shape) != len(a.shape):
        raise Invalid("gather: input and index must have the same rank")
    if idx.a.size and any(idx.shape[d] > a.shape[d] for d in range(len(a.shape)) if d != dim):
        raise Invalid("gather: index shape exceeds the input outside dim")
    return A.gather(a, dim, idx)


def _index_select(a, dim, idx):
    if not isinstance(a, A.Ten) or not isinstance(idx, A.Ten):
        raise Invalid("index_select requires tensor input and index")
    if len(a.shape) == 0:
        raise Unsupported("scalar index_select is outside the reviewed profile")
    dim = _axis(dim, len(a.shape), "index_select")
    if len(idx.shape) != 1:
        raise Invalid("index_select: index must be one-dimensional")
    return A.index_select(a, dim, idx)


def _embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    if isinstance(padding_idx, (bool, np.bool_)) or not isinstance(padding_idx, (int, np.integer)):
        raise Invalid("embedding: padding_idx must be an integer")
    if int(padding_idx) != -1:
        raise Unsupported("nondefault embedding padding_idx is outside this forward profile")
    if not isinstance(scale_grad_by_freq, (bool, np.bool_)) or not isinstance(sparse, (bool, np.bool_)):
        raise Invalid("embedding: scale_grad_by_freq and sparse must be booleans")
    if scale_grad_by_freq or sparse:
        raise Unsupported("nondefault embedding gradient or sparse options are outside this forward profile")
    if not isinstance(weight, A.Ten) or not isinstance(indices, A.Ten):
        raise Invalid("embedding requires weight and indices tensors")
    if len(weight.shape) != 2:
        raise Invalid("embedding: weight must be two-dimensional")
    return A.embedding(weight, indices)


def _functional_embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                          scale_grad_by_freq=False, sparse=False):
    # Public F.embedding is not an alias of the native schema: indices come
    # first, and None is translated to the native no-padding sentinel -1.
    # Renormalisation writes weight; it cannot be silently treated as lookup.
    if max_norm is not None:
        raise Unsupported("max_norm mutation is outside the read-only INDEX profile")
    if padding_idx is not None:
        raise Unsupported("nondefault public padding_idx is outside the INDEX profile; use the EMB task")
    if rational(norm_type, "norm_type") != 2:
        raise Unsupported("nondefault norm_type is outside the INDEX profile")
    return _embedding(weight, input, -1, scale_grad_by_freq, sparse)


OPS = {
    "aten.gather": _gather, "torch.gather": _gather,
    "aten.index_select": _index_select, "torch.index_select": _index_select,
    "aten.embedding": _embedding, "torch.nn.functional.embedding": _functional_embedding,
}

# PyTorch 2.10 native_functions.yaml uses self/index; the public Python docs
# use input/index. Keep the existing DSL a/idx spellings for compatibility.
ARGUMENT_ALIASES = {
    "aten.gather": {"self": "a", "index": "idx"},
    "aten.index_select": {"self": "a", "index": "idx"},
    "torch.gather": {"input": "a", "index": "idx"},
    "torch.index_select": {"input": "a", "index": "idx"},
}

reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument,
                           argument_aliases=ARGUMENT_ALIASES)
check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument,
                                  argument_aliases=ARGUMENT_ALIASES)


def _mut(**replacements):
    return make_reference(COVERS, extra={**OPS, **replacements}, profiles=PROFILE,
                          argument_decoder=_decode_argument, argument_aliases=ARGUMENT_ALIASES)


def _gather_next_axis(a, dim, idx, *, sparse_grad=False):
    dim = _axis(dim, len(a.shape), "gather")
    return A.gather(a, (dim + 1) % len(a.shape), idx)


def _gather_shifted(a, dim, idx, *, sparse_grad=False):
    dim = _axis(dim, len(a.shape), "gather")
    n = a.shape[dim]
    shifted = np.vectorize(lambda value: (int(value) + 1) % n, otypes=[object])(idx.a)
    return A.gather(a, dim, A.Ten(shifted, "i64"))


def _index_select_unique(a, dim, idx):
    values = []
    for value in idx.a.reshape(-1):
        if int(value) not in values:
            values.append(int(value))
    return A.index_select(a, _axis(dim, len(a.shape), "index_select"), A.Ten(np.asarray(values, dtype=object), "i64"))


def _index_select_reversed(a, dim, idx):
    return A.index_select(a, _axis(dim, len(a.shape), "index_select"),
                          A.Ten(idx.a.reshape(-1)[::-1].copy(), "i64"))


def _embedding_following(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    shifted = np.vectorize(lambda value: (int(value) + 1) % weight.shape[0], otypes=[object])(indices.a)
    return A.embedding(weight, A.Ten(shifted, "i64"))


def _embedding_flat(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return A.embedding(weight, A.Ten(indices.a.reshape(-1), "i64"))


MUTANTS = {
    "IX.M1": _mut(**{"aten.gather": _gather_next_axis}),
    "IX.M2": _mut(**{"aten.gather": _gather_shifted}),
    "IX.M3": _mut(**{"aten.index_select": _index_select_unique}),
    "IX.M4": _mut(**{"aten.index_select": _index_select_reversed}),
    "IX.M5": _mut(**{"aten.embedding": _embedding_following}),
    "IX.M6": _mut(**{"aten.embedding": _embedding_flat}),
}

READINGS = {}
