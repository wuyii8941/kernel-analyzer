"""Reviewed vLLM 0.16.0 deterministic top-k then top-p keep-mask core."""
from __future__ import annotations

import math

import numpy as np

from kernel_spec import d1_attention as A
from kernel_spec import ieee
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Unsupported, require_fields
from kernel_spec.real import R


PROFILE = "vllm016_topk_topp_keep_real"
REQUIRED = {"logits", "k", "p", "dtype"}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _physical(value, rank, name):
    for item in _walk(value):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name}: Boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported(f"{name}: special values are outside the reviewed filtering profile")
        if isinstance(item, str) and item.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported(f"{name}: special values are outside the reviewed filtering profile")
    tensor, shape = A.tensor(value, rank, name)
    for item in _walk(tensor):
        rounded = ieee.round_to(R(item), "f32")
        if not rounded.exact or rounded.v != item:
            raise Invalid(f"{name}: value {item!r} is not exactly representable as f32")
    return tensor, shape


def _integers(value, batch):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != batch:
        raise Invalid("k must be null or one integer per row")
    out = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise Invalid("k must contain only integers")
        out.append(int(item))
    return out


def _parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, REQUIRED, "SAMPLING")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"fields outside the selected deterministic filtering task: {sorted(extra)}")
    if inp["dtype"] != "f32":
        raise Unsupported("the reviewed vLLM filtering profile requires float32 logits and p")
    logits, shape = _physical(inp["logits"], 2, "logits")
    batch, vocabulary = shape
    if batch <= 0 or vocabulary <= 0:
        raise Invalid("logits must have positive shape [B,V]")
    for row, values in enumerate(logits):
        if len(set(values)) != vocabulary:
            raise Unsupported(f"row {row}: tied logits are outside the reviewed profile")
    k = _integers(inp["k"], batch)
    if k is not None and any(value < 1 or value > vocabulary for value in k):
        raise Invalid("each k must lie in 1..V")
    p = None
    if inp["p"] is not None:
        p, pshape = _physical(inp["p"], 1, "p")
        if pshape != (batch,):
            raise Invalid("p must be null or one float32 value per row")
        if any(value <= 0 or value > 1 for value in p):
            raise Invalid("each p must lie in (0,1]")
    if k is None and p is None:
        raise Unsupported("the identity call with no active filter is outside the selected task")
    return {"logits": logits, "B": batch, "V": vocabulary, "k": k, "p": p}


def _topk_order(row, k, mode):
    reverse = mode != "bottom_k"
    order = sorted(range(len(row)), key=lambda index: row[index], reverse=reverse)
    count = len(row) if k is None else min(len(row), k + (1 if mode == "topk_plus_one" else 0))
    return order[:count]


def _probabilities(row, active):
    maximum = max(row[index] for index in active)
    values = {index: rl.exp(R(row[index] - maximum)) for index in active}
    total = R(0)
    for value in values.values():
        total = total + value
    return {index: value / total for index, value in values.items()}


def _top_p(row, active, p, mode):
    if p is None or p == 1:
        return set(active)
    probability_indices = list(range(len(row))) if mode == "topp_full" else active
    probabilities = _probabilities(row, probability_indices)
    ordered = sorted(active, key=lambda index: row[index], reverse=True)
    kept, above = set(), R(0)
    for index in ordered:
        if mode == "cumulative_through":
            if above + probabilities[index] <= R(p):
                kept.add(index)
        elif above < R(p):
            kept.add(index)
        above = above + probabilities[index]
    return kept


def _row_mask(row, k, p, mode=None):
    if mode == "topp_before_topk":
        full = _top_p(row, list(range(len(row))), p, None)
        ordered = sorted(full, key=lambda index: row[index], reverse=True)
        active = ordered if k is None else ordered[:k]
        kept = set(active)
    else:
        active = _topk_order(row, k, mode)
        kept = _top_p(row, active, p, mode)
    return [index in kept for index in range(len(row))]


def _evaluate(profile, inp, mode=None):
    parsed = _parse(profile, inp)
    k_values = parsed["k"]
    p_values = parsed["p"]
    if mode == "row_parameters_swapped" and parsed["B"] > 1:
        if k_values is not None:
            k_values = list(reversed(k_values))
        if p_values is not None:
            p_values = list(reversed(p_values))
    keep = []
    for row_index, row in enumerate(parsed["logits"]):
        k = None if k_values is None else k_values[row_index]
        p = None if p_values is None else p_values[row_index]
        keep.append(_row_mask(row, k, p, mode))
    return Result("value", {"keep": keep})


def reference(profile, inp):
    return _evaluate(profile, inp)


def _observed(value, shape):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != "b8" or value.get("shape") != list(shape):
            raise Invalid("observed keep tensor record has the wrong dtype or shape")
        value = value["v"]
    if (not isinstance(value, (list, tuple)) or len(value) != shape[0]
            or any(not isinstance(row, (list, tuple)) or len(row) != shape[1] for row in value)
            or any(type(item) is not bool for row in value for item in row)):
        raise Invalid("observed keep must be a complete Boolean tensor")
    return [list(row) for row in value]


def check_output(profile, inp, observed):
    try:
        parsed = _parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"keep"}:
            return False
        candidate = _observed(observed["keep"], (parsed["B"], parsed["V"]))
    except Invalid:
        return False
    return candidate == reference(profile, inp).fields["keep"]


def _mutant(mode):
    return lambda profile, inp: _evaluate(profile, inp, mode)


MUTANTS = {
    "SP.M1": _mutant("topk_plus_one"),
    "SP.M2": _mutant("bottom_k"),
    "SP.M3": _mutant("topp_full"),
    "SP.M4": _mutant("topp_before_topk"),
    "SP.M5": _mutant("cumulative_through"),
    "SP.M6": _mutant("row_parameters_swapped"),
}
READINGS = {}
