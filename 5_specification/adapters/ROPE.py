"""Reviewed Transformers 4.57.3 Llama apply_rotary_pos_emb value and q/k VJP."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from kernel_spec import ieee
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational
from kernel_spec.real import R


PROFILE = "hf4573_llama_tables_real"
REQUIRED = {
    "q", "k", "cos", "sin", "upstream_q", "upstream_k", "dtype",
    "position_ids", "unsqueeze_dim",
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


def _map(function, value):
    return [_map(function, item) for item in value] if isinstance(value, list) else function(value)


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
        raise Invalid("RoPE input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"RoPE input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"RoPE fields outside the selected task: {sorted(extra)}")
    dtype = inp["dtype"]
    if dtype not in DTYPES:
        raise Unsupported("the reviewed RoPE profile requires dtype f16, bf16, f32 or f64")
    if inp["position_ids"] is not None:
        raise Unsupported("position_ids is deprecated and fixed to null in the reviewed Llama task")
    unsqueeze = inp["unsqueeze_dim"]
    if isinstance(unsqueeze, (bool, np.bool_)) or not isinstance(unsqueeze, (int, np.integer)):
        raise Invalid("unsqueeze_dim must be an integer")
    if integer(unsqueeze, "unsqueeze_dim") != 1:
        raise Unsupported("the reviewed profile selects [B,H,L,D] tensors and unsqueeze_dim=1")

    q_shape = _shape(inp["q"], "q")
    k_shape = _shape(inp["k"], "k")
    if len(q_shape) != 4 or len(k_shape) != 4 or any(size <= 0 for size in q_shape + k_shape):
        raise Invalid("q and k must be nonempty [B,H,L,D] tensors")
    if q_shape[0] != k_shape[0] or q_shape[2:] != k_shape[2:]:
        raise Invalid("q and k must have the same batch, sequence length and head dimension")
    batch, _, length, width = q_shape
    if width % 2:
        raise Invalid("the Llama rotate-half formula requires an even head dimension")
    table_shape = (batch, length, width)
    tensors = {
        "q": _physical(inp["q"], q_shape, "q", dtype),
        "k": _physical(inp["k"], k_shape, "k", dtype),
        "cos": _physical(inp["cos"], table_shape, "cos", dtype),
        "sin": _physical(inp["sin"], table_shape, "sin", dtype),
        "upstream_q": _physical(inp["upstream_q"], q_shape, "upstream_q", dtype),
        "upstream_k": _physical(inp["upstream_k"], k_shape, "upstream_k", dtype),
    }
    return {**tensors, "q_shape": q_shape, "k_shape": k_shape, "table_shape": table_shape, "dtype": dtype}


def _apply(x, upstream, cos, sin, mode=None, key=False):
    if key and mode == "key_passthrough":
        return x, upstream
    batch, heads, length, width = _shape(x, "tensor")
    half = width // 2
    out = [[[[Fraction(0) for _ in range(width)] for _ in range(length)]
            for _ in range(heads)] for _ in range(batch)]
    grad = [[[[Fraction(0) for _ in range(width)] for _ in range(length)]
             for _ in range(heads)] for _ in range(batch)]
    pairs = [(i, i + half) for i in range(half)]
    if mode == "interleaved":
        pairs = [(2 * i, 2 * i + 1) for i in range(half)]
    for b in range(batch):
        table_batch = 0 if mode == "batch_zero_table" else b
        for h in range(heads):
            for t in range(length):
                for p, s in pairs:
                    cp, cs = cos[table_batch][t][p], cos[table_batch][t][s]
                    sp, ss = sin[table_batch][t][p], sin[table_batch][t][s]
                    if mode == "duplicate_shortcut":
                        cs, ss = cp, sp
                    xp, xs = x[b][h][t][p], x[b][h][t][s]
                    gp, gs = upstream[b][h][t][p], upstream[b][h][t][s]
                    sign = -1 if mode == "opposite_sine" else 1
                    out[b][h][t][p] = xp * cp - sign * xs * sp
                    out[b][h][t][s] = xs * cs + sign * xp * ss
                    if mode == "backward_forward":
                        grad[b][h][t][p] = gp * cp - gs * sp
                        grad[b][h][t][s] = gs * cs + gp * ss
                    else:
                        grad[b][h][t][p] = gp * cp + gs * ss
                        grad[b][h][t][s] = -gp * sp + gs * cs
    return out, grad


def _evaluate(profile, inp, mode=None):
    parsed = _parse(profile, inp)
    q_out, dq = _apply(parsed["q"], parsed["upstream_q"], parsed["cos"], parsed["sin"], mode)
    k_out, dk = _apply(parsed["k"], parsed["upstream_k"], parsed["cos"], parsed["sin"], mode, key=True)
    return Result("value", {"q_out": q_out, "k_out": k_out, "dq": dq, "dk": dk})


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
        shapes = {"q_out": parsed["q_shape"], "k_out": parsed["k_shape"],
                  "dq": parsed["q_shape"], "dk": parsed["k_shape"]}
        if not isinstance(observed, dict) or set(observed) != set(shapes):
            return False
        candidate = {name: _observed(observed[name], shape, parsed["dtype"], name)
                     for name, shape in shapes.items()}
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    return lambda profile, inp: _evaluate(profile, inp, mode)


MUTANTS = {
    "RO.M1": _mutant("interleaved"),
    "RO.M2": _mutant("duplicate_shortcut"),
    "RO.M3": _mutant("backward_forward"),
    "RO.M4": _mutant("opposite_sine"),
    "RO.M5": _mutant("key_passthrough"),
    "RO.M6": _mutant("batch_zero_table"),
}
READINGS = {}
