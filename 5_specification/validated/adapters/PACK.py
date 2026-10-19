"""Reviewed FlashAttention v2.8.3 variable-length forward reference."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from kernel_spec import d1_attention as A
from kernel_spec import d1_indep as D
from kernel_spec import d2_common as U
from kernel_spec import ieee
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational
from kernel_spec.real import R


PROFILES = {
    "flashattn283_varlen_causal_real": True,
    "flashattn283_varlen_bidirectional_real": False,
}
KEYS = {
    "q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k",
    "dropout_p", "softmax_scale", "causal", "window_size", "softcap", "alibi_slopes",
    "deterministic", "return_attn_probs", "block_table", "dtype",
}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _physical(value, rank, name, dtype):
    for item in _walk(value):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name}: Boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported(f"{name}: special values are outside the reviewed profile")
        if isinstance(item, str) and item.strip().lower() in {
                "inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported(f"{name}: special values are outside the reviewed profile")
    tensor, shape = A.tensor(value, rank, name)
    for item in _walk(tensor):
        rounded = ieee.round_to(item, dtype)
        if not rounded.exact or rounded.v != item:
            raise Invalid(f"{name}: value {item!r} is not exactly representable as {dtype}")
    return tensor, shape


def _i32_vector(value, name):
    tensor, shape = U.ints(value, 1, name)
    if any(not -(1 << 31) <= item < (1 << 31) for item in tensor):
        raise Invalid(f"{name}: value is outside signed int32")
    return tensor, shape


def _bool(value, name):
    if type(value) is not bool:
        raise Invalid(f"{name} must be a Boolean")
    return value


def parse(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS, required=KEYS)
    dtype = inp["dtype"]
    if dtype not in ("f16", "bf16"):
        raise Unsupported("the reviewed CUDA profile supports physical f16 or bf16 q/k/v/out")

    q, q_shape = _physical(inp["q"], 3, "q", dtype)
    k, k_shape = _physical(inp["k"], 3, "k", dtype)
    v, v_shape = _physical(inp["v"], 3, "v", dtype)
    total_q, hq, dim = q_shape
    total_k, hkv, k_dim = k_shape
    if min(total_q, total_k, hq, hkv, dim, k_dim) <= 0:
        raise Invalid("q, k and v dimensions must be positive")
    if v_shape != k_shape:
        raise Invalid("k and v must have the same packed shape")
    if k_dim != dim:
        raise Invalid("q and k head dimensions must match")
    if hq % hkv:
        raise Invalid("query heads must be a multiple of key/value heads")

    cu_q, cu_q_shape = _i32_vector(inp["cu_seqlens_q"], "cu_seqlens_q")
    cu_k, cu_k_shape = _i32_vector(inp["cu_seqlens_k"], "cu_seqlens_k")
    if cu_q_shape != cu_k_shape or cu_q_shape[0] < 2:
        raise Invalid("q and k cumulative lengths must describe the same nonempty batch")
    if cu_q[0] != 0 or cu_q[-1] != total_q or any(b <= a for a, b in zip(cu_q, cu_q[1:])):
        raise Invalid("cu_seqlens_q must start at 0, increase strictly and end at total_q")
    if cu_k[0] != 0 or cu_k[-1] != total_k or any(b <= a for a, b in zip(cu_k, cu_k[1:])):
        raise Invalid("cu_seqlens_k must start at 0, increase strictly and end at total_k")

    q_lens = [b - a for a, b in zip(cu_q, cu_q[1:])]
    k_lens = [b - a for a, b in zip(cu_k, cu_k[1:])]
    max_q = integer(inp["max_seqlen_q"], "max_seqlen_q", lo=1)
    max_k = integer(inp["max_seqlen_k"], "max_seqlen_k", lo=1)
    if max_q != max(q_lens) or max_k != max(k_lens):
        raise Invalid("max_seqlen_q and max_seqlen_k must equal the represented maxima")

    causal = _bool(inp["causal"], "causal")
    if causal != PROFILES[profile]:
        raise Unsupported("causal selects the other reviewed FlashAttention profile")
    dropout = rational(inp["dropout_p"], "dropout_p")
    if not 0 <= dropout <= 1:
        raise Invalid("dropout_p must lie in [0, 1]")
    if dropout:
        raise Unsupported("dropout is random and outside the reviewed forward profile")
    left, right = A.window_pair(inp["window_size"], "window_size")
    if (left, right) != (-1, -1):
        raise Unsupported("local attention is outside the reviewed varlen profile")
    softcap = _physical([inp["softcap"]], 1, "softcap", "f32")[0][0]
    if softcap < 0:
        raise Invalid("softcap must be nonnegative")
    if softcap:
        raise Unsupported("softcap is outside the reviewed varlen profile")
    if inp["alibi_slopes"] is not None:
        raise Unsupported("ALiBi is outside the reviewed varlen profile")
    _bool(inp["deterministic"], "deterministic")
    if _bool(inp["return_attn_probs"], "return_attn_probs"):
        raise Unsupported("the optional testing outputs are outside the reviewed profile")
    if inp["block_table"] is not None:
        raise Unsupported("paged K/V is outside the reviewed packed profile")

    scale_default = inp["softmax_scale"] is None
    scale = A.default_scale(dim) if scale_default else R(
        _physical([inp["softmax_scale"]], 1, "softmax_scale", "f32")[0][0])
    sequences = [{
        "q_start": cu_q[i], "q_len": q_lens[i],
        "k_start": cu_k[i], "k_len": k_lens[i],
    } for i in range(len(q_lens))]
    return {
        "q": q, "k": k, "v": v, "total_q": total_q, "total_k": total_k,
        "hq": hq, "hkv": hkv, "dim": dim, "dtype": dtype, "causal": causal,
        "scale": scale, "scale_default": scale_default, "sequences": sequences,
        "cu_q": cu_q,
    }


def _heads(parsed, sequence):
    qs, qn = sequence["q_start"], sequence["q_len"]
    ks, kn = sequence["k_start"], sequence["k_len"]
    q = [[[parsed["q"][qs + i][h] for i in range(qn)] for h in range(parsed["hq"])]]
    k = [[[parsed["k"][ks + j][h] for j in range(kn)] for h in range(parsed["hkv"])]]
    v = [[[parsed["v"][ks + j][h] for j in range(kn)] for h in range(parsed["hkv"])]]
    return q, k, v


def _evaluate(parsed, mode=None, independent=False):
    sequences = parsed["sequences"]
    if mode == "cross_sequence":
        sequences = [{
            "q_start": 0, "q_len": parsed["total_q"],
            "k_start": 0, "k_len": parsed["total_k"],
        }]
    elif mode == "q_boundaries_for_k":
        sequences = [dict(sequence, k_start=parsed["cu_q"][i], k_len=sequence["q_len"])
                     for i, sequence in enumerate(sequences)]

    scale = parsed["scale"]
    if mode == "default_one_over_dim" and parsed["scale_default"]:
        scale = R(Fraction(1, parsed["dim"]))
    elif mode == "ignore_explicit_scale" and not parsed["scale_default"]:
        scale = A.default_scale(parsed["dim"])

    output = [[[None] * parsed["dim"] for _ in range(parsed["hq"])]
              for _ in range(parsed["total_q"])]
    for sequence in sequences:
        q, k, v = _heads(parsed, sequence)
        q_len, k_len = sequence["q_len"], sequence["k_len"]

        def allowed(b, h, i, j):
            if not parsed["causal"] or mode == "causal_omitted":
                return True
            limit = i if mode == "top_left_causal" else i + k_len - q_len
            return j <= limit

        if independent:
            result = D.iv_attention(
                q, k, v, allowed=allowed, bias=lambda b, h, i, j: 0,
                scale=D.I(scale.v), empty="uniform" if mode == "empty_uniform" else "zero",
                kv_head=(lambda h: h % parsed["hkv"]) if mode == "gqa_modulo" else None,
            )
        else:
            result = A.attention(
                q, k, v, allowed=allowed, bias=lambda b, h, i, j: Fraction(0),
                scale=scale, empty="zero",
            )
        for i in range(q_len):
            for head in range(parsed["hq"]):
                value = result["out"][0][head][i]
                output[sequence["q_start"] + i][head] = value if independent else A.values(value)
    return output


def reference(profile, inp):
    return Result("value", {"out": _evaluate(parse(profile, inp))})


def _observed(value, shape, dtype):
    if not isinstance(value, dict) or set(value) != {"t", "shape", "v"}:
        raise Invalid("observed out must include exactly dtype, shape and values")
    if value["t"] != dtype or value["shape"] != list(shape):
        raise Invalid("observed out has the wrong dtype or shape")
    tensor, actual = U.tensor(value["v"], 3, "observed out")
    if actual != tuple(shape):
        raise Invalid("observed out values have the wrong shape")
    return tensor


def check_output(profile, inp, observed):
    try:
        parsed = parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"out"}:
            return False
        candidate = {"out": _observed(
            observed["out"], (parsed["total_q"], parsed["hq"], parsed["dim"]), parsed["dtype"])}
    except (Invalid, Unsupported):
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    def run(profile, inp):
        return Result("value", {"out": _evaluate(parse(profile, inp), mode, independent=True)})
    return run


MUTANTS = {
    "PK.M1": _mutant("cross_sequence"),
    "PK.M2": _mutant("top_left_causal"),
    "PK.M3": _mutant("empty_uniform"),
    "PK.M4": _mutant("gqa_modulo"),
    "PK.M5": _mutant("default_one_over_dim"),
    "PK.M6": _mutant("ignore_explicit_scale"),
    "PK.M7": _mutant("causal_omitted"),
    "PK.M8": _mutant("q_boundaries_for_k"),
}
READINGS = {}
