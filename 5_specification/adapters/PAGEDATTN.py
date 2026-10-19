"""Reviewed vLLM 0.16.0 causal paged-attention task reference."""
from __future__ import annotations

import math

import numpy as np

from kernel_spec import d1_attention as A
from kernel_spec import d1_indep as D
from kernel_spec import d2_common as U
from kernel_spec import ieee
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported
from kernel_spec.real import R


PROFILE = "vllm016_causal_2d_real"
KEYS = {
    "query", "cu_seqlens_q", "seq_lens", "key_cache", "value_cache",
    "block_table", "scale", "causal", "window", "softcap", "dtype",
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


def _i32_tensor(value, rank, name):
    tensor, shape = U.ints(value, rank, name)
    if any(not -(1 << 31) <= item < (1 << 31) for item in _walk(tensor)):
        raise Invalid(f"{name}: value is outside signed int32")
    return tensor, shape


def _cache_physical(value, name, dtype):
    """Physical block/slot geometry; unread slots may be null semantic placeholders."""
    if not isinstance(value, list) or not value or not all(isinstance(block, list) for block in value):
        raise Invalid(f"{name}: expected [blocks,block_size,heads,head_dim]")
    sizes = {len(block) for block in value}
    if len(sizes) != 1 or not next(iter(sizes)):
        raise Invalid(f"{name}: every physical block must have the same positive size")
    block_size = next(iter(sizes))
    dims = None
    cache = []
    for block_index, block in enumerate(value):
        converted = []
        for slot_index, slot in enumerate(block):
            if slot is None:
                converted.append(None)
                continue
            tensor, shape = _physical(slot, 2, f"{name}[{block_index}][{slot_index}]", dtype)
            if dims is None:
                dims = shape
            elif shape != dims:
                raise Invalid(f"{name}: populated slots have inconsistent head dimensions")
            converted.append(tensor)
        cache.append(converted)
    if dims is None or min(dims) <= 0:
        raise Invalid(f"{name}: at least one populated positive-size slot is required")
    return cache, (len(cache), block_size, *dims)


def parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS, required=KEYS)
    dtype = inp["dtype"]
    if dtype not in ("f16", "bf16", "f32"):
        raise Unsupported("the reviewed unquantized path supports f16, bf16 or f32 q/k/v/out")
    if inp["causal"] is not True:
        if isinstance(inp["causal"], bool):
            raise Unsupported("vLLM 0.16.0 unified_attention only supports causal attention")
        raise Invalid("causal must be a Boolean")

    q, q_shape = _physical(inp["query"], 3, "query", dtype)
    kc, k_shape = _cache_physical(inp["key_cache"], "key_cache", dtype)
    vc, v_shape = _cache_physical(inp["value_cache"], "value_cache", dtype)
    total, hq, dim = q_shape
    blocks, block_size, hkv, cache_dim = k_shape
    if min(total, hq, dim, blocks, block_size, hkv) <= 0:
        raise Invalid("all tensor dimensions must be positive")
    if v_shape != k_shape:
        raise Invalid("key_cache and value_cache must have the same physical shape")
    if cache_dim != dim:
        raise Invalid("cache head size must equal query head size")
    if hq % hkv:
        raise Invalid("query heads must be a multiple of key/value heads")

    cu, cu_shape = _i32_tensor(inp["cu_seqlens_q"], 1, "cu_seqlens_q")
    seq_lens, _ = _i32_tensor(inp["seq_lens"], 1, "seq_lens")
    table, table_shape = _i32_tensor(inp["block_table"], 2, "block_table")
    num_seqs = len(seq_lens)
    if num_seqs <= 0 or cu_shape != (num_seqs + 1,) or table_shape[0] != num_seqs:
        raise Invalid("cu_seqlens_q and block_table must describe every sequence")
    if cu[0] != 0 or cu[-1] != total or any(right <= left for left, right in zip(cu, cu[1:])):
        raise Invalid("cu_seqlens_q must start at 0, increase strictly and end at query token count")
    if table_shape[1] <= 0:
        raise Invalid("block_table must have at least one column")
    if any(index < 0 or index >= blocks for row in table for index in row):
        raise Invalid("every physical block-table entry must be in range")

    seqs = []
    for sequence, seq_len in enumerate(seq_lens):
        q_len = cu[sequence + 1] - cu[sequence]
        if seq_len < q_len:
            raise Invalid(f"sequence {sequence}: query length exceeds KV length")
        needed = (seq_len + block_size - 1) // block_size
        if needed > table_shape[1]:
            raise Invalid(f"sequence {sequence}: block table is too short")
        keys, values = [], []
        for position in range(seq_len):
            physical_block = table[sequence][position // block_size]
            key = kc[physical_block][position % block_size]
            value = vc[physical_block][position % block_size]
            if key is None or value is None:
                raise Invalid(f"sequence {sequence}: logical position {position} reads an empty cache slot")
            keys.append(key)
            values.append(value)
        seqs.append({
            "start": cu[sequence], "q_len": q_len, "seq_len": seq_len,
            "keys": keys, "values": values,
        })

    left, right = A.window_pair(inp["window"])
    if right != -1:
        raise Unsupported("the reviewed causal profile fixes the unused right window to -1")
    scale = _physical([inp["scale"]], 1, "scale", "f32")[0][0]
    softcap = _physical([inp["softcap"]], 1, "softcap", "f32")[0][0]
    if softcap < 0:
        raise Invalid("softcap must be nonnegative")
    return {
        "q": q, "total": total, "hq": hq, "hkv": hkv, "dim": dim,
        "blocks": blocks, "block_size": block_size, "kc": kc, "vc": vc,
        "table": table, "seqs": seqs, "scale": R(scale), "left": left,
        "softcap": None if softcap == 0 else R(softcap), "dtype": dtype,
    }


def _allowed(position, key, left, mode=None):
    if mode != "causal_omitted" and key > position:
        return False
    if mode == "window_ignored" or left < 0:
        return True
    distance = position - key
    if mode == "window_strict":
        return distance < left
    return distance <= left


def _heads(seq, parsed, mode=None):
    q = [[[parsed["q"][seq["start"] + i][head]
           for i in range(seq["q_len"])] for head in range(parsed["hq"])]]
    if mode == "gqa_modulo":
        keys = [[[seq["keys"][j][head % parsed["hkv"]]
                  for j in range(seq["seq_len"])] for head in range(parsed["hq"])]]
        values = [[[seq["values"][j][head % parsed["hkv"]]
                    for j in range(seq["seq_len"])] for head in range(parsed["hq"])]]
    else:
        keys = [[[seq["keys"][j][head]
                  for j in range(seq["seq_len"])] for head in range(parsed["hkv"])]]
        values = [[[seq["values"][j][head]
                    for j in range(seq["seq_len"])] for head in range(parsed["hkv"])]]
    return q, keys, values


def _evaluate(parsed, mode=None, independent=False):
    output = [[None] * parsed["hq"] for _ in range(parsed["total"])]
    for seq in parsed["seqs"]:
        q, k, v = _heads(seq, parsed, mode)
        context = 0 if mode == "positions_top_left" else seq["seq_len"] - seq["q_len"]
        scale = R(1) if mode == "scale_ignored" else parsed["scale"]
        softcap = None if mode == "softcap_ignored" else parsed["softcap"]
        allowed = lambda b, h, i, j, c=context: _allowed(c + i, j, parsed["left"], mode)
        if independent:
            result = D.iv_attention(
                q, k, v, allowed=allowed, bias=lambda b, h, i, j: 0,
                scale=D.I(scale.v), softcap=None if softcap is None else softcap.v,
                empty="undefined",
            )
        else:
            result = A.attention(
                q, k, v, allowed=allowed, bias=lambda b, h, i, j: 0,
                scale=scale, softcap=softcap, empty="undefined",
            )
        for i in range(seq["q_len"]):
            for head in range(parsed["hq"]):
                value = result["out"][0][head][i]
                output[seq["start"] + i][head] = value if independent else A.values(value)
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
            observed["out"], (parsed["total"], parsed["hq"], parsed["dim"]), parsed["dtype"])}
    except (Invalid, Unsupported):
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    def run(profile, inp):
        parsed = parse(profile, inp)
        if mode == "block_table_ignored":
            changed = []
            for seq in parsed["seqs"]:
                keys, values = [], []
                for position in range(seq["seq_len"]):
                    block = position // parsed["block_size"]
                    if block >= parsed["blocks"]:
                        raise Invalid("mutant has no same-index physical block")
                    keys.append(parsed["kc"][block][position % parsed["block_size"]])
                    values.append(parsed["vc"][block][position % parsed["block_size"]])
                changed.append(dict(seq, keys=keys, values=values))
            parsed = dict(parsed, seqs=changed)
        actual_mode = None if mode == "block_table_ignored" else mode
        return Result("value", {"out": _evaluate(parsed, actual_mode, independent=True)})
    return run


MUTANTS = {
    "PA.M1": _mutant("block_table_ignored"),
    "PA.M2": _mutant("positions_top_left"),
    "PA.M3": _mutant("window_ignored"),
    "PA.M4": _mutant("window_strict"),
    "PA.M5": _mutant("softcap_ignored"),
    "PA.M6": _mutant("gqa_modulo"),
    "PA.M7": _mutant("scale_ignored"),
    "PA.M8": _mutant("causal_omitted"),
}
READINGS = {}
