"""vLLM 0.16.0 multi-LoRA shrink task reference."""
from __future__ import annotations

import math

import numpy as np

from kernel_spec import d2_common as U
from kernel_spec import ieee
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported
from kernel_spec.real import R


PROFILE = "vllm016_shrink_f32out_real"
KEYS = {"x", "weights", "mapping", "scaling", "out_before", "input_dtype", "output_dtype"}


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
        if isinstance(item, str) and item.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported(f"{name}: special values are outside the reviewed profile")
    tensor, shape = U.tensor(value, rank, name)
    for item in _walk(tensor):
        rounded = ieee.round_to(item, dtype)
        if not rounded.exact or rounded.v != item.v:
            raise Invalid(f"{name}: value {item.v!r} is not exactly representable as {dtype}")
    return tensor, shape


def parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS, required=KEYS)
    dtype = inp["input_dtype"]
    if dtype not in ("f16", "bf16"):
        raise Unsupported("the installed active LoRA shrink call requires float16 or bfloat16 input and weights")
    if inp["output_dtype"] != "f32":
        raise Unsupported("the reviewed profile fixes the common float32 output buffer")

    x, x_shape = _physical(inp["x"], 2, "x", dtype)
    weights, weight_shape = _physical(inp["weights"], 4, "weights", dtype)
    before, before_shape = _physical(inp["out_before"], 3, "out_before", "f32")
    tokens, hidden = x_shape
    slices, loras, rank, weight_hidden = weight_shape
    if min(tokens, hidden, slices, loras, rank) <= 0:
        raise Invalid("token, hidden, slice, LoRA and rank dimensions must be positive")
    if weight_hidden != hidden:
        raise Invalid("the input and LoRA A weights must have the same hidden dimension")
    if before_shape != (slices, tokens, rank):
        raise Invalid("out_before must have shape [slices,tokens,rank]")
    mapping, mapping_shape = U.ints(inp["mapping"], 1, "mapping")
    if mapping_shape != (tokens,) or any(index < -1 or index >= loras for index in mapping):
        raise Invalid("mapping must contain one value in [-1,num_loras) for every token")
    scaling = _physical([inp["scaling"]], 1, "scaling", "f32")[0][0]
    return {
        "x": x, "weights": weights, "mapping": mapping, "scaling": scaling,
        "before": before, "tokens": tokens, "hidden": hidden, "slices": slices,
        "loras": loras, "rank": rank,
    }


def reference(profile, inp):
    parsed = parse(profile, inp)
    if all(index == -1 for index in parsed["mapping"]):
        return Result("value", {"y": U.plain(parsed["before"])})
    output = U.zeros(parsed["slices"], parsed["tokens"], parsed["rank"])
    for slice_index in range(parsed["slices"]):
        for token in range(parsed["tokens"]):
            lora = parsed["mapping"][token]
            if lora < 0:
                continue
            for rank_index in range(parsed["rank"]):
                total = R(0)
                for hidden in range(parsed["hidden"]):
                    total += parsed["weights"][slice_index][lora][rank_index][hidden] * parsed["x"][token][hidden]
                output[slice_index][token][rank_index] = parsed["scaling"] * total
    return Result("value", {"y": U.plain(output)})


def _observed(value, shape):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != "f32" or value.get("shape") != list(shape):
            raise Invalid("observed y tensor record has the wrong dtype or shape")
        value = value["v"]
    tensor, actual = U.tensor(value, 3, "observed y")
    if actual != tuple(shape):
        raise Invalid(f"observed y shape {actual}, expected {tuple(shape)}")
    return tensor


def check_output(profile, inp, observed):
    try:
        parsed = parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"y"}:
            return False
        candidate = {"y": _observed(observed["y"], (parsed["slices"], parsed["tokens"], parsed["rank"]))}
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _loop(profile, inp, mode):
    parsed = parse(profile, inp)
    if all(index == -1 for index in parsed["mapping"]) and mode != "zero_always":
        return Result("value", {"y": U.plain(parsed["before"])})
    output = [[[R(0) for _ in range(parsed["rank"])] for _ in range(parsed["tokens"])]
              for _ in range(parsed["slices"])]
    for slice_index in range(parsed["slices"]):
        source_slice = 0 if mode == "first_slice_only" else slice_index
        for token in range(parsed["tokens"]):
            lora = parsed["mapping"][token]
            if lora < 0 and mode != "negative_index_wraps":
                continue
            for rank_index in range(parsed["rank"]):
                total = R(0)
                for hidden in range(parsed["hidden"]):
                    if mode == "transposed":
                        weight = parsed["weights"][source_slice][lora][hidden][rank_index]
                    else:
                        weight = parsed["weights"][source_slice][lora][rank_index][hidden]
                    total += weight * parsed["x"][token][hidden]
                output[slice_index][token][rank_index] = total if mode == "scaling_ignored" else parsed["scaling"] * total
    return Result("value", {"y": U.plain(output)})


def _mutant(mode):
    return lambda profile, inp: _loop(profile, inp, mode)


MUTANTS = {
    "LR.M1": _mutant("scaling_ignored"),
    "LR.M2": _mutant("negative_index_wraps"),
    "LR.M3": _mutant("first_slice_only"),
    "LR.M4": _mutant("transposed"),
    "LR.M5": _mutant("zero_always"),
}
READINGS = {}
