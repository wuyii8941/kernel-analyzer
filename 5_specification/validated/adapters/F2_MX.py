"""Reviewed torchao 0.16.0 finite MXFP8 FLOOR conversion."""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from kernel_spec import d2_common as U
from kernel_spec import ieee, mx
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational, require_fields
from kernel_spec.real import R


PROFILE = "torchao016_mxfp8_floor"
REQUIRED = {
    "x", "input_dtype", "elem_format", "block_size", "scaling_mode", "is_swizzled_scales",
}
ELEMENT_DTYPES = {"e4m3": "f8e4m3", "e5m2": "f8e5m2"}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _shape(value, name):
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    children = [_shape(item, name) for item in value]
    if any(child != children[0] for child in children):
        raise Invalid(f"{name} is ragged")
    return (len(value),) + children[0]


def _physical(value, dtype):
    for item in _walk(value):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid("x: Boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported("special input values are outside the reviewed MXFP8 profile")
        if isinstance(item, str) and item.strip().lower() in {
                "inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported("special input values are outside the reviewed MXFP8 profile")
    shape = _shape(value, "x")
    if not shape:
        raise Invalid("x must have at least one dimension")
    tensor, actual = U.tensor(value, len(shape), "x")
    if actual != shape:
        raise Invalid("x shape changed while parsing")
    tensor = U.plain(tensor)
    for item in _walk(tensor):
        rounded = ieee.round_to(R(item), dtype)
        if not rounded.exact or rounded.v != item:
            raise Invalid(f"x value {item!r} is not exactly representable as {dtype}")
    return tensor, shape


def _parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, REQUIRED, "F2_MX")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"fields outside the reviewed torchao MXFP8 call: {sorted(extra)}")
    dtype = inp["input_dtype"]
    if dtype not in ("f32", "bf16"):
        raise Unsupported("torchao 0.16 to_mx accepts f32 or bf16 input in this profile")
    fmt = inp["elem_format"]
    if fmt not in ELEMENT_DTYPES:
        raise Unsupported("the reviewed MXFP8 element format is e4m3 or e5m2")
    block_size = integer(inp["block_size"], "block_size", lo=1)
    if block_size != 32:
        raise Unsupported("the reviewed MX task fixes the standard block size at 32")
    if inp["scaling_mode"] != "floor":
        raise Unsupported("RCEIL, CEIL and EVEN are separate torchao scale profiles")
    if type(inp["is_swizzled_scales"]) is not bool:
        raise Invalid("is_swizzled_scales must be a Boolean")
    if inp["is_swizzled_scales"]:
        raise Unsupported("swizzled scale storage requires a separate physical layout binding")
    values, shape = _physical(inp["x"], dtype)
    if shape[-1] <= 0 or shape[-1] % 32:
        raise Invalid("the last input dimension must be a positive multiple of 32")
    return {"x": values, "shape": shape, "dtype": dtype, "fmt": fmt}


def reference(profile, inp):
    parsed = _parse(profile, inp)
    return Result("value", mx.torchao016_floor_tensor(parsed["x"], parsed["fmt"]))


def _tensor_record(value, *, dtype, shape, name, discrete=False):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"}:
            raise Invalid(f"observed {name} tensor record is incomplete")
        if value["t"] != dtype or value["shape"] != list(shape):
            raise Invalid(f"observed {name} has the wrong dtype or shape")
        value = value["v"]
    if _shape(value, f"observed {name}") != tuple(shape):
        raise Invalid(f"observed {name} has the wrong shape")
    if discrete:
        parsed = [integer(item, f"observed {name}", lo=0) for item in _walk(value)]
        if any(item > 254 for item in parsed):
            raise Invalid(f"observed {name} contains the reserved code255 or a value outside uint8")
    else:
        parsed = [rational(item, f"observed {name}") for item in _walk(value)]
    iterator = iter(parsed)

    def rebuild(dims):
        if not dims:
            return next(iterator)
        return [rebuild(dims[1:]) for _ in range(dims[0])]

    return rebuild(tuple(shape))


def check_output(profile, inp, observed):
    try:
        parsed = _parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"scale_codes", "elements"}:
            return False
        scale_shape = parsed["shape"][:-1] + (parsed["shape"][-1] // 32,)
        candidate = {
            "scale_codes": _tensor_record(
                observed["scale_codes"], dtype="u8", shape=scale_shape,
                name="scale_codes", discrete=True),
            "elements": _tensor_record(
                observed["elements"], dtype=ELEMENT_DTYPES[parsed["fmt"]],
                shape=parsed["shape"], name="elements"),
        }
    except (Invalid, Unsupported):
        return False
    return candidate == reference(profile, inp).fields


def _map(value, fn):
    return [_map(item, fn) for item in value] if isinstance(value, list) else fn(value)


def _drop_last(value):
    if value and isinstance(value[0], list):
        result = deepcopy(value)
        result[0] = _drop_last(result[0])
        return result
    return list(value[:-1])


def _mutant(mode):
    def run(profile, inp):
        parsed = _parse(profile, inp)
        result = deepcopy(mx.torchao016_floor_tensor(parsed["x"], parsed["fmt"]))
        if mode == "unit_scale":
            result["scale_codes"] = _map(result["scale_codes"], lambda _: 127)
        elif mode == "drop_last":
            result["elements"] = _drop_last(result["elements"])
        elif mode == "global_block":
            flat = list(_walk(parsed["x"]))
            wrong = mx.torchao016_floor_block((flat + [0] * 32)[:32], parsed["fmt"])
            result["scale_codes"] = _map(result["scale_codes"], lambda _: wrong["scale_code"])
        elif mode == "wrong_format":
            other = "e5m2" if parsed["fmt"] == "e4m3" else "e4m3"
            result = mx.torchao016_floor_tensor(parsed["x"], other)
        return Result("value", result)
    return run


MUTANTS = {
    "MX.M1": _mutant("unit_scale"),
    "MX.M2": _mutant("drop_last"),
    "MX.M3": _mutant("global_block"),
    "MX.M4": _mutant("wrong_format"),
}
READINGS = {}
