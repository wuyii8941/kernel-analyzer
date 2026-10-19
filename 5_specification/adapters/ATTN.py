"""Reviewed defined-row public PyTorch 2.10 SDPA forward and VJP references."""
from __future__ import annotations

import math
from fractions import Fraction

from kernel_spec import d1_attention as A
from kernel_spec import d1_indep as D
from kernel_spec import ieee
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, rational, require_fields
from kernel_spec.real import NEG, POS, R


FORWARD_PROFILE = "torch210_sdpa_defined_real"
VJP_PROFILE = "torch210_sdpa_defined_real_vjp"
PROFILES = {FORWARD_PROFILE, VJP_PROFILE}
BASE_REQUIRED = {
    "q", "k", "v", "dtype", "attn_mask", "attn_mask_dtype",
    "is_causal", "scale", "enable_gqa", "dropout_p",
}
DTYPES = {"f16", "bf16", "f32", "f64"}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _physical_tensor(value, rank, name, dtype):
    for item in _walk(value):
        if isinstance(item, float) and math.isinf(item) or isinstance(item, str) and item.strip() in {"inf", "+inf", "-inf"}:
            raise Unsupported(f"{name} contains infinity outside the finite SDPA profile")
    tensor, shape = A.tensor(value, rank, name)
    for item in _walk(tensor):
        rounded = ieee.round_to(R(item), dtype)
        if not rounded.exact or rounded.v != item:
            raise Invalid(f"{name} value {item!r} is not exactly representable as {dtype}")
    return tensor, shape


def _parse(profile, inp, *, allow_empty=False):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    required = BASE_REQUIRED | ({"upstream"} if profile == VJP_PROFILE else set())
    require_fields(inp, required, "ATTN")
    extra = set(inp) - required
    if extra:
        raise Unsupported(f"fields outside the selected public SDPA task: {sorted(extra)}")

    dtype = inp["dtype"]
    if dtype not in DTYPES:
        raise Unsupported("the reviewed SDPA profile requires dtype f16, bf16, f32 or f64")
    rank = len(A.shape_of(inp["q"], "q"))
    if rank not in (3, 4):
        raise Invalid("q must have rank 3 [H,L,E] or rank 4 [B,H,L,E]")
    q, qs = _physical_tensor(inp["q"], rank, "q", dtype)
    k, ks = _physical_tensor(inp["k"], rank, "k", dtype)
    v, vs = _physical_tensor(inp["v"], rank, "v", dtype)
    if any(dimension <= 0 for dimension in (*qs, *ks, *vs)):
        raise Invalid("q, k and v dimensions must all be positive")
    if rank == 3:
        q, k, v = [q], [k], [v]
        qs, ks, vs = (1,) + qs, (1,) + ks, (1,) + vs

    batch, hq, length, head_dim = qs
    if ks[0] != batch or vs[0] != batch:
        raise Unsupported("batch broadcasting is outside the reviewed SDPA profile")
    hkv, source = ks[1], ks[2]
    if ks[3] != head_dim:
        raise Invalid("query and key head dimensions differ")
    if vs[1] != hkv or vs[2] != source:
        raise Invalid("key and value head counts and sequence lengths must match")
    output_shape = (batch, hq, length, vs[3])
    dout = None
    if profile == VJP_PROFILE:
        dout, ds = _physical_tensor(inp["upstream"], rank, "upstream", dtype)
        if rank == 3:
            dout, ds = [dout], (1,) + ds
        if ds != output_shape:
            raise Invalid(f"upstream shape {ds[1:] if rank == 3 else ds} does not match output shape")

    enable_gqa = inp["enable_gqa"]
    if type(enable_gqa) is not bool:
        raise Invalid("enable_gqa must be a boolean")
    if enable_gqa:
        if hq % hkv:
            raise Invalid("the query head count must be divisible by the key/value head count with GQA")
    elif hkv not in (1, hq):
        raise Invalid("without GQA, key/value heads must equal query heads or be a singleton broadcast head")

    dropout = rational(inp["dropout_p"], "dropout_p")
    if not 0 <= dropout <= 1:
        raise Invalid("dropout_p must lie in [0, 1]")
    if dropout != 0:
        raise Unsupported("dropout_p > 0 is random and outside the reviewed SDPA profile")
    causal = inp["is_causal"]
    if type(causal) is not bool:
        raise Invalid("is_causal must be a boolean")
    mask = inp["attn_mask"]
    mask_dtype = inp["attn_mask_dtype"]
    if mask is None:
        if mask_dtype is not None:
            raise Invalid("attn_mask_dtype must be null when attn_mask is null")
    elif causal:
        raise Invalid("attn_mask and is_causal cannot both be set")

    get_mask = None
    is_bool = False
    if mask is not None:
        values, mask_shape, is_bool = A.mask_tensor(mask, "attn_mask")
        if not mask_shape or any(dimension <= 0 for dimension in mask_shape):
            raise Invalid("attn_mask must be a nonempty tensor")
        if is_bool:
            if mask_dtype != "b8":
                raise Invalid("a Boolean attention mask must declare dtype b8")
        else:
            if mask_dtype != dtype:
                raise Invalid("an additive attention mask must have the same dtype as q, k and v")
            for item in _walk(values):
                if item == POS:
                    raise Unsupported("positive infinity in an additive mask is outside the defined-real profile")
                if item != NEG:
                    rounded = ieee.round_to(R(item), dtype)
                    if not rounded.exact or rounded.v != item:
                        raise Invalid(f"additive mask value {item!r} is not exactly representable as {dtype}")
        target = (batch, hq, length, source) if rank == 4 else (hq, length, source)
        get_mask = A.broadcast_get(values, mask_shape, target)

    scale_value = inp["scale"]
    scale = A.default_scale(head_dim) if scale_value is None else R(rational(scale_value, "scale"))

    def index(b, h, i, j):
        return (b, h, i, j) if rank == 4 else (h, i, j)

    def allowed(b, h, i, j):
        if causal and j > i:
            return False
        return get_mask is None or not is_bool or bool(get_mask(index(b, h, i, j)))

    def bias(b, h, i, j):
        return get_mask(index(b, h, i, j)) if get_mask is not None and not is_bool else Fraction(0)

    if not allow_empty:
        for b in range(batch):
            for h in range(hq):
                for i in range(length):
                    if not any(allowed(b, h, i, j) and bias(b, h, i, j) != NEG for j in range(source)):
                        raise Unsupported("a query row has no participating finite score")
    return {
        "q": q, "k": k, "v": v, "dout": dout, "rank": rank, "shape": output_shape,
        "qshape": qs, "kshape": ks, "vshape": vs, "dtype": dtype,
        "allowed": allowed, "bias": bias, "scale": scale, "E": head_dim, "Hq": hq, "Hkv": hkv,
        "L": length, "S": source, "causal": causal, "get_mask": get_mask, "is_bool": is_bool, "index": index,
    }


def _strip(value, rank):
    value = A.values(value)
    return value if rank == 4 else value[0]


def reference(profile, inp):
    parsed = _parse(profile, inp)
    result = A.attention(
        parsed["q"], parsed["k"], parsed["v"], allowed=parsed["allowed"], bias=parsed["bias"],
        scale=parsed["scale"], empty="undefined", dout=parsed["dout"],
    )
    fields = {"out": _strip(result["out"], parsed["rank"])}
    if profile == VJP_PROFILE:
        fields.update({name: _strip(result[name], parsed["rank"]) for name in ("dq", "dk", "dv")})
    return Result("value", fields)


def _observed(value, rank, shape, dtype, name):
    expected_shape = shape if rank == 4 else shape[1:]
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != dtype or value.get("shape") != list(expected_shape):
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    tensor, actual = A.tensor(value, rank, f"observed {name}")
    if actual != tuple(expected_shape):
        raise Invalid(f"observed {name} shape {actual}, expected {tuple(expected_shape)}")
    return tensor


def check_output(profile, inp, observed):
    try:
        parsed = _parse(profile, inp)
        shapes = {"out": parsed["shape"]}
        if profile == VJP_PROFILE:
            shapes.update(dq=parsed["qshape"], dk=parsed["kshape"], dv=parsed["vshape"])
        if not isinstance(observed, dict) or set(observed) != set(shapes):
            return False
        candidate = {name: _observed(observed[name], parsed["rank"], shape, parsed["dtype"], name)
                     for name, shape in shapes.items()}
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _interval(value):
    return D.I(value.v if isinstance(value, R) else value)


def _independent(profile, inp, mode=None):
    parsed = _parse(profile, inp, allow_empty=(mode == "empty_uniform"))
    allowed, bias = parsed["allowed"], parsed["bias"]
    scale = _interval(parsed["scale"])
    if mode == "scale_one_over_E":
        scale = D.I(Fraction(1, parsed["E"]))
    elif mode == "ignore_explicit_scale":
        scale = _interval(A.default_scale(parsed["E"]))
    if mode == "invert_bool" and parsed["get_mask"] is not None and parsed["is_bool"]:
        get_mask, index = parsed["get_mask"], parsed["index"]
        allowed = lambda b, h, i, j: not get_mask(index(b, h, i, j))
    elif mode == "numeric_as_keep" and parsed["get_mask"] is not None and not parsed["is_bool"]:
        get_mask, index = parsed["get_mask"], parsed["index"]
        allowed = lambda b, h, i, j: get_mask(index(b, h, i, j)) == 1
        bias = lambda b, h, i, j: Fraction(0)
    elif mode == "bottom_right" and parsed["causal"]:
        offset = parsed["S"] - parsed["L"]
        allowed = lambda b, h, i, j: j <= i + offset
    kv_head = (lambda h: h % parsed["Hkv"]) if mode == "gqa_round_robin" else None
    result = D.iv_attention(
        parsed["q"], parsed["k"], parsed["v"], allowed=allowed, bias=bias, scale=scale,
        empty="uniform" if mode == "empty_uniform" else "undefined", kv_head=kv_head, dout=parsed["dout"],
    )
    names = ("out", "dq", "dk", "dv") if profile == VJP_PROFILE else ("out",)
    return Result("value", {name: result[name] if parsed["rank"] == 4 else result[name][0] for name in names})


def _mask_and_causal(profile, inp):
    if inp.get("is_causal") and inp.get("attn_mask") is not None:
        parsed = _parse(profile, dict(inp, is_causal=False))
        mask_allowed = parsed["allowed"]
        result = D.iv_attention(
            parsed["q"], parsed["k"], parsed["v"],
            allowed=lambda b, h, i, j: j <= i and mask_allowed(b, h, i, j),
            bias=parsed["bias"], scale=_interval(parsed["scale"]), empty="undefined",
        )
        out = result["out"] if parsed["rank"] == 4 else result["out"][0]
        return Result("value", {"out": out})
    return _independent(profile, inp)


def _mutant(mode):
    return lambda profile, inp: _independent(profile, inp, mode)


MUTANTS = {
    "AT.M1": _mutant("scale_one_over_E"),
    "AT.M2": _mutant("ignore_explicit_scale"),
    "AT.M3": _mutant("invert_bool"),
    "AT.M4": _mutant("numeric_as_keep"),
    "AT.M5": _mutant("bottom_right"),
    "AT.M6": _mutant("gqa_round_robin"),
    "AT.M7": _mutant("empty_uniform"),
    "AT.M8": _mask_and_causal,
}
READINGS = {}
