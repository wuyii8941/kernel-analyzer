"""Defined-row public PyTorch 2.10 SDPA forward and VJP references, plus (draft path A A2, 2026-10-20) an ATen-call
interface for the fused overloads aten._scaled_dot_product_flash_attention, its backward and
aten._scaled_dot_product_efficient_attention_backward (kernel_spec.attn_aten_semantics), dropout_p == 0 only."""
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


def _public_reference(profile, inp):
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


def _public_check_output(profile, inp, observed):
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
    return check_enclosed_output(_public_reference(profile, inp).fields, candidate)


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


# ---------------------------------------------------------------------------------------------------------------
# ATen-call interface (draft path A A2): fused SDPA overloads, profile torch210_aten_sdpa_fused_dropout0.
# Built with kernel_spec.aten_adapter.make_reference; arguments are passed through undecoded and validated by
# kernel_spec.attn_aten_semantics (tensor records are decoded there at their exact values).

from kernel_spec import attn_aten_indep as AI  # noqa: E402
from kernel_spec import attn_aten_semantics as AS  # noqa: E402
from kernel_spec.aten_adapter import make_reference  # noqa: E402

ATEN_PROFILE = "torch210_aten_sdpa_fused_dropout0"
# The covered overloads are fixed here (not read from a contract path) so that a staged copy of this adapter
# needs no access to the top-level contracts directory.
ATEN_COVERS = AS.COVERS

# Harness convention (path A rule v2): number of ATen return values and the positions the task never defines.
ATEN_RETURNS = {
    AS.FLASH: {"positions": 9, "not_compared": {
        "2": "AT.ATEN_AUX: cum_seq_q is an undefined tensor for dense inputs; it carries no task value",
        "3": "AT.ATEN_AUX: cum_seq_k is an undefined tensor for dense inputs; it carries no task value",
        "6": "AT.ATEN_AUX: rng_state (philox seed) is RNG bookkeeping not defined by the task",
        "7": "AT.ATEN_AUX: unused (philox offset) is RNG bookkeeping not defined by the task",
        "8": "AT.ATEN_AUX: debug_attn_mask is not requested (return_debug_mask=False); no task value",
    }},
    AS.FLASH_BWD: {"positions": 3, "not_compared": {}},
    AS.EFF_BWD: {"positions": 4, "not_compared": {}},
}


def _raw_argument(op, name, value, arguments):
    return value


_aten_reference = make_reference(ATEN_COVERS, extra=AS.OPS, profiles=(ATEN_PROFILE,), encode_output=False,
                                 argument_decoder=_raw_argument)


def reference(profile, inp):
    if profile == ATEN_PROFILE:
        return _aten_reference(profile, inp)
    return _public_reference(profile, inp)


def _aten_check_output(profile, inp, observed):
    """Complete-output comparison for an ATen call: every compared position with its dtype and shape;
    positions the profile does not compare (None in the reference) accept any observed value."""
    expected = _aten_reference(profile, inp).fields["out"]
    op = inp["op"]
    args, kwargs = list(inp.get("args", [])), dict(inp.get("kwargs") or {})
    parsed = AS.PARSERS[op](*args, **kwargs)
    layout = AS.output_layout(op, parsed)
    if not isinstance(observed, dict) or set(observed) != {"out"}:
        return False
    cand = observed["out"]
    if not isinstance(cand, (list, tuple)) or len(cand) != ATEN_RETURNS[op]["positions"]:
        return False
    ref_vals, obs_vals = [], []
    for ref, got, lay in zip(expected, cand, layout):
        if ref is None:
            continue
        if lay == "int":
            if type(got) is not int or got != ref:
                return False
            continue
        dtype, shape = lay
        if not isinstance(got, dict) or set(got) != {"t", "v", "shape"}:
            return False
        try:
            gd, gs, gv = AS.tensor(got, "observed")
        except (Invalid, Unsupported):
            return False
        if gd != dtype or tuple(gs) != tuple(shape):
            return False
        ref_vals.append(ref)
        obs_vals.append(_observed_points(gv))
    return check_enclosed_output(ref_vals, obs_vals)


def _observed_points(v):
    if isinstance(v, list):
        return [_observed_points(x) for x in v]
    if isinstance(v, str):
        return float(v) if v != "nan" else float("nan")
    return v


def check_output(profile, inp, observed):
    if profile == ATEN_PROFILE:
        try:
            return _aten_check_output(profile, inp, observed)
        except Invalid:
            return False
    return _public_check_output(profile, inp, observed)


def _call(inp):
    if not isinstance(inp, dict) or "op" not in inp:
        raise Unsupported("not an ATen call")
    op = inp["op"]
    if op not in AS.PARSERS:
        raise Unsupported(f"{op} is not covered")
    return op, list(inp.get("args", [])), dict(inp.get("kwargs") or {})


def _iv_scale(parsed):
    return AI.I(parsed["scale"])


def _contiguous(parsed):
    grp = parsed["Hq"] // parsed["Hkv"]
    return lambda h: h // grp


def _causal_pred(parsed, offset=0, strict=False):
    if not parsed["causal"]:
        return lambda i, j: True
    if strict:
        return lambda i, j: j < i
    return lambda i, j: j <= i + offset


def _aten_independent(profile, inp, *, scale=None, pred=None, kv_head=None, lse_divisor=None, group_sum=True,
                      saved_lse=True, saved_out=True, parse_kw=None):
    """Independent interval evaluation of one fused ATen call with the misreading given by the parameters."""
    if profile != ATEN_PROFILE:
        raise Unsupported("ATen mutant on a public profile")
    op, args, kwargs = _call(inp)
    parsed = AS.PARSERS[op](*args, **kwargs, **(parse_kw or {}))
    sc = scale(parsed) if scale else _iv_scale(parsed)
    allowed = pred(parsed) if pred else _causal_pred(parsed)
    head = kv_head(parsed) if kv_head else _contiguous(parsed)
    if op == AS.FLASH:
        out, lse = AI.iv_forward(parsed["q"], parsed["k"], parsed["v"], scale=sc, allowed=allowed, kv_head=head,
                                 lse_divisor=lse_divisor)
        res = [out, lse, None, None, parsed["L"], parsed["S"], None, None, None]
    else:
        dq, dk, dv = AI.iv_backward(parsed["q"], parsed["k"], parsed["v"], parsed["dout"], parsed["out"],
                                    parsed["lse"], scale=sc, allowed=allowed, kv_head=head, Hkv=parsed["Hkv"],
                                    group_sum=group_sum, saved_lse=saved_lse, saved_out=saved_out)
        res = [dq, dk, dv] + ([None] if op == AS.EFF_BWD else [])
    return Result("value", {"out": res})


def _am(**kw):
    return lambda profile, inp: _aten_independent(profile, inp, **kw)


def _default_scale_iv(parsed):
    return AI.I(A.default_scale(parsed["E"]))


def _round_robin(parsed):
    return lambda h: h % parsed["Hkv"]


def _bottom_right(parsed):
    return _causal_pred(parsed, offset=parsed["S"] - parsed["L"])


ATEN_MUTANTS = {
    "AT.M9": _am(scale=_default_scale_iv),                                   # ignore the explicit scale keyword
    "AT.M10": _am(pred=lambda parsed: _causal_pred(parsed, strict=True)),    # causal excludes the diagonal
    "AT.M11": _am(lse_divisor=AI.iv.log(2)),                                 # logsumexp in base 2
    "AT.M12": _am(kv_head=_round_robin),                                     # GQA heads round-robin
    "AT.M13": _am(group_sum=False),                                          # dk/dv not summed over the group
    "AT.M14": _am(saved_lse=False),                                          # recompute the LSE, ignore supplied
    "AT.M15": _am(saved_out=False),                                          # recompute out for D, ignore supplied
    "AT.M16": _am(parse_kw={"allow_dropout": True}),                         # treat dropout_p > 0 as 0
    "AT.M17": _am(pred=_bottom_right, parse_kw={"allow_nonsquare_causal": True}),  # bottom-right non-square causal
    "AT.M18": _am(kv_head=lambda parsed: (lambda h: 0)),                     # ignore GQA: every head uses KV head 0
    "AT.M19": _am(parse_kw={"allow_value_dim": True}),                      # flash accepts Ev != E
}


def _reading_recompute(profile, inp):
    """Diagnostic reading recompute_saved: backward overloads evaluated as the exact VJP of the forward at
    (q, k, v), ignoring the supplied out and logsumexp."""
    op, args, kwargs = _call(inp)
    if op == AS.FLASH:
        return reference(profile, inp)
    parsed = AS.PARSERS[op](*args, **kwargs)
    grads = [AS.values(x) for x in AS.recomputed_backward(parsed)]
    return Result("value", {"out": grads + ([None] if op == AS.EFF_BWD else [])})


def _mutant(mode):
    return lambda profile, inp: _independent(profile, inp, mode)


MUTANTS = {
    **ATEN_MUTANTS,
    "AT.M1": _mutant("scale_one_over_E"),
    "AT.M2": _mutant("ignore_explicit_scale"),
    "AT.M3": _mutant("invert_bool"),
    "AT.M4": _mutant("numeric_as_keep"),
    "AT.M5": _mutant("bottom_right"),
    "AT.M6": _mutant("gqa_round_robin"),
    "AT.M7": _mutant("empty_uniform"),
    "AT.M8": _mask_and_causal,
}
READINGS = {"recompute_saved": _reading_recompute}
