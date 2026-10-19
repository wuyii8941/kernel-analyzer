"""Adapter for contracts/NORM.yaml: LayerNorm, RMSNorm (and its offset variant) and GroupNorm with their VJPs.

Input: {op: layer_norm | rms_norm | group_norm, x, normalized_shape (layer/rms) or num_groups (group), weight, bias
(layer/group), eps, upstream, offset (profile rms_offset only)}; tensors are nested lists of exact rationals.
Outputs: y; saved = the statistics the operation returns (mean and rstd, rstd only for RMSNorm); grad = the VJP
against the upstream (x, and weight / bias when given).

The reference is kernel_spec.c_norm (exact statistics, enclosure of 1/sqrt). The mutants are an independent compact
implementation (below) with one misreading switched on each.
"""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import c_norm as CN
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational
from kernel_spec.real import R

KEYS = {"layer_norm": {"op", "x", "normalized_shape", "weight", "bias", "eps", "upstream"},
        "rms_norm": {"op", "x", "normalized_shape", "weight", "eps", "upstream"},
        "group_norm": {"op", "x", "num_groups", "weight", "bias", "eps", "upstream"}}
PROFILES = {"torch": ("layer_norm", "rms_norm", "group_norm"),
            "torch_autograd": ("layer_norm", "rms_norm", "group_norm"),
            "rms_offset": ("rms_norm",)}


def _check(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    op = inp.get("op")
    if op not in PROFILES[profile]:
        raise Unsupported(f"{op!r} is not in profile {profile}") if op in KEYS else Invalid(f"unknown op {op!r}")
    want = KEYS[op] | ({"offset"} if profile == "rms_offset" else set())
    if set(inp) != want:
        raise Invalid(f"{op} in profile {profile} takes exactly {sorted(want)}, got {sorted(inp)}")
    if not isinstance(inp.get("normalized_shape", [1]), list):
        raise Invalid("normalized_shape must be a list")
    if profile == "torch_autograd" and op == "group_norm":
        shape = CN.shape_of(inp.get("x"), "x")
        if len(shape) >= 2:
            groups = integer(inp.get("num_groups"), "num_groups", lo=1)
            n, channels = shape[:2]
            if channels % groups == 0:
                public_size = n * (channels // groups) * CN.prod(shape[2:])
                if public_size == 1:
                    raise Invalid("public torch group_norm requires more than one value in its batch-size check")
    return op


def reference(profile, inp):
    op = _check(profile, inp)
    if op == "layer_norm":
        out = CN.layer_norm(inp["x"], inp["normalized_shape"], inp["weight"], inp["bias"], inp["eps"], inp["upstream"])
    elif op == "rms_norm":
        off = rational(inp["offset"], "offset") if profile == "rms_offset" else None
        out = CN.rms_norm(inp["x"], inp["normalized_shape"], inp["weight"], inp["eps"], inp["upstream"], offset=off)
    else:
        out = CN.group_norm(inp["x"], inp["num_groups"], inp["weight"], inp["bias"], inp["eps"], inp["upstream"])
    if profile == "torch_autograd":
        out = {name: out[name] for name in ("y", "grad")}
    return Result("value", out)


def check_output(profile, inp, observed):
    """Compare the complete interface selected by the profile."""
    from kernel_spec.compare import check_enclosed_output
    return check_enclosed_output(reference(profile, inp).fields, observed)


# ------------------------------------------------------------------ independent implementation for the mutants
def _flat(x):
    return [v for e in x for v in _flat(e)] if isinstance(x, list) else [x]


def _shape(x):
    return (len(x),) + _shape(x[0]) if isinstance(x, list) else ()


def _nest(f, shape):
    if not shape:
        return f[0]
    k = len(f) // shape[0]
    return [_nest(f[i * k:(i + 1) * k], shape[1:]) for i in range(shape[0])]


def _v(r):
    return r.v if isinstance(r, R) else r


def _independent(profile, inp, mis=None):
    op = _check(profile, inp)
    x = [rational(v) for v in _flat(inp["x"])]
    shp = _shape(inp["x"])
    g = [rational(v) for v in _flat(inp["upstream"])]
    eps = rational(inp["eps"])
    if mis == "negative_eps_invalid" and eps < 0:
        raise Invalid("eps < 0")
    w = None if inp.get("weight") is None else [rational(v) for v in _flat(inp["weight"])]
    b = None if inp.get("bias") is None else [rational(v) for v in _flat(inp["bias"])]
    if op in ("layer_norm", "rms_norm"):
        ns = [integer(v) for v in inp["normalized_shape"]]
        if mis == "last_axis_only":
            ns = ns[-1:]
        inner = 1
        for v in ns:
            inner *= v
        sets = [list(range(i, i + inner)) for i in range(0, len(x), inner)]
        chan = lambda i: i % inner                                        # noqa: E731  parameter index
        outer = shp[:len(shp) - len(inp["normalized_shape"])]
        pshape = (inner,) if mis == "last_axis_only" else tuple(inp["normalized_shape"])
    else:
        N, C = shp[0], shp[1]
        G = integer(inp["num_groups"])
        HW = len(x) // (N * C)
        cpg = C // G
        if mis == "interleaved_groups":
            sets = [[(n * C + c) * HW + s for c in range(C) if c % G == gi for s in range(HW)]
                    for n in range(N) for gi in range(G)]
        else:
            sets = [[(n * C + c) * HW + s for c in range(gi * cpg, (gi + 1) * cpg) for s in range(HW)]
                    for n in range(N) for gi in range(G)]
        chan = (lambda i: (i // HW) % C // cpg) if mis == "weight_per_group" else (lambda i: (i // HW) % C)
        outer = (N, G)
        pshape = (C,)
    off = rational(inp["offset"]) if profile == "rms_offset" and mis != "offset_ignored" else 0
    if profile == "torch" and op == "rms_norm" and mis == "offset_in_torch":
        off = 1
    weff = None if w is None else [off + v for v in w]
    y, dx = [None] * len(x), [None] * len(x)
    dw = None if w is None else [R(0)] * len(w)
    db = None if b is None else [R(0)] * len(b)
    means, rstds = [], []
    centred = op != "rms_norm" or mis == "rms_subtracts_mean"
    for S in sets:
        n = len(S)
        vals = [x[i] for i in S]
        mu = sum(vals, Fraction(0)) / n if centred else Fraction(0)
        ss = sum(((v - mu) ** 2 for v in vals), Fraction(0))
        var = ss / (n - 1) if mis == "unbiased" else ss / n
        if mis == "zero_var_undefined" and var == 0:
            raise Undefined("constant set")
        if mis in ("eps_outside_sqrt", "rms_eps_outside_sqrt"):
            den = rl.sqrt(R(var)) + eps
            if den.sign() == 0:
                raise Undefined("zero denominator")
            r = R(1) / den
        else:
            if var + eps <= 0:
                raise Undefined("var + eps <= 0")
            r = R(1) / rl.sqrt(R(var + eps))
        xh = [R(v - mu) * r for v in vals]
        h = [R(g[i]) * (weff[chan(i)] if weff is not None else 1) for i in S]
        mh = sum(h, R(0)) / n if centred and mis != "dx_no_mean_term" else R(0)
        mhx = sum((a * c for a, c in zip(h, xh)), R(0)) / n if mis != "dx_no_var_term" else R(0)
        for k, i in enumerate(S):
            c = chan(i)
            yi = xh[k] * (weff[c] if weff is not None else 1)
            y[i] = yi + b[c] if b is not None else yi
            dx[i] = r * (h[k] - mh - xh[k] * mhx)
        if dw is not None:
            for k, i in enumerate(S):
                if mis == "dw_first_set_only" and S is not sets[0]:
                    continue
                dw[chan(i)] = dw[chan(i)] + xh[k] * g[i]
        if db is not None:
            for i in S:
                db[chan(i)] = db[chan(i)] + g[i]
        means.append(mu), rstds.append(_v(r))
    grad = {"x": _nest([_v(v) for v in dx], shp)}
    if dw is not None:
        grad["weight"] = _nest([_v(v) for v in dw], pshape)
    if db is not None:
        grad["bias"] = _nest([_v(v) for v in db], pshape)
    saved = {"rstd": _nest(rstds, outer)} if op == "rms_norm" else {"mean": _nest(means, outer),
                                                                     "rstd": _nest(rstds, outer)}
    return Result("value", {"y": _nest([_v(v) for v in y], shp), "saved": saved, "grad": grad})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "NM.M1": _m("unbiased"),
    "NM.M2": _m("eps_outside_sqrt"),
    "NM.M3": _m("offset_ignored"),
    "NM.M4": _m("offset_in_torch"),
    "NM.M5": _m("rms_subtracts_mean"),
    "NM.M6": _m("interleaved_groups"),
    "NM.M7": _m("dx_no_mean_term"),
    "NM.M8": _m("dx_no_var_term"),
    "NM.M9": _m("rms_eps_outside_sqrt"),
    "NM.M10": _m("last_axis_only"),
    "NM.M11": _m("weight_per_group"),
    "NM.M12": _m("dw_first_set_only"),
    "NM.M13": _m("zero_var_undefined"),
    "NM.M14": _m("negative_eps_invalid"),
}
READINGS = {}
