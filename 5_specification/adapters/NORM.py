"""Adapter for contracts/NORM.yaml: LayerNorm, RMSNorm (and its offset variant) and GroupNorm with their VJPs.

Input: {op: layer_norm | rms_norm | group_norm, x, normalized_shape (layer/rms) or num_groups (group), weight, bias
(layer/group), eps, upstream, offset (profile rms_offset only)}; tensors are nested lists of exact rationals.
Outputs: y; saved = the statistics the operation returns (mean and rstd, rstd only for RMSNorm); grad = the VJP
against the upstream (x, and weight / bias when given).

The reference is kernel_spec.c_norm (exact statistics, enclosure of 1/sqrt). The mutants are an independent compact
implementation (below) with one misreading switched on each.

Profile aten_native is the ATen-call interface (kernel_spec.aten_adapter.make_reference) for aten.native_layer_norm,
aten.native_layer_norm_backward and aten.rms_norm: input {op, args, kwargs}, output out = the ATen return (semantics in
kernel_spec.norm_emb_nll_aten); its mutants NM.M15-NM.M21 are independent implementations below.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

import yaml

from kernel_spec import aten as A
from kernel_spec import c_norm as CN
from kernel_spec import norm_emb_nll_aten as NA
from kernel_spec import real as rl
from kernel_spec.aten_adapter import make_reference
from pathlib import Path

# The staged contract next to this adapter (stage/<X>/adapters/NORM.py -> stage/<X>/contracts/NORM.yaml).
CONTRACTS = Path(__file__).resolve().parent.parent / "contracts"
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational
from kernel_spec.real import R

KEYS = {"layer_norm": {"op", "x", "normalized_shape", "weight", "bias", "eps", "upstream"},
        "rms_norm": {"op", "x", "normalized_shape", "weight", "eps", "upstream"},
        "group_norm": {"op", "x", "num_groups", "weight", "bias", "eps", "upstream"}}
PROFILES = {"torch": ("layer_norm", "rms_norm", "group_norm"),
            "torch_autograd": ("layer_norm", "rms_norm", "group_norm"),
            "rms_offset": ("rms_norm",)}


ATEN_PROFILE = "aten_native"
ATEN_OPS = ("aten.native_layer_norm", "aten.native_layer_norm_backward", "aten.rms_norm")
# Harness convention for tuple-returning overloads (ATen return order); every position is defined by the task.
ATEN_RETURNS = {"aten.native_layer_norm": {"positions": 3, "not_compared": {}},
                "aten.native_layer_norm_backward": {"positions": 3, "not_compared": {}}}
_COVERS = yaml.safe_load((CONTRACTS / "NORM.yaml").read_text())["covers"]
assert set(ATEN_OPS) <= set(_COVERS)


def _aten_reference(extra=None):
    return make_reference(ATEN_OPS, extra={**{k: NA.OPS[k] for k in ATEN_OPS}, **(extra or {})},
                          profiles=(ATEN_PROFILE,), argument_decoder=NA.decode_argument)


_aten = _aten_reference()


def _check(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    op = inp.get("op")
    if isinstance(op, str) and op.startswith("aten."):
        raise Unsupported(f"{op}: ATen calls are answered by profile {ATEN_PROFILE} only")
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
    if profile == ATEN_PROFILE:
        return _aten(profile, inp)
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
    if profile == ATEN_PROFILE:
        return _check_aten(profile, inp, observed)
    return check_enclosed_output(reference(profile, inp).fields, observed)


def _check_aten(profile, inp, observed):
    """Complete ATen result: every position, shape and value; out / grad dtypes as the reference; the dtype of the
    returned mean / rstd is metadata (decision NM.ATEN_DTYPES): the input dtype or float32 are both accepted."""
    from kernel_spec.aten_adapter import decode, encode
    from kernel_spec.compare import check_enclosed_output
    raw = make_reference(ATEN_OPS, extra={k: NA.OPS[k] for k in ATEN_OPS}, profiles=(ATEN_PROFILE,),
                         encode_output=False, argument_decoder=NA.decode_argument)
    expected = raw(profile, inp).fields["out"]
    if not isinstance(observed, dict) or set(observed) != {"out"}:
        return False
    exp = expected if isinstance(expected, list) else [expected]
    obs = observed["out"] if isinstance(expected, list) else [observed["out"]]
    if not isinstance(obs, (list, tuple)) or len(obs) != len(exp):
        return False
    stats = {1, 2} if inp.get("op") == "aten.native_layer_norm" else set()
    values = []
    for i, (e, o) in enumerate(zip(exp, obs)):
        if e is None or o is None:
            if e is not o:
                return False
            values.append(None)
            continue
        if not isinstance(o, dict) or set(o) != {"t", "v", "shape"} or tuple(o["shape"]) != e.shape:
            return False
        allowed = {e.dtype, "f32"} if i in stats else {e.dtype}
        try:
            if A.dt(o["t"]) not in allowed:
                return False
            d = decode(o)
        except (Invalid, Unsupported):
            return False
        values.append(encode(d))
    return check_enclosed_output(encode(exp), values)


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


# ------------------------------------------------------------------ independent ATen-call implementations (mutants)
def _rows(t, n):
    vals = [v.v for v in t.a.reshape(-1)]
    return [vals[i:i + n] for i in range(0, len(vals), n)]


def _aten_ln(mis):
    def op(input, normalized_shape, weight, bias, eps):
        n = 1
        for v in normalized_shape:
            n *= int(v)
        e = Fraction(eps) if not isinstance(eps, R) else eps.v
        w = None if weight is None else [v.v for v in weight.a.reshape(-1)]
        b = None if bias is None else [v.v for v in bias.a.reshape(-1)]
        ys, ms, rs = [], [], []
        for row in _rows(input, n):
            mu = sum(row, Fraction(0)) / len(row)
            ss = sum(((v - mu) ** 2 for v in row), Fraction(0))
            var = ss / (len(row) - 1) if mis == "unbiased" else ss / len(row)
            if mis == "eps_outside_sqrt":
                r = R(1) / (rl.sqrt(R(var)) + e)
            else:
                r = R(1) / rl.sqrt(R(var + e))
            for j, v in enumerate(row):
                y = R(v - mu) * r
                y = y * w[j] if w is not None else y
                ys.append(y + b[j] if b is not None else y)
            ms.append(R(mu))
            rs.append(R(1) / r if mis == "rstd_as_std" else r)
        shp = tuple(input.shape)
        stat = shp[:len(shp) - len(normalized_shape)] + (1,) * len(normalized_shape)
        mk = lambda vals, s: A.Ten(np.array(vals, dtype=object).reshape(s), input.dtype)  # noqa: E731
        return [mk(ys, shp), mk(ms, stat), mk(rs, stat)]
    return op


def _aten_ln_bwd(mis):
    def op(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask):
        n = 1
        for v in normalized_shape:
            n *= int(v)
        xs, gs = _rows(input, n), _rows(grad_out, n)
        mu = [v.v for v in mean.a.reshape(-1)]
        rs = [v.v for v in rstd.a.reshape(-1)]
        w = None if weight is None else [v.v for v in weight.a.reshape(-1)]
        dx, dw, db = [], [Fraction(0)] * n, [Fraction(0)] * n
        for i, (row, g) in enumerate(zip(xs, gs)):
            if mis == "recompute_stats":
                m = sum(row, Fraction(0)) / n
                var = sum(((v - m) ** 2 for v in row), Fraction(0)) / n
                r = R(1) / rl.sqrt(R(var))
            else:
                m, r = mu[i], R(rs[i])
            xh = [R(v - m) * r for v in row]
            h = [R(a) * (w[j] if w is not None else 1) for j, a in enumerate(g)]
            mh = sum(h, R(0)) / n
            mhx = sum((a * c for a, c in zip(h, xh)), R(0)) / n
            dx.extend(r * (h[j] - mh - xh[j] * mhx) for j in range(n))
            for j in range(n):
                dw[j] = xh[j] * g[j] + dw[j]
                db[j] = db[j] + g[j]
        mask = [True, True, True] if mis == "mask_ignored" else list(output_mask)
        ns = tuple(int(v) for v in normalized_shape)
        mk = lambda vals, s, d: A.Ten(np.array([R(v) for v in vals], dtype=object).reshape(s), d)  # noqa: E731
        return [mk(dx, tuple(input.shape), input.dtype) if mask[0] else None,
                mk(dw, ns, input.dtype) if mask[1] and weight is not None else None,
                mk(db, ns, input.dtype) if mask[2] and bias is not None else None]
    return op


def _aten_rms(default_eps):
    def op(input, normalized_shape, weight=None, eps=None):
        n = 1
        for v in normalized_shape:
            n *= int(v)
        e = default_eps if eps is None else (eps.v if isinstance(eps, R) else Fraction(eps))
        w = None if weight is None else [v.v for v in weight.a.reshape(-1)]
        ys = []
        for row in _rows(input, n):
            ms = sum((v * v for v in row), Fraction(0)) / n + e
            r = R(1) / rl.sqrt(R(ms))
            ys.extend(R(v) * r * (w[j] if w is not None else 1) for j, v in enumerate(row))
        return A.Ten(np.array(ys, dtype=object).reshape(tuple(input.shape)), input.dtype)
    return op


def _am(op, fn):
    ref = _aten_reference({op: fn})
    return lambda profile, inp: ref(profile, inp)


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
    "NM.M15": _am("aten.native_layer_norm", _aten_ln("eps_outside_sqrt")),
    "NM.M16": _am("aten.native_layer_norm", _aten_ln("unbiased")),
    "NM.M17": _am("aten.native_layer_norm", _aten_ln("rstd_as_std")),
    "NM.M18": _am("aten.native_layer_norm_backward", _aten_ln_bwd("recompute_stats")),
    "NM.M19": _am("aten.native_layer_norm_backward", _aten_ln_bwd("mask_ignored")),
    "NM.M20": _am("aten.rms_norm", _aten_rms(Fraction(1, 100000))),
    "NM.M21": _am("aten.rms_norm", _aten_rms(Fraction(0))),
}
READINGS = {}
