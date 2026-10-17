"""Adapter for contracts/EMB.yaml: F.embedding and F.embedding_bag with padding_idx, max_norm renormalisation,
scale_grad_by_freq, bags (sum / mean / max, offsets, include_last_offset, per_sample_weights) and the backward pass.

Outputs: out, dweight (VJP against the explicit upstream), dpsw (gradient of per_sample_weights; [] without them) and
weight_after (the weight after the in-place renormalisation). Exact rationals except the 2-norm of a row, enclosed by
kernel_spec.real. The mutants are a separate implementation (plain loops, mpmath intervals) with the misreading as a
parameter.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import d1_attention as A
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational
from kernel_spec.real import R

PROFILES = {"renorm_doc": None, "renorm_aten": Fraction(1, 10 ** 7)}
KEYS = {"op", "weight", "indices", "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "mode", "offsets",
        "per_sample_weights", "include_last_offset", "upstream"}


def _ints(x, what):
    shp = A.shape_of(x, what)

    def conv(e, depth):
        if depth == len(shp):
            return integer(e, what)
        return [conv(s, depth + 1) for s in e]
    return conv(x, 0), shp


def _flat(x):
    return [e for s in x for e in _flat(s)] if isinstance(x, list) else [x]


def parse(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    extra = set(inp) - KEYS
    if extra:
        raise Invalid(f"unknown input fields {sorted(extra)}")
    op = inp.get("op")
    if op not in ("embedding", "embedding_bag"):
        raise Unsupported(f"op {op!r}")
    W, ws = A.tensor(inp["weight"], 2, "weight")
    V, Dm = ws
    idx, ishp = _ints(inp["indices"], "indices")
    flat = _flat(idx) if ishp else [idx]
    if not ishp:
        raise Invalid("indices must be a tensor of at least one dimension")
    for i in flat:
        if not 0 <= i < V:
            raise Invalid(f"index {i} outside [0, {V})")
    pad = inp.get("padding_idx")
    if pad is not None:
        pad = integer(pad, "padding_idx")
        if not -V <= pad < V:
            raise Invalid("padding_idx must be within num_embeddings")
        pad = pad + V if pad < 0 else pad
    mn = inp.get("max_norm")
    mn = None if mn is None else rational(mn, "max_norm")
    if mn is not None and mn <= 0:
        raise Invalid("max_norm must be positive")
    p = inp.get("norm_type", 2)
    if p not in (1, 2, "inf"):
        raise Unsupported(f"norm_type {p!r} (1, 2 and inf only)")
    freq = inp.get("scale_grad_by_freq", False)
    if not isinstance(freq, bool):
        raise Invalid("scale_grad_by_freq must be a boolean")
    a = dict(op=op, W=W, V=V, D=Dm, idx=idx, ishp=ishp, flat=flat, pad=pad, mn=mn, p=p, freq=freq,
             eps=PROFILES[profile])
    if op == "embedding":
        for k in ("mode", "offsets", "per_sample_weights", "include_last_offset"):
            if inp.get(k) not in (None, False):
                raise Invalid(f"{k} is an embedding_bag argument")
        g, gs = A.tensor(inp["upstream"], len(ishp) + 1, "upstream")
        if gs != tuple(ishp) + (Dm,):
            raise Invalid("upstream must have the shape of out")
        a["g"] = _flat_rows(g, len(ishp))
        return a
    mode = inp.get("mode", "mean")
    if mode not in ("sum", "mean", "max"):
        raise Invalid("mode must be sum, mean or max")
    if mode == "max" and freq:
        raise Invalid("mode max does not support scale_grad_by_freq (documented)")
    psw = inp.get("per_sample_weights")
    if psw is not None:
        if mode != "sum":
            raise Invalid("per_sample_weights is only supported with mode sum (documented)")
        pv, ps = A.tensor(psw, len(ishp), "per_sample_weights")
        if tuple(ps) != tuple(ishp):
            raise Invalid("per_sample_weights must have the shape of indices")
        psw = _flat(pv)
    last = inp.get("include_last_offset", False)
    offs = inp.get("offsets")
    N = len(flat)
    if len(ishp) == 2:
        if offs is not None:
            raise Invalid("offsets must be None for 2-D indices")
        bounds = [(b * ishp[1], (b + 1) * ishp[1]) for b in range(ishp[0])]
    elif len(ishp) == 1:
        if offs is None:
            raise Invalid("1-D indices need offsets")
        offs = [integer(o, "offsets") for o in offs]
        if not offs or offs[0] != 0:
            raise Invalid("offsets[0] has to be 0")
        if any(b < a_ for a_, b in zip(offs, offs[1:])) or offs[-1] > N:
            raise Invalid("offsets must be non-decreasing and within the indices")
        if last:
            if offs[-1] != N or len(offs) < 2:
                raise Invalid("with include_last_offset the last offset is the end of the last bag")
            bounds = list(zip(offs[:-1], offs[1:]))
        else:
            bounds = list(zip(offs, offs[1:] + [N]))
    else:
        raise Invalid("embedding_bag indices must be 1-D or 2-D")
    g, gs = A.tensor(inp["upstream"], 2, "upstream")
    if gs != (len(bounds), Dm):
        raise Invalid("upstream must have the shape of out (bags x embedding_dim)")
    a.update(mode=mode, psw=psw, bounds=bounds, g=g)
    return a


def _flat_rows(g, depth):
    return [g] if depth == 0 else [r for s in g for r in _flat_rows(s, depth - 1)]


def _renorm(a):
    """weight after renormalising every row referenced by the indices whose p-norm exceeds max_norm."""
    W = [[R(x) for x in row] for row in a["W"]]
    if a["mn"] is None:
        return W
    for k in sorted(set(a["flat"])):
        row = a["W"][k]
        if a["p"] == 2:
            sq = sum((x * x for x in row), Fraction(0))
            over = sq > a["mn"] * a["mn"]
            nrm = rl.sqrt(R(sq))
        else:
            n = sum((abs(x) for x in row), Fraction(0)) if a["p"] == 1 else max(abs(x) for x in row)
            over = n > a["mn"]
            nrm = R(n)
        if over:
            den = nrm if a["eps"] is None else nrm + R(a["eps"])
            W[k] = [R(x) * R(a["mn"]) / den for x in row]
    return W


def reference(profile, inp):
    a = parse(profile, inp)
    W = _renorm(a)
    V, Dm, pad = a["V"], a["D"], a["pad"]
    zero = R(0)
    dW = [[zero] * Dm for _ in range(V)]
    cnt = {}
    for i in a["flat"]:
        cnt[i] = cnt.get(i, 0) + 1
    if a["op"] == "embedding":
        out_rows = [W[i] for i in a["flat"]]
        for n, i in enumerate(a["flat"]):
            if i == pad:
                continue
            f = Fraction(1, cnt[i]) if a["freq"] else Fraction(1)
            dW[i] = [x + R(gv) * R(f) for x, gv in zip(dW[i], a["g"][n])]
        out = _reshape(out_rows, a["ishp"])
        return Result("value", {"out": A.values(out), "dweight": A.values(dW), "dpsw": [],
                                "weight_after": A.values(W)})
    mode, psw, g = a["mode"], a["psw"], a["g"]
    out, dpsw = [], ([zero] * len(a["flat"]) if psw is not None else None)
    for b, (lo, hi) in enumerate(a["bounds"]):
        ents = [e for e in range(lo, hi) if a["flat"][e] != pad]
        if hi == lo:
            out.append([zero] * Dm)                     # empty bag: zeros (documented)
            continue
        if not ents:
            if mode == "sum":
                out.append([zero] * Dm)                 # empty sum
                continue
            raise Undefined(f"bag {b} holds only padding_idx entries: its {mode} has no value (EM.PAD_BAG)")
        if mode == "max":
            row, arg = [], []
            for d in range(Dm):
                best = ents[0]
                for e in ents[1:]:
                    if W[a["flat"][e]][d] > W[a["flat"][best]][d]:
                        best = e
                row.append(W[a["flat"][best]][d])
                arg.append(best)
            out.append(row)
            for d in range(Dm):
                i = a["flat"][arg[d]]
                dW[i][d] = dW[i][d] + R(g[b][d])
            continue
        wts = [R(psw[e]) if psw is not None else R(1) for e in ents]
        acc = [zero] * Dm
        for e, w in zip(ents, wts):
            acc = [s + w * x for s, x in zip(acc, W[a["flat"][e]])]
        scale = R(Fraction(1, len(ents))) if mode == "mean" else R(1)
        out.append([s * scale for s in acc])
        for e, w in zip(ents, wts):
            i = a["flat"][e]
            f = R(Fraction(1, cnt[i])) if a["freq"] else R(1)
            dW[i] = [x + R(gv) * w * scale * f for x, gv in zip(dW[i], g[b])]
            if dpsw is not None:
                tot = zero
                for gv, x in zip(g[b], W[i]):
                    tot = tot + R(gv) * x
                dpsw[e] = tot
    dp = [] if dpsw is None else _reshape(dpsw, a["ishp"])
    return Result("value", {"out": A.values(out), "dweight": A.values(dW), "dpsw": A.values(dp),
                            "weight_after": A.values(W)})


def _reshape(flat, shp):
    if len(shp) <= 1:
        return list(flat)
    step = len(flat) // shp[0]
    return [_reshape(flat[i * step:(i + 1) * step], shp[1:]) for i in range(shp[0])]


# ------------------------------------------------------------------ mutants: separate implementation
def _I(x):
    x = Fraction(x)
    return iv.mpf(x.numerator) / x.denominator


def _indep(profile, inp, m=None):
    if m == "pad_negative_not_normalised" and inp.get("padding_idx") is not None and inp["padding_idx"] < 0:
        inp = dict(inp, padding_idx=None)               # a negative padding_idx that never matches an index
    if m == "last_offset_ignored" and inp.get("include_last_offset"):
        extra = len(inp["offsets"]) - len(inp["upstream"])          # the last offset now starts one more bag
        inp = dict(inp, include_last_offset=False,
                   upstream=list(inp["upstream"]) + [[0] * len(inp["weight"][0])] * extra)
    a = parse(profile, inp)
    V, Dm, pad, flat = a["V"], a["D"], a["pad"], a["flat"]
    W = [[_I(x) for x in row] for row in a["W"]]
    eps = a["eps"]
    if m == "renorm_eps_in_doc":
        eps = Fraction(1, 10 ** 7)
    if m == "renorm_exact_in_aten":
        eps = None
    jac = {}
    if a["mn"] is not None:
        rows = range(V) if m == "renorm_every_row" else sorted(set(flat))
        for k in rows:
            x = a["W"][k]
            if a["p"] == 2:
                sq = sum((v * v for v in x), Fraction(0))
                over = sq >= a["mn"] ** 2 if m == "renorm_ge" else sq > a["mn"] ** 2
                n = iv.sqrt(_I(sq))
            else:
                nn = sum(abs(v) for v in x) if a["p"] == 1 else max(abs(v) for v in x)
                over = nn >= a["mn"] if m == "renorm_ge" else nn > a["mn"]
                n = _I(nn)
            if over:
                den = n if eps is None else n + _I(eps)
                W[k] = [_I(v) * _I(a["mn"]) / den for v in x]
                jac[k] = (x, n)
    dW = [[iv.mpf(0)] * Dm for _ in range(V)]
    cnt = {i: flat.count(i) for i in set(flat)}

    def add(i, vec):
        if m == "through_renorm" and i in jac and a["p"] == 2:
            x, n = jac[i]
            xs = [_I(v) for v in x]
            dot = sum((u * w for u, w in zip(vec, xs)), iv.mpf(0))
            vec = [_I(a["mn"]) / n * (u - w * dot / (n * n)) for u, w in zip(vec, xs)]
        dW[i] = [s + u for s, u in zip(dW[i], vec)]
    if a["op"] == "embedding":
        out = [[iv.mpf(0)] * Dm if (m == "pad_forward_zero" and i == pad) else W[i] for i in flat]
        for n_, i in enumerate(flat):
            if i == pad and m != "pad_gets_gradient":
                continue
            f = Fraction(1, cnt[i]) if a["freq"] and m != "freq_ignored" else 1
            add(i, [_I(gv) * _I(f) for gv in a["g"][n_]])
        return Result("value", {"out": _reshape(out, a["ishp"]), "dweight": dW, "dpsw": [], "weight_after": W})
    mode, psw = a["mode"], a["psw"]
    if m == "psw_ignored":
        psw = None
    out = []
    dpsw = [iv.mpf(0)] * len(flat) if a["psw"] is not None else None
    for b, (lo, hi) in enumerate(a["bounds"]):
        ents = [e for e in range(lo, hi) if flat[e] != pad]
        cnt_b = (hi - lo) if m == "mean_counts_padding" else len(ents)
        if hi == lo:
            if mode == "mean" and m == "empty_mean_undefined":
                raise Undefined("0/0")
            out.append([iv.mpf(0)] * Dm)
            continue
        if not ents:
            if mode != "sum":
                raise Undefined("only padding")
            out.append([iv.mpf(0)] * Dm)
            continue
        if mode == "max":
            if a["mn"] is not None:
                raise Invalid("mutant implementation: max mode with max_norm not implemented")
            row = []
            for d in range(Dm):
                vals = [a["W"][flat[e]][d] for e in ents]
                mx = max(vals)
                tied = [e for e, v_ in zip(ents, vals) if v_ == mx]
                row.append(_I(mx))
                share = tied if m == "max_grad_split" else tied[:1]
                for e in share:
                    dW[flat[e]][d] = dW[flat[e]][d] + _I(a["g"][b][d]) / len(share)
            out.append(row)
            continue
        sc = _I(Fraction(1, cnt_b)) if mode == "mean" else iv.mpf(1)
        acc = [iv.mpf(0)] * Dm
        for e in ents:
            w = _I(psw[e]) if psw is not None else iv.mpf(1)
            acc = [s + w * x for s, x in zip(acc, W[flat[e]])]
            f = _I(Fraction(1, cnt[flat[e]])) if a["freq"] else iv.mpf(1)
            add(flat[e], [_I(gv) * w * sc * f for gv in a["g"][b]])
            if dpsw is not None:
                dpsw[e] = sum((_I(gv) * x for gv, x in zip(a["g"][b], W[flat[e]])), iv.mpf(0))
        out.append([s * sc for s in acc])
    dp = [] if dpsw is None else _reshape(dpsw, a["ishp"])
    return Result("value", {"out": out, "dweight": dW, "dpsw": dp, "weight_after": W})


def _m(kind):
    return lambda profile, inp: _indep(profile, inp, kind)


MUTANTS = {
    "EM.M1": _m("pad_gets_gradient"),
    "EM.M2": _m("pad_forward_zero"),
    "EM.M3": _m("freq_ignored"),
    "EM.M4": _m("renorm_eps_in_doc"),
    "EM.M5": _m("renorm_exact_in_aten"),
    "EM.M6": _m("renorm_every_row"),
    "EM.M7": _m("through_renorm"),
    "EM.M8": _m("mean_counts_padding"),
    "EM.M9": _m("empty_mean_undefined"),
    "EM.M10": _m("psw_ignored"),
    "EM.M11": _m("last_offset_ignored"),
    "EM.M12": _m("max_grad_split"),
    "EM.M13": _m("renorm_ge"),
    "EM.M14": _m("pad_negative_not_normalised"),
}
READINGS = {}
