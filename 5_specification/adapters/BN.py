"""Adapter for contracts/BN.yaml: one BatchNorm call (training or evaluation) with the running-statistics update and
the VJP.

Input: {x (N, C, *), weight, bias (null or [C]), training (bool), momentum (rational, or null for the cumulative
average), eps, state (null = track_running_stats False, or {running_mean, running_var[, num_batches_tracked]}),
upstream (shape of x)}. Outputs: y, state (post-state, {} when untracked), saved (mean, rstd used to normalise), grad.

The reference is kernel_spec.c_norm.batch_norm; the mutants are an independent implementation (below).
"""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import c_norm as CN
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational
from kernel_spec.real import R

KEYS = {"x", "weight", "bias", "training", "momentum", "eps", "state", "upstream"}


def _check(profile, inp):
    if profile != "torch":
        raise Unsupported(f"profile {profile!r}")
    if set(inp) != KEYS:
        raise Invalid(f"BN takes exactly {sorted(KEYS)}, got {sorted(inp)}")


def reference(profile, inp):
    _check(profile, inp)
    out = CN.batch_norm(inp["x"], inp["weight"], inp["bias"], inp["training"], inp["momentum"], inp["eps"],
                        inp["state"], inp["upstream"])
    return Result("value", out)


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
    _check(profile, inp)
    x = [rational(v) for v in _flat(inp["x"])]
    shp = _shape(inp["x"])
    N, C = shp[0], shp[1]
    HW = len(x) // (N * C)
    g = [rational(v) for v in _flat(inp["upstream"])]
    w = None if inp["weight"] is None else [rational(v) for v in inp["weight"]]
    b = None if inp["bias"] is None else [rational(v) for v in inp["bias"]]
    eps = rational(inp["eps"])
    tr = inp["training"]
    st = inp["state"]
    if eps <= 0 or (tr and N * HW == 1):
        raise Invalid("eps <= 0 or one value per channel in training")
    rm = rv = nbt = None
    if st is not None:
        rm = [rational(v) for v in st["running_mean"]]
        rv = [rational(v) for v in st["running_var"]]
        nbt = integer(st["num_batches_tracked"]) if "num_batches_tracked" in st else None
    use_batch = tr or st is None or mis == "eval_batch_stats"
    new_nbt = nbt
    if tr and st is not None and nbt is not None and mis != "nbt_not_incremented":
        new_nbt = nbt + 1
    y, dx = [None] * len(x), [None] * len(x)
    dw = None if w is None else [R(0)] * C
    db = None if b is None else [R(0)] * C
    new_rm, new_rv = (list(rm), list(rv)) if st is not None else (None, None)
    means, rstds = [], []
    count = nbt or 0
    for c in range(C):
        if mis == "per_position":
            groups = [[(k * C + c) * HW + s for k in range(N)] for s in range(HW)]
        else:
            groups = [[(k * C + c) * HW + s for k in range(N) for s in range(HW)]]
        if tr and st is not None and mis == "counter_per_channel":
            count += 1
        if (tr or mis == "eval_updates") and st is not None and mis != "no_update":
            allv = [x[(k * C + c) * HW + s] for k in range(N) for s in range(HW)]
            na = len(allv)
            f = rational(inp["momentum"]) if inp["momentum"] is not None else Fraction(
                1, count if mis == "counter_per_channel" else new_nbt)
            bm = sum(allv, Fraction(0)) / na
            bss = sum(((v - bm) ** 2 for v in allv), Fraction(0))
            unb = bss / na if mis == "biased_running" else bss / (na - 1)
            if mis == "momentum_reversed":
                new_rm[c], new_rv[c] = f * rm[c] + (1 - f) * bm, f * rv[c] + (1 - f) * unb
            else:
                new_rm[c], new_rv[c] = (1 - f) * rm[c] + f * bm, (1 - f) * rv[c] + f * unb
        for S in groups:
            n = len(S)
            vals = [x[i] for i in S]
            if use_batch:
                mu = sum(vals, Fraction(0)) / n
                ss = sum(((v - mu) ** 2 for v in vals), Fraction(0))
                var = ss / (n - 1) if mis == "unbiased_normalisation" else ss / n
            else:
                mu, var = rm[c], rv[c]
            r = R(1) / rl.sqrt(R(var + eps))
            xh = [R(v - mu) * r for v in vals]
            wc = w[c] if w is not None else 1
            h = [R(g[i]) * wc for i in S]
            if use_batch and mis != "train_dx_as_eval":
                mh = sum(h, R(0)) / n
                mhx = sum((a * q for a, q in zip(h, xh)), R(0)) / n
                for k, i in enumerate(S):
                    dx[i] = r * (h[k] - mh - xh[k] * mhx)
            else:
                for k, i in enumerate(S):
                    dx[i] = r * h[k]
            for k, i in enumerate(S):
                yi = xh[k] * wc
                y[i] = yi + b[c] if b is not None else yi
                if dw is not None:
                    dw[c] = dw[c] + xh[k] * g[i]
                if db is not None:
                    db[c] = db[c] + g[i]
            means.append(mu), rstds.append(_v(r))
    state = {}
    if st is not None:
        state = {"running_mean": new_rm, "running_var": new_rv}
        if nbt is not None:
            state["num_batches_tracked"] = new_nbt
    grad = {"x": _nest([_v(v) for v in dx], shp)}
    if dw is not None:
        grad["weight"] = [_v(v) for v in dw]
    if db is not None:
        grad["bias"] = [_v(v) for v in db]
    return Result("value", {"y": _nest([_v(v) for v in y], shp), "state": state,
                            "saved": {"mean": means, "rstd": rstds}, "grad": grad})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "BN.M1": _m("biased_running"),
    "BN.M2": _m("momentum_reversed"),
    "BN.M3": _m("eval_batch_stats"),
    "BN.M4": _m("counter_per_channel"),
    "BN.M5": _m("unbiased_normalisation"),
    "BN.M6": _m("train_dx_as_eval"),
    "BN.M7": _m("no_update"),
    "BN.M8": _m("nbt_not_incremented"),
    "BN.M9": _m("per_position"),
    "BN.M10": _m("eval_updates"),
}
READINGS = {}
