"""Adapter for contracts/DYT.yaml: Dynamic Tanh y = gamma * tanh(alpha x) + beta over the last dimension, with the
VJP w.r.t. x, alpha, gamma and beta.

Input: {x (*, C), alpha ([a], one element), gamma [C], beta [C] or null, upstream (shape of x)}.
Outputs: y and grad = {x, alpha ([.]), gamma, beta (when given)}. Reference on kernel_spec.real; the mutants are an
independent mpmath-interval implementation.
"""
from __future__ import annotations

from mpmath import iv

from kernel_spec import c_ce as T
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Unsupported, rational, to_iv, vector
from kernel_spec.real import R


def _inputs(profile, inp):
    if profile != "paper":
        raise Unsupported(f"profile {profile!r}")
    if set(inp) != {"x", "alpha", "gamma", "beta", "upstream"}:
        raise Invalid("DyT takes exactly x, alpha, gamma, beta, upstream")
    shape, x = T.tensor(inp["x"], "x")
    if not shape:
        raise Invalid("x needs a channel (last) dimension")
    C = shape[-1]
    al = vector(inp["alpha"], "alpha", 1)[0]
    ga = vector(inp["gamma"], "gamma", C)
    be = None if inp["beta"] is None else vector(inp["beta"], "beta", C)
    us, g = T.tensor(inp["upstream"], "upstream")
    if us != shape:
        raise Invalid("upstream must have the shape of x")
    return shape, x, C, al, ga, be, g


def _v(r):
    return r.v if isinstance(r, R) else r


def reference(profile, inp):
    shape, x, C, al, ga, be, g = _inputs(profile, inp)
    y, dx = [], []
    da, dg = R(0), [R(0)] * C
    db = [R(0)] * C
    for i, (xv, gv) in enumerate(zip(x, g)):
        c = i % C
        t = rl.tanh(R(al) * R(xv))
        y.append(_v(t * ga[c] + (be[c] if be is not None else 0)))
        tmp = (1 - t * t) * gv * ga[c]
        dx.append(_v(tmp * al))
        da = da + tmp * xv
        dg[c] = dg[c] + t * gv
        db[c] = db[c] + gv
    grad = {"x": T.nest(dx, shape), "alpha": [_v(da)], "gamma": [_v(v) for v in dg]}
    if be is not None:
        grad["beta"] = [_v(v) for v in db]
    return Result("value", {"y": T.nest(y, shape), "grad": grad})


def _tanh(z):
    return 1 - 2 / (iv.exp(2 * z) + 1)


def _independent(profile, inp, mis=None):
    shape, x, C, al, ga, be, g = _inputs(profile, inp)
    rows = len(x) // C
    y, dx = [None] * len(x), [None] * len(x)
    da = to_iv(0)
    dg, db = [to_iv(0)] * C, [to_iv(0)] * C
    for r in range(rows):
        for c in range(C):
            i = r * C + c
            z = _tanh(to_iv(xv := x[i])) * to_iv(al) if mis == "alpha_outside" else _tanh(to_iv(al) * to_iv(x[i]))
            y[i] = z * to_iv(ga[c]) + to_iv(be[c] if be is not None else 0)
            d = (1 - z * z) * to_iv(g[i]) * to_iv(ga[c])
            dx[i] = d if mis == "dx_no_alpha" else d * to_iv(al)
            da = da + (d if mis == "dalpha_no_x" else d * to_iv(x[i]))
            if not (mis == "dgamma_first_row" and r > 0):
                dg[c] = dg[c] + z * to_iv(g[i])
            if not (mis == "dbeta_first_row" and r > 0):
                db[c] = db[c] + to_iv(g[i])
    grad = {"x": T.nest(dx, shape), "alpha": [da], "gamma": dg}
    if be is not None:
        grad["beta"] = db
    return Result("value", {"y": T.nest(y, shape), "grad": grad})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {"DY.M1": _m("alpha_outside"), "DY.M2": _m("dx_no_alpha"), "DY.M3": _m("dalpha_no_x"),
           "DY.M4": _m("dgamma_first_row"), "DY.M5": _m("dbeta_first_row")}
READINGS = {}
