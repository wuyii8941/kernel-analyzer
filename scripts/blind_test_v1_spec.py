"""Task specifications f of blind_test_v1 phase 2, evaluated independently of every program's TTIR.

Each family's formula from the phase-2 release is written out directly on the family's inputs, using the
interval operations of reference_eval.intervals (rigorous float64 enclosures with directed rounding):

* sums use fsum on the endpoint arrays (the exact sum of the lower / upper endpoints, rounded to nearest,
  then moved one ulp outward), so sums of exactly representable terms have a width of about one ulp;
* products / quotients / square roots use the directed bounds of the interval module; (.)^(-1/2) is a
  directed square root followed by a directed reciprocal;
* exp / log (F7) use MPFR with directed rounding (intervals.elementary_bounds);
* F8: sign(0) = +1.

``scalars="given"`` uses the runtime scalars exactly as make_inputs returns them (Python floats);
``scalars="fp32"`` rounds them to float32 first, the type the kernels receive (a sensitivity check).
Every function returns (lo, hi) of the output, flattened in row-major order.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from kernel_analyzer.reference_eval import intervals as iv


def _f64(t):
    return t.detach().cpu().to(torch.float64).numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float64)


def _scalar(v, mode):
    return float(np.float32(v)) if mode == "fp32" else float(v)


def tsum(lo, hi, axis):
    """Tight enclosure of the exact sum along ``axis`` (fsum of each endpoint array, one ulp outward)."""

    lo_m = np.moveaxis(lo, axis, -1)
    hi_m = np.moveaxis(hi, axis, -1)
    flat_lo = lo_m.reshape(-1, lo_m.shape[-1])
    flat_hi = hi_m.reshape(-1, hi_m.shape[-1])
    s_lo = np.array([math.fsum(r) for r in flat_lo]).reshape(lo_m.shape[:-1])
    s_hi = np.array([math.fsum(r) for r in flat_hi]).reshape(hi_m.shape[:-1])
    return np.nextafter(s_lo, -np.inf), np.nextafter(s_hi, np.inf)


def _point(x):
    x = np.asarray(x, dtype=np.float64)
    return x, x


def _const(c, shape=()):
    v = np.full(shape, float(c))
    return v, v


def _rsqrt(lo, hi):
    s_lo, s_hi = iv.isqrt(lo, hi)
    return iv.idiv(np.ones_like(s_lo), np.ones_like(s_hi), s_lo, s_hi)


def _gate(alo, ahi):
    """g(a) = 0.5 * (1 + a / (1 + |a|)) for point a."""

    one = np.ones_like(alo)
    den = iv.iadd(one, one, np.abs(alo), np.abs(ahi))
    q = iv.idiv(alo, ahi, *den)
    g = iv.iadd(one, one, *q)
    return iv.imul(*g, 0.5 * one, 0.5 * one)


def f_F1(inp, scalars="given"):
    x, s, b = _f64(inp["x"]), _f64(inp["s"]), _f64(inp["b"])
    eps = _scalar(inp["eps"], scalars)
    d = x.shape[1]
    sq = iv.imul(x, x, x, x, same=True)
    ms = tsum(*sq, axis=1)
    ms = iv.idiv(*ms, np.full(ms[0].shape, float(d)), np.full(ms[0].shape, float(d)))
    v = iv.iadd(*ms, *_const(eps, ms[0].shape))
    r = _rsqrt(*v)
    y = iv.imul(x, x, r[0][:, None] * np.ones_like(x), r[1][:, None] * np.ones_like(x))
    y = iv.imul(*y, *iv.iadd(np.ones_like(s), np.ones_like(s), s, s))
    y = iv.iadd(*y, b, b)
    return y[0].reshape(-1), y[1].reshape(-1)


def _glu(inp):
    a, b = _f64(inp["a"]), _f64(inp["b"])
    g = _gate(a, a)
    ag = iv.imul(a, a, *g)
    return iv.imul(*ag, b, b)


def f_F2(inp, scalars="given"):
    y = _glu(inp)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_F2S(inp, scalars="given"):
    y = _glu(inp)
    t = _scalar(inp["t"], scalars)
    y = iv.imul(*y, *_const(t, y[0].shape))
    return y[0].reshape(-1), y[1].reshape(-1)


def f_F3(inp, scalars="given"):
    x = _f64(inp["x"])
    eps = _scalar(inp["eps"], scalars)
    r, d = x.shape
    nd = np.full(r, float(d))
    mean = iv.idiv(*tsum(x, x, axis=1), nd, nd)
    dev = iv.isub(x, x, mean[0][:, None] * np.ones_like(x), mean[1][:, None] * np.ones_like(x))
    sq = iv.imul(*dev, *dev, same=True)
    var = iv.idiv(*tsum(*sq, axis=1), nd, nd)
    rstd = _rsqrt(*iv.iadd(*var, *_const(eps, var[0].shape)))
    lo = np.stack([mean[0], rstd[0]], axis=1)
    hi = np.stack([mean[1], rstd[1]], axis=1)
    return lo.reshape(-1), hi.reshape(-1)


def f_F4(inp, scalars="given"):
    x, c, s = _f64(inp["x"]), _f64(inp["cos"]), _f64(inp["sin"])
    t = _scalar(inp["t"], scalars)
    d = x.shape[1]
    p = d // 4
    x1, x2, xp = x[:, :p], x[:, p:2 * p], x[:, 2 * p:]
    a = iv.isub(*iv.imul(x1, x1, c, c), *iv.imul(x2, x2, s, s))
    bb = iv.iadd(*iv.imul(x1, x1, s, s), *iv.imul(x2, x2, c, c))
    tt = _const(t, a[0].shape)
    y1, y2 = iv.imul(*tt, *a), iv.imul(*tt, *bb)
    lo = np.concatenate([y1[0], y2[0], xp], axis=1)
    hi = np.concatenate([y1[1], y2[1], xp], axis=1)
    return lo.reshape(-1), hi.reshape(-1)


def f_F5(inp, scalars="given", group=128):
    q = inp["q"].detach().cpu().numpy().astype(np.float64)
    sc = _f64(inp["sc"])
    a = inp["a"].detach().cpu().to(torch.float64).numpy()
    k = q.shape[1]
    scf = np.repeat(sc, group, axis=1)[:, :k]
    t = iv.imul(q, q, scf, scf)
    t = iv.imul(*t, np.broadcast_to(a, q.shape), np.broadcast_to(a, q.shape))
    y = tsum(*t, axis=1)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_F6(inp, scalars="given"):
    x, h, m = _f64(inp["x"]), _f64(inp["h"]), _f64(inp["m"])
    g = _gate(h, h)
    w = iv.imul(m, m, *g)
    num = tsum(*iv.imul(w[0][:, :, None] * np.ones_like(x), w[1][:, :, None] * np.ones_like(x), x, x), axis=1)
    den = tsum(*w, axis=1)
    y = iv.idiv(*num, den[0][:, None] * np.ones_like(num[0]), den[1][:, None] * np.ones_like(num[1]))
    return y[0].reshape(-1), y[1].reshape(-1)


def _log_softmax(p):
    z = p - p.max(axis=1, keepdims=True)  # exact: differences of float32 values are exact in float64
    e_lo, e_hi, _ = iv.elementary_bounds("exp", z, z)
    s = tsum(e_lo, e_hi, axis=1)
    l_lo, l_hi, _ = iv.elementary_bounds("log", *s)
    lp = iv.isub(z, z, l_lo[:, None] * np.ones_like(z), l_hi[:, None] * np.ones_like(z))
    sm = iv.idiv(e_lo, e_hi, s[0][:, None] * np.ones_like(z), s[1][:, None] * np.ones_like(z))
    return lp, sm


def f_F7(inp, scalars="given"):
    p, q = _f64(inp["p"]), _f64(inp["q"])
    lp, sp = _log_softmax(p)
    lq, _ = _log_softmax(q)
    diff = iv.isub(*lp, *lq)
    y = tsum(*iv.imul(*sp, *diff), axis=1)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_F8(inp, scalars="given"):
    pred, tgt, m = _f64(inp["pred"]), _f64(inp["target"]), _f64(inp["m"])
    delta = _scalar(inp["delta"], scalars)
    nvalid = _scalar(inp["nvalid"], scalars)
    d_lo, d_hi = iv.isub(pred, pred, tgt, tgt)
    inside = (np.maximum(np.abs(d_lo), np.abs(d_hi)) <= delta)
    outside = (np.minimum(np.abs(d_lo), np.abs(d_hi)) > delta) & ((d_lo > 0) | (d_hi < 0) | (d_lo >= 0))
    sign = np.where(d_lo >= 0, 1.0, -1.0)  # sign(0) = +1
    g_lo = np.where(inside, d_lo, delta * sign)
    g_hi = np.where(inside, d_hi, delta * sign)
    undecided = ~inside & ~outside
    g_lo = np.where(undecided, np.minimum(d_lo, -delta), g_lo)  # hull of both branches
    g_hi = np.where(undecided, np.maximum(d_hi, delta), g_hi)
    y = iv.imul(g_lo, g_hi, m, m)
    y = iv.idiv(*y, *_const(nvalid, y[0].shape))
    return y[0].reshape(-1), y[1].reshape(-1)


SPEC = {"F1": f_F1, "F2": f_F2, "F2S": f_F2S, "F3": f_F3, "F4": f_F4, "F5": f_F5, "F6": f_F6, "F7": f_F7, "F8": f_F8}
