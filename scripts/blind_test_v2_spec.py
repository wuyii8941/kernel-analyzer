"""Task specifications f of blind_test_v2 (phase 2 release), evaluated independently of every program's TTIR.

Each family's formula is written out on the family's inputs with the interval operations of
reference_eval.intervals (rigorous float64 enclosures, directed rounding):

* sums: the exact sum of each endpoint array (math.fsum, correctly rounded) moved one ulp outward
  (intervals.fsum_bounds); products of float32 / fp16 values are exact in float64 where the bit counts allow,
  and directed otherwise;
* (.)^(-1/2) and sqrt: directed square root (and a directed reciprocal);
* G8: the prefix sums are exact integers (every float32 value is a multiple of 2^-149), divided by D = 2^10.

Runtime scalars are float32-exact (eps = 2^-20, lr = 2^-10, mu = 7/8, wd = 2^-8, scale = 1/8), so the published
value is the value received.  Every function returns (lo, hi) of the output, flattened in row-major order.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch
from gmpy2 import mpq

from kernel_analyzer.reference_eval import intervals as iv


def _f64(t):
    return t.detach().cpu().to(torch.float64).numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float64)


def _c(value, shape):
    """An exact dyadic constant as a point interval."""

    v = float(Fraction(value))
    assert Fraction(v) == Fraction(value), value
    a = np.full(shape, v)
    return a, a


def tsum(lo, hi, axis):
    return iv.fsum_bounds(lo, hi, axis)


def _bcast(x, shape):
    return np.broadcast_to(x, shape)


def _rsqrt(lo, hi):
    s = iv.isqrt(lo, hi)
    one = np.ones_like(s[0])
    return iv.idiv(one, one, *s)


def f_G1(inp):
    x, res, gamma, beta = (_f64(inp[k]) for k in ("x", "res", "gamma", "beta"))
    eps = inp["eps"]
    r, d = x.shape
    m_lo, m_hi = tsum(x, x, axis=1)
    m_lo, m_hi = iv.div_bounds(m_lo, float(d))[0], iv.div_bounds(m_hi, float(d))[1]  # D = 2048: exact
    dev = iv.isub(x, x, _bcast(m_lo[:, None], x.shape), _bcast(m_hi[:, None], x.shape))
    sq = iv.isquare(*dev)
    v_lo, v_hi = tsum(*sq, axis=1)
    v_lo, v_hi = iv.div_bounds(v_lo, float(d))[0], iv.div_bounds(v_hi, float(d))[1]
    rstd = _rsqrt(*iv.iadd(v_lo, v_hi, *_c(eps, v_lo.shape)))
    y = iv.imul(*dev, _bcast(rstd[0][:, None], x.shape), _bcast(rstd[1][:, None], x.shape))
    y = iv.imul(*y, _bcast(gamma, x.shape), _bcast(gamma, x.shape))
    y = iv.iadd(*y, _bcast(beta, x.shape), _bcast(beta, x.shape))
    y = iv.iadd(*y, res, res)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_G2(inp):
    a, b = _f64(inp["a"]), _f64(inp["b"])
    r = np.maximum(a, 0.0)
    r2 = iv.isquare(r, r)
    y = iv.imul(*r2, b, b)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_G3(inp):
    h = _f64(inp["h"])
    one = np.ones_like(h)
    den = iv.iadd(one, one, np.abs(h), np.abs(h))
    t = iv.iadd(one, one, *iv.idiv(h, h, *den))
    s_lo, s_hi = tsum(*t, axis=1)
    y = iv.idiv(*t, _bcast(s_lo[:, None], h.shape), _bcast(s_hi[:, None], h.shape))  # t > 0, s > 0
    return y[0].reshape(-1), y[1].reshape(-1)


def f_G4(inp):
    theta, v, g = _f64(inp["theta"]), _f64(inp["v"]), _f64(inp["g"])
    shape = theta.shape
    vp = iv.iadd(*iv.imul(*_c(inp["mu"], shape), v, v), g, g)
    inner = iv.iadd(*vp, *iv.imul(*_c(inp["wd"], shape), theta, theta))
    step = iv.imul(*_c(inp["lr"], shape), *inner)
    y = iv.isub(theta, theta, *step)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_G5(inp, group=128):
    packed = inp["packed"].detach().cpu().numpy().astype(np.int64)
    sc = _f64(inp["sc"])
    a = inp["a"].detach().cpu().to(torch.float64).numpy()  # fp16 values, exact
    r, half = packed.shape
    q = np.empty((r, 2 * half), dtype=np.float64)
    q[:, 0::2] = (packed & 15) - 8
    q[:, 1::2] = ((packed >> 4) & 15) - 8
    k = q.shape[1]
    scf = np.repeat(sc, group, axis=1)[:, :k]
    t = iv.imul(q, q, scf, scf)
    t = iv.imul(*t, _bcast(a, q.shape), _bcast(a, q.shape))
    y = tsum(*t, axis=1)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_G6(inp):
    a, b = _f64(inp["a"]), _f64(inp["b"])
    ab = tsum(*iv.imul(a, a, b, b), axis=1)
    aa = tsum(*iv.isquare(a, a), axis=1)
    bb = tsum(*iv.isquare(b, b), axis=1)
    den = iv.imul(*iv.isqrt(*aa), *iv.isqrt(*bb))
    y = iv.idiv(*ab, *den)
    return y[0].reshape(-1), y[1].reshape(-1)


def f_G7(inp):
    q, k, v = _f64(inp["q"]), _f64(inp["k"]), _f64(inp["v"])
    pos = inp["pos"].detach().cpu().numpy().astype(np.int64)
    scale = inp["scale"]
    r, length, d = k.shape
    qk = iv.imul(_bcast(q[:, None, :], k.shape), _bcast(q[:, None, :], k.shape), k, k)
    dot = tsum(*qk, axis=2)  # (r, L)
    s = iv.imul(*dot, *_c(scale, dot[0].shape))
    s = (np.maximum(s[0], 0.0), np.maximum(s[1], 0.0))
    mask = np.arange(length)[None, :] <= pos[:, None]
    s = (np.where(mask, s[0], 0.0), np.where(mask, s[1], 0.0))
    sv = iv.imul(_bcast(s[0][:, :, None], v.shape), _bcast(s[1][:, :, None], v.shape), v, v)
    num = tsum(*sv, axis=1)  # (r, d)
    den = tsum(*s, axis=1)
    den = iv.iadd(*den, *_c(1, den[0].shape))
    y = iv.idiv(*num, _bcast(den[0][:, None], num[0].shape), _bcast(den[1][:, None], num[0].shape))
    return y[0].reshape(-1), y[1].reshape(-1)


def f_G8(inp):
    x = inp["x"].detach().cpu().to(torch.float64).numpy()
    r, d = x.shape
    scale = 2 ** 149  # every float32 value is an integer multiple of 2^-149
    lo, hi = np.empty((r, d)), np.empty((r, d))
    den = scale * d
    for i in range(r):
        acc = 0
        for j in range(d):
            acc += int(x[i, j] * 2.0 ** 149)  # exact: power-of-two scaling, integral result
            q = mpq(acc, den)
            lo[i, j], hi[i, j] = iv.rational_down(q), iv.rational_up(q)
    return lo.reshape(-1), hi.reshape(-1)


SPEC = {"G1": f_G1, "G2": f_G2, "G3": f_G3, "G4": f_G4, "G5": f_G5, "G6": f_G6, "G7": f_G7, "G8": f_G8}


def plain(family, inp):
    """The same formulas in plain float64 (a cross-check of the interval code, not an enclosure)."""

    f = lambda k: _f64(inp[k])  # noqa: E731
    if family == "G1":
        x = f("x")
        m = x.mean(1, keepdims=True)
        var = ((x - m) ** 2).mean(1, keepdims=True)
        return (((x - m) / np.sqrt(var + inp["eps"])) * f("gamma") + f("beta") + f("res")).reshape(-1)
    if family == "G2":
        return (np.maximum(f("a"), 0) ** 2 * f("b")).reshape(-1)
    if family == "G3":
        t = 1 + f("h") / (1 + np.abs(f("h")))
        return (t / t.sum(1, keepdims=True)).reshape(-1)
    if family == "G4":
        th = f("theta")
        return (th - inp["lr"] * (inp["mu"] * f("v") + f("g") + inp["wd"] * th)).reshape(-1)
    if family == "G5":
        p = inp["packed"].cpu().numpy().astype(np.int64)
        q = np.empty((p.shape[0], 2 * p.shape[1]))
        q[:, 0::2], q[:, 1::2] = (p & 15) - 8, ((p >> 4) & 15) - 8
        return (q * np.repeat(f("sc"), 128, axis=1) * inp["a"].cpu().double().numpy()).sum(1)
    if family == "G6":
        a, b = f("a"), f("b")
        return (a * b).sum(1) / (np.sqrt((a * a).sum(1)) * np.sqrt((b * b).sum(1)))
    if family == "G7":
        q, k, v = f("q"), f("k"), f("v")
        s = np.maximum(np.einsum("rd,rld->rl", q, k) * inp["scale"], 0)
        s = np.where(np.arange(k.shape[1])[None, :] <= inp["pos"].cpu().numpy()[:, None], s, 0)
        return (np.einsum("rl,rld->rd", s, v) / (s.sum(1, keepdims=True) + 1)).reshape(-1)
    if family == "G8":
        x = f("x")
        return (np.cumsum(x, 1) / x.shape[1]).reshape(-1)
    raise KeyError(family)
