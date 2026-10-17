"""Independent implementations used only by the mutants of the group D1 contracts (ATTN, PACK, PAGEDATTN).

Written separately from kernel_spec/d1_attention.py: plain mpmath interval arithmetic, the softmax without the max
shift, explicit per-row loops over a dense allowed matrix, and the backward pass written from the chain rule of the
dense computation. Each misreading named in a contract's mutation requirements is a parameter of this code (never a
switch inside the reference). Results are mpmath intervals (or real.R for -inf).
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from .real import NEG, R

ZERO = iv.mpf(0)


def I(x):
    if isinstance(x, Fraction):
        return iv.mpf(x.numerator) / x.denominator
    if isinstance(x, int):
        return iv.mpf(x)
    return x


def iv_tanh(x):
    return 1 - 2 / (iv.exp(2 * x) + 1)


def iv_attention(q, k, v, *, allowed, bias, scale, softcap=None, softcap_on_raw=False, empty="undefined",
                 dout=None, kv_head=None, bwd_allowed=None, group_sum=True):
    """q [B][Hq][L][E] ... as in d1_attention.attention, but: scale is an interval; kv_head(h) maps query heads to
    key/value heads (default h // (Hq/Hkv)); softcap_on_raw applies the cap to the unscaled dot product and scales
    afterwards; empty in {'undefined', 'zero', 'uniform'} ('uniform': a row without allowed key averages every key);
    bwd_allowed: the allowed predicate the backward pass uses (default: the forward one); group_sum=False keeps only
    the first query head of each group in dk / dv."""
    from .num import Undefined
    B, Hq, L = len(q), len(q[0]), len(q[0][0])
    Hkv, S = len(k[0]), len(k[0][0])
    E, Ev = len(q[0][0][0]), len(v[0][0][0])
    grp = Hq // Hkv
    kv_head = kv_head or (lambda h: h // grp)
    bwd_allowed = bwd_allowed or allowed
    out = [[[[ZERO] * Ev for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    lse = [[[R(NEG)] * L for _ in range(Hq)] for _ in range(B)]
    dq = [[[[ZERO] * E for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    dk = [[[[ZERO] * E for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    dv = [[[[ZERO] * Ev for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]

    def row(b, h, i, pred):
        g = kv_head(h)
        cols, zs, caps = [], [], []
        for j in range(S):
            raw = sum((I(Fraction(q[b][h][i][e])) * I(Fraction(k[b][g][j][e])) for e in range(E)), ZERO)
            if softcap is not None and softcap_on_raw:
                t = iv_tanh(raw / I(softcap))
                s, dsdraw = I(softcap) * t * scale, (1 - t * t) * scale
            elif softcap is not None:
                t = iv_tanh(raw * scale / I(softcap))
                s, dsdraw = I(softcap) * t, (1 - t * t) * scale
            else:
                s, dsdraw = raw * scale, scale
            if not pred(b, h, i, j):
                continue
            bj = bias(b, h, i, j)
            if bj == NEG:
                continue
            cols.append(j)
            zs.append(s + I(bj))
            caps.append(dsdraw)
        return g, cols, zs, caps

    for b in range(B):
        for h in range(Hq):
            for i in range(L):
                g, cols, zs, caps = row(b, h, i, allowed)
                if not cols:
                    if empty == "undefined":
                        raise Undefined("row without an allowed key")
                    if empty == "zero":
                        continue
                    cols, zs, caps = list(range(S)), [ZERO] * S, [ZERO] * S     # uniform average of every key
                w = [iv.exp(z) for z in zs]
                tot = sum(w[1:], w[0])
                p = [x / tot for x in w]
                lse[b][h][i] = iv.log(tot)
                for d in range(Ev):
                    out[b][h][i][d] = sum((pj * I(Fraction(v[b][g][j][d])) for pj, j in zip(p, cols)), ZERO)
                if dout is None:
                    continue
                if bwd_allowed is not allowed:
                    g, cols, zs, caps = row(b, h, i, bwd_allowed)
                    if not cols:
                        continue
                    w = [iv.exp(z) for z in zs]
                    tot = sum(w[1:], w[0])
                    p = [x / tot for x in w]
                do = [I(Fraction(x)) for x in dout[b][h][i]]
                dp = [sum((do[d] * I(Fraction(v[b][g][j][d])) for d in range(Ev)), ZERO) for j in cols]
                D = sum((pj * x for pj, x in zip(p, dp)), ZERO)
                keep = group_sum or h % grp == 0
                for n, j in enumerate(cols):
                    if keep:
                        for d in range(Ev):
                            dv[b][g][j][d] = dv[b][g][j][d] + p[n] * do[d]
                    dz = p[n] * (dp[n] - D) * caps[n]
                    for e in range(E):
                        dq[b][h][i][e] = dq[b][h][i][e] + dz * I(Fraction(k[b][g][j][e]))
                        if keep:
                            dk[b][g][j][e] = dk[b][g][j][e] + dz * I(Fraction(q[b][h][i][e]))
    return {"out": out, "lse": lse, "dq": dq, "dk": dk, "dv": dv}
