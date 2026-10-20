"""Independent interval implementations used only by the ATTN fused-overload mutants (draft path A A2).

Written separately from kernel_spec/attn_aten_semantics.py: plain mpmath interval arithmetic, no max shift, explicit
dense loops. Each misreading named in the contract's mutation requirements is a parameter of this code, never a switch
inside the reference. Input parsing (validation and binding of the ATen arguments) is shared; the arithmetic is not.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

ZERO = iv.mpf(0)


def I(x):
    if isinstance(x, Fraction):
        return iv.mpf(x.numerator) / x.denominator
    if isinstance(x, int):
        return iv.mpf(x)
    if hasattr(x, "v"):          # kernel_spec.real.R
        return I(x.v)
    return x


def _score(q, k, scale):
    return sum((I(a) * I(b) for a, b in zip(q, k)), ZERO) * scale


def iv_forward(q, k, v, *, scale, allowed, kv_head, lse_divisor=None):
    """out [B][Hq][L][Ev] and lse [B][Hq][L] as intervals; lse_divisor divides the natural LSE (log2 misreading)."""
    from .num import Undefined
    B, Hq, L = len(q), len(q[0]), len(q[0][0])
    S, Ev = len(k[0][0]), len(v[0][0][0])
    out = [[[[ZERO] * Ev for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    lse = [[[ZERO] * L for _ in range(Hq)] for _ in range(B)]
    for b in range(B):
        for h in range(Hq):
            g = kv_head(h)
            for i in range(L):
                cols = [j for j in range(S) if allowed(i, j)]
                if not cols:
                    raise Undefined("row without an allowed key")
                w = [iv.exp(_score(q[b][h][i], k[b][g][j], scale)) for j in cols]
                tot = sum(w[1:], w[0])
                lse[b][h][i] = iv.log(tot) if lse_divisor is None else iv.log(tot) / lse_divisor
                for d in range(Ev):
                    out[b][h][i][d] = sum((wj / tot * I(v[b][g][j][d]) for wj, j in zip(w, cols)), ZERO)
    return out, lse


def iv_backward(q, k, v, dout, out, lse, *, scale, allowed, kv_head, Hkv, group_sum=True, saved_lse=True,
                saved_out=True):
    """dq, dk, dv as intervals. saved_lse / saved_out False recompute the softmax normaliser / the output from
    q, k, v instead of using the supplied values; group_sum False keeps only the first query head of each group."""
    B, Hq, L = len(q), len(q[0]), len(q[0][0])
    S, E, Ev = len(k[0][0]), len(q[0][0][0]), len(v[0][0][0])
    grp = Hq // Hkv
    dq = [[[[ZERO] * E for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    dk = [[[[ZERO] * E for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    dv = [[[[ZERO] * Ev for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    for b in range(B):
        for h in range(Hq):
            g = kv_head(h)
            keep = group_sum or h % grp == 0
            for i in range(L):
                cols = [j for j in range(S) if allowed(i, j)]
                s = [_score(q[b][h][i], k[b][g][j], scale) for j in cols]
                if saved_lse:
                    norm = I(lse[b][h][i])
                else:
                    norm = iv.log(sum((iv.exp(x) for x in s), ZERO))
                p = [iv.exp(x - norm) for x in s]
                if saved_out:
                    o = [I(x) for x in out[b][h][i]]
                else:
                    tot = sum(p, ZERO)
                    o = [sum((pj / tot * I(v[b][g][j][d]) for pj, j in zip(p, cols)), ZERO) for d in range(Ev)]
                do = [I(x) for x in dout[b][h][i]]
                D = sum((a * c for a, c in zip(do, o)), ZERO)
                for pj, j in zip(p, cols):
                    dp = sum((do[d] * I(v[b][g][j][d]) for d in range(Ev)), ZERO)
                    ds = pj * (dp - D) * scale
                    for e in range(E):
                        dq[b][h][i][e] = dq[b][h][i][e] + ds * I(k[b][g][j][e])
                        if keep:
                            dk[b][g][j][e] = dk[b][g][j][e] + ds * I(q[b][h][i][e])
                    if keep:
                        for d in range(Ev):
                            dv[b][g][j][d] = dv[b][g][j][d] + pj * do[d]
    return dq, dk, dv
