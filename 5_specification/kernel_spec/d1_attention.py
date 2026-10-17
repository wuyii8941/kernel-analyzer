"""Real-number attention semantics shared by the group D1 contracts ATTN, PACK and PAGEDATTN.

One function, `attention`, evaluates softmax attention on exact rational inputs with kernel_spec.real.R values:
scores s = scale * q.k (exact), optional softcap c * tanh(s / c), an additive bias, a structural mask (boolean mask,
causal, window), softmax over the allowed keys of each query row, out = P v, the natural log-sum-exp of the row, and
the vector-Jacobian product (dq, dk, dv) for an explicit upstream gradient of out. Key/value head g serves the query
heads h with h // (Hq / Hkv) == g (repeat_interleave). Which keys are allowed, the alignment of causal masks and windows,
and what happens to a row without any allowed key are decided by the calling adapter (the contracts' profiles).

Inputs are validated by the helpers below before any arithmetic: nested lists must be rectangular with the stated rank,
values must be exact rationals (strings 'p/q', integers); '-inf' is accepted only where a caller allows it (additive
masks). Nothing is truncated or broadcast silently: broadcasting of masks follows the documented rule and is checked.
"""
from __future__ import annotations

from fractions import Fraction

from . import real as rl
from .num import Invalid, Undefined, integer, rational
from .real import NEG, POS, R

# ------------------------------------------------------------------ input helpers


def shape_of(x, what):
    """Shape of a rectangular nested list (scalars have shape ()); ragged lists are invalid."""
    if not isinstance(x, (list, tuple)):
        return ()
    if not x:
        return (0,)
    subs = [shape_of(e, what) for e in x]
    if any(s != subs[0] for s in subs):
        raise Invalid(f"{what}: ragged nested list")
    return (len(x),) + subs[0]


def _leaf(v, what, allow_inf=False, allow_bool=False):
    if isinstance(v, bool):
        if allow_bool:
            return v
        raise Invalid(f"{what}: boolean where a number is required")
    if isinstance(v, str) and v.strip() in ("inf", "+inf", "-inf"):
        if not allow_inf:
            raise Invalid(f"{what}: infinity not allowed here")
        return NEG if v.strip() == "-inf" else POS
    return rational(v, what)


def tensor(x, rank, what, allow_inf=False):
    """Nested list of Fractions with exactly `rank` dimensions."""
    shp = shape_of(x, what)
    if len(shp) != rank:
        raise Invalid(f"{what}: rank {len(shp)}, expected {rank}")

    def conv(e, depth):
        if depth == rank:
            return _leaf(e, what, allow_inf)
        return [conv(s, depth + 1) for s in e]
    return conv(x, 0), shp


def mask_tensor(x, what):
    """A mask: all booleans (boolean mask) or all numbers / -inf / +inf (additive). Returns (values, shape, is_bool)."""
    shp = shape_of(x, what)
    flat = []

    def walk(e, depth):
        if depth == len(shp):
            flat.append(e)
        else:
            for s in e:
                walk(s, depth + 1)
    walk(x, 0)
    kinds = {isinstance(v, bool) for v in flat}
    if len(kinds) > 1:
        raise Invalid(f"{what}: mixes booleans and numbers")
    is_bool = kinds == {True}

    def conv(e, depth):
        if depth == len(shp):
            return e if is_bool else _leaf(e, what, allow_inf=True)
        return [conv(s, depth + 1) for s in e]
    return conv(x, 0), shp, is_bool


def broadcast_get(values, shp, target):
    """Accessor idx -> element for `values` broadcast to `target` (numpy rule: align right, size 1 or equal)."""
    if len(shp) > len(target):
        raise Invalid(f"mask of rank {len(shp)} does not broadcast to {target}")
    off = len(target) - len(shp)
    for a, b in zip(shp, target[off:]):
        if a not in (1, b):
            raise Invalid(f"mask shape {shp} does not broadcast to {target}")

    def get(idx):
        e = values
        for d, s in enumerate(shp):
            e = e[0 if s == 1 else idx[off + d]]
        return e
    return get


def window_pair(w, what="window"):
    if not isinstance(w, (list, tuple)) or len(w) != 2:
        raise Invalid(f"{what}: expected [left, right]")
    left, right = integer(w[0], f"{what}[0]"), integer(w[1], f"{what}[1]")
    if left < -1 or right < -1:
        raise Invalid(f"{what}: entries must be >= -1 (-1 = unbounded)")
    return left, right


def in_window(i_pos, j, left, right):
    """FlashAttention window convention around the aligned query position: pos - left <= j <= pos + right; -1 unbounded."""
    if left >= 0 and j < i_pos - left:
        return False
    if right >= 0 and j > i_pos + right:
        return False
    return True


# ------------------------------------------------------------------ the semantics


def _dot(a, b):
    s = Fraction(0)
    for x, y in zip(a, b):
        s += x * y
    return s


def attention(q, k, v, *, allowed, bias, scale, softcap=None, empty="undefined", dout=None):
    """q [B][Hq][L][E], k [B][Hkv][S][E], v [B][Hkv][S][Ev] (Fractions); allowed(b, h, i, j) -> bool;
    bias(b, h, i, j) -> Fraction | NEG | POS; scale: R; softcap: R or None; empty: 'undefined' (a row without an
    allowed key raises Undefined) or 'zero' (that row gives out 0, lse -inf and no gradient contribution);
    dout [B][Hq][L][Ev] or None. Returns dict(out, lse, dq, dk, dv) of nested lists of R (gradients None without
    dout)."""
    B, Hq, L = len(q), len(q[0]), len(q[0][0])
    Hkv, S = len(k[0]), len(k[0][0])
    Ev = len(v[0][0][0]) if S else 0
    E = len(q[0][0][0]) if L else 0
    group = Hq // Hkv
    zero = R(0)
    out = [[[[zero] * Ev for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    lse = [[[R(NEG)] * L for _ in range(Hq)] for _ in range(B)]
    grads = dout is not None
    if grads:
        dq = [[[[zero] * E for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
        dk = [[[[zero] * E for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
        dv = [[[[zero] * Ev for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    for b in range(B):
        for h in range(Hq):
            g = h // group
            for i in range(L):
                keys, z, tanh_t = [], [], []
                for j in range(S):
                    if not allowed(b, h, i, j):
                        continue
                    bj = bias(b, h, i, j)
                    if bj == NEG:
                        continue
                    if bj == POS:
                        raise Undefined("+inf in the additive mask: the softmax of that row has no real value")
                    s = scale * R(_dot(q[b][h][i], k[b][g][j]))
                    t = None
                    if softcap is not None:
                        t = rl.tanh(s / softcap)
                        s = softcap * t
                    keys.append(j)
                    z.append(s + R(bj))
                    tanh_t.append(t)
                if not keys:
                    if empty == "zero":
                        continue                      # out row 0, lse -inf, no gradient contribution
                    raise Undefined(f"query row (batch {b}, head {h}, position {i}) has no allowed key")
                m = z[0]
                for x in z[1:]:
                    m = rl.max_(m, x)
                ex = [rl.exp(x - m) for x in z]
                tot = ex[0]
                for x in ex[1:]:
                    tot = tot + x
                p = [x / tot for x in ex]
                lse[b][h][i] = m + rl.log(tot)
                for d in range(Ev):
                    acc = zero
                    for pj, j in zip(p, keys):
                        acc = acc + pj * R(v[b][g][j][d])
                    out[b][h][i][d] = acc
                if not grads:
                    continue
                do = dout[b][h][i]
                dp = [R(_dot(do, v[b][g][j])) for j in keys]
                D = zero
                for pj, dpj in zip(p, dp):
                    D = D + pj * dpj
                for n, j in enumerate(keys):
                    pj = p[n]
                    for d in range(Ev):
                        dv[b][g][j][d] = dv[b][g][j][d] + pj * R(do[d])
                    ds = pj * (dp[n] - D)
                    if tanh_t[n] is not None:
                        ds = ds * (R(1) - tanh_t[n] * tanh_t[n])
                    ds = ds * scale
                    for e in range(E):
                        dq[b][h][i][e] = dq[b][h][i][e] + ds * R(k[b][g][j][e])
                        dk[b][g][j][e] = dk[b][g][j][e] + ds * R(q[b][h][i][e])
    res = {"out": out, "lse": lse}
    if grads:
        res.update(dq=dq, dk=dk, dv=dv)
    else:
        res.update(dq=None, dk=None, dv=None)
    return res


def default_scale(E):
    """1 / sqrt(E) as an R (exact when E is a perfect square)."""
    if E <= 0:
        raise Invalid("head dimension E must be positive for the default scale")
    return R(1) / rl.sqrt(R(E))


def values(x):
    """Nested lists of R -> nested lists of the plain values the checker compares (Fraction / IV / infinity)."""
    if isinstance(x, list):
        return [values(e) for e in x]
    return x.v if isinstance(x, R) else x
