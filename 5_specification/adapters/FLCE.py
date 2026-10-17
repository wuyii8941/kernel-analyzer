"""Adapter for contracts/FLCE.yaml: fused linear cross entropy CE(x W^T + b) and its VJP w.r.t. x, W, b.

Input: {hidden (N, H), weight (V, H), bias [V] or null, target [N] class indices, ce_weight [V] or null, eps,
ignore_index, reduction, upstream (scalar for sum / mean, [N] for none)}. Outputs: loss; grad = {hidden, weight,
bias (when given)}.

The reference forms the logits exactly, takes the cross entropy and its logit gradient from the v2.4 phase1 reference
(the pilot's numerics, kernel_spec.c_ce) and chains the gradient through the linear map in interval arithmetic. The
mutants are an independent mpmath implementation with one misreading switched on each.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import c_ce as K
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, matrix, rational, to_iv, vector

PROFILES = ("index_unsmoothed", "index_smoothing_unweighted", "index_smoothing_R_A")
KEYS = {"hidden", "weight", "bias", "target", "ce_weight", "eps", "ignore_index", "reduction", "upstream"}


def _inputs(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    if set(inp) != KEYS:
        raise Invalid(f"FLCE takes exactly {sorted(KEYS)}, got {sorted(inp)}")
    x = matrix(inp["hidden"], "hidden")
    N, H = len(x), len(x[0])
    W = matrix(inp["weight"], "weight", cols=H)
    V = len(W)
    b = None if inp["bias"] is None else vector(inp["bias"], "bias", V)
    cw = None if inp["ce_weight"] is None else vector(inp["ce_weight"], "ce_weight", V)
    if cw is not None and any(v < 0 for v in cw):
        raise Unsupported("negative class weights")
    if V < 2:
        raise Unsupported("V = 1 (one class) is outside these profiles")
    eps = K.check_eps(inp["eps"])
    red = K.check_reduction(inp["reduction"])
    ii = integer(inp["ignore_index"], "ignore_index")
    y = vector(inp["target"], "target", N, item=integer)
    for t in y:
        if t != ii and not 0 <= t < V:
            raise Invalid(f"target {t} outside [0, {V}) and not the ignore_index")
    if profile == "index_unsmoothed" and eps != 0:
        raise Invalid("index_unsmoothed requires eps = 0")
    if profile == "index_smoothing_unweighted" and cw is not None:
        raise Invalid("index_smoothing_unweighted takes no ce_weight")
    if profile == "index_smoothing_R_A" and cw is None:
        raise Invalid("index_smoothing_R_A needs ce_weight")
    u = inp["upstream"]
    if red == "none":
        u = vector(u, "upstream", N)
    else:
        if isinstance(u, (list, tuple)):
            raise Invalid("upstream must be a scalar for sum / mean")
        u = rational(u, "upstream")
    return x, W, b, y, cw, eps, ii, red, u, N, H, V


def _logits(x, W, b):
    return [[sum((xi * wi for xi, wi in zip(row, wrow)), Fraction(0)) + (b[v] if b is not None else 0)
             for v, wrow in enumerate(W)] for row in x]


def reference(profile, inp):
    x, W, b, y, cw, eps, ii, red, u, N, H, V = _inputs(profile, inp)
    z = _logits(x, W, b)
    loss, g = K.ce_rows(z, y, cw, eps, ii, red, False)
    rows_u = u if red == "none" else [u] * N
    g = [[to_iv(K.scale(v, rows_u[n])) for v in row] for n, row in enumerate(g)]
    dx = [[K.iv_sum(g[n][v] * to_iv(W[v][h]) for v in range(V)) for h in range(H)] for n in range(N)]
    dW = [[K.iv_sum(g[n][v] * to_iv(x[n][h]) for n in range(N)) for h in range(H)] for v in range(V)]
    grad = {"hidden": dx, "weight": dW}
    if b is not None:
        grad["bias"] = [K.iv_sum(g[n][v] for n in range(N)) for v in range(V)]
    return Result("value", {"loss": loss, "grad": grad})


# ------------------------------------------------------------------ independent implementation for the mutants
def _independent(profile, inp, mis=None):
    x, W, b, y, cw, eps, ii, red, u, N, H, V = _inputs(profile, inp)
    w = cw or [Fraction(1)] * V
    Z = [[sum((to_iv(a) * to_iv(c) for a, c in zip(row, wrow)), to_iv(0)) + to_iv(b[v] if b is not None else 0)
          for v, wrow in enumerate(W)] for row in x]
    losses, G, mass = [], [], []
    for n in range(N):
        if y[n] == ii:
            losses.append(to_iv(0)), G.append([to_iv(0)] * V), mass.append(Fraction(0))
            continue
        e = [iv.exp(zv) for zv in Z[n]]
        s = sum(e[1:], e[0])
        p = [v / s for v in e]
        a = [w[c] * ((1 - eps) * (c == y[n]) + eps / V) for c in range(V)]
        if mis == "weighted_smoothing_R_B":
            a = [w[y[n]] * ((1 - eps) * (c == y[n]) + eps / V) for c in range(V)]
        A = sum(a)
        losses.append(-sum((to_iv(a[c]) * iv.log(p[c]) for c in range(V)), to_iv(0)))
        G.append([p[c] * to_iv(A) - to_iv(a[c]) for c in range(V)])
        mass.append(w[y[n]])
    if red == "none":
        if mis == "none_first_upstream_only":
            up = [u[0]] * N
        elif mis == "upstream_ignored":
            up = [Fraction(1)] * N
        else:
            up = u
        G = [[v * to_iv(up[n]) for v in row] for n, row in enumerate(G)]
        loss = losses
    else:
        tot = sum(losses[1:], losses[0])
        den = Fraction(1)
        if red == "mean":
            den = Fraction(N) if mis == "mean_by_N" else sum(mass)
            if den == 0:
                raise Undefined("zero mass")
        loss = tot / to_iv(den)
        G = [[v * to_iv(u) / to_iv(den) for v in row] for row in G]
    if mis == "ignored_rows_in_grad_weight":
        # ignored rows contribute their unignored-CE gradient to dW
        for n in range(N):
            if y[n] == ii:
                e = [iv.exp(zv) for zv in Z[n]]
                s = sum(e[1:], e[0])
                G[n] = [v / s for v in e]
    if mis == "dx_with_W_transposed":                      # dx = G W^T (dimensionally valid only when V == H)
        if V != H:
            raise Undefined("G W^T needs V == H")
        dx = [[sum((G[n][v] * to_iv(W[h][v]) for v in range(V)), to_iv(0)) for h in range(H)] for n in range(N)]
    else:
        dx = [[sum((G[n][v] * to_iv(W[v][h]) for v in range(V)), to_iv(0)) for h in range(H)] for n in range(N)]
    if mis == "dW_missing_upstream" and red != "none":
        Gw = [[v / to_iv(u) for v in row] for row in G]
    else:
        Gw = G
    dW = [[sum((Gw[n][v] * to_iv(x[n][h]) for n in range(N)), to_iv(0)) for h in range(H)] for v in range(V)]
    grad = {"hidden": dx, "weight": dW}
    if b is not None:
        if mis == "bias_grad_first_row":
            grad["bias"] = [G[0][v] for v in range(V)]
        else:
            grad["bias"] = [sum((G[n][v] for n in range(N)), to_iv(0)) for v in range(V)]
    return Result("value", {"loss": loss, "grad": grad})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "FL.M1": _m("mean_by_N"),
    "FL.M2": _m("weighted_smoothing_R_B"),
    "FL.M3": _m("none_first_upstream_only"),
    "FL.M4": _m("upstream_ignored"),
    "FL.M5": _m("ignored_rows_in_grad_weight"),
    "FL.M6": _m("dx_with_W_transposed"),
    "FL.M7": _m("dW_missing_upstream"),
    "FL.M8": _m("bias_grad_first_row"),
}
READINGS = {}
