"""Adapter for contracts/SPARSEMAX.yaml: sparsemax along a dimension (Martins & Astudillo 2016) and its VJP.

Input: {x (nested list), dim (int), upstream (shape of x)}. Outputs: y and grad, exact rationals.
The reference follows Proposition 1 / Algorithm 1 (sort, k(z), tau) and the support VJP of Eq. 14; the mutants are an
independent implementation (tau by bisection-free search over candidate supports) with one misreading each.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import c_ce as T
from kernel_spec.num import Invalid, Result, Unsupported, integer, to_iv


def _lines(shape, dim):
    """Index lists (flat C-order positions) of the 1-D slices along dim."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    others = [i for i in range(len(shape)) if i != dim]
    out = []

    def rec(k, base):
        if k == len(others):
            out.append([base + j * strides[dim] for j in range(shape[dim])])
            return
        ax = others[k]
        for v in range(shape[ax]):
            rec(k + 1, base + v * strides[ax])
    rec(0, 0)
    return out


def _inputs(profile, inp):
    if profile != "paper":
        raise Unsupported(f"profile {profile!r}")
    if set(inp) != {"x", "dim", "upstream"}:
        raise Invalid("sparsemax takes exactly x, dim, upstream")
    shape, x = T.tensor(inp["x"], "x")
    if not shape:
        raise Invalid("x must have at least one dimension")
    dim = integer(inp["dim"], "dim")
    if not -len(shape) <= dim < len(shape):
        raise Invalid(f"dim {dim} out of range for rank {len(shape)}")
    us, g = T.tensor(inp["upstream"], "upstream")
    if us != shape:
        raise Invalid("upstream must have the shape of x")
    return shape, x, dim % len(shape), g


def _tau(z):
    zs = sorted(z, reverse=True)
    k, csum, best = 0, Fraction(0), None
    for i, v in enumerate(zs, 1):
        csum += v
        if 1 + i * v > csum:
            k, best = i, csum
    return (best - 1) / k


def reference(profile, inp):
    shape, x, dim, g = _inputs(profile, inp)
    y, gr = [None] * len(x), [None] * len(x)
    for line in _lines(shape, dim):
        z = [x[i] for i in line]
        tau = _tau(z)
        p = [max(v - tau, Fraction(0)) for v in z]
        S = [j for j, v in enumerate(p) if v > 0]
        vhat = sum((g[line[j]] for j in S), Fraction(0)) / len(S)
        for j, i in enumerate(line):
            y[i] = p[j]
            gr[i] = g[i] - vhat if p[j] > 0 else Fraction(0)
    return Result("value", {"y": T.nest(y, shape), "grad": T.nest(gr, shape)})


def _independent(profile, inp, mis=None):
    shape, x, dim, g = _inputs(profile, inp)
    y, gr = [None] * len(x), [None] * len(x)
    for line in _lines(shape, dim):
        z = [x[i] for i in line]
        K = len(z)
        if mis == "softmax":
            e = [iv.exp(to_iv(v)) for v in z]
            s = sum(e[1:], e[0])
            p = [v / s for v in e]
            for j, i in enumerate(line):
                y[i] = p[j]
                gr[i] = p[j] * (to_iv(g[i]) - sum((p[m] * to_iv(g[line[m]]) for m in range(K)), to_iv(0)))
            continue
        # tau: the support is the largest prefix of the sorted values that stays positive after the shift
        order = sorted(range(K), key=lambda j: -z[j])
        tau = None
        for k in range(K, 0, -1):
            top = [z[j] for j in order[:k]]
            t = (sum(top) - 1) / (K if mis == "tau_over_K" else k)
            if all(v - t > 0 for v in top) or k == 1:
                tau = t
                break
        p = [(v - tau) if mis == "no_clip" else max(v - tau, Fraction(0)) for v in z]
        if mis == "support_ge_tau":
            S = [j for j in range(K) if z[j] >= tau]
        elif mis == "mean_over_all":
            S = list(range(K))
        else:
            S = [j for j in range(K) if p[j] > 0]
        vhat = Fraction(0) if mis == "no_mean_subtraction" else sum((g[line[j]] for j in S), Fraction(0)) / len(S)
        for j, i in enumerate(line):
            y[i] = p[j]
            inS = (j in S) if mis != "mean_over_all" else p[j] > 0
            gr[i] = g[i] - vhat if inS else Fraction(0)
    return Result("value", {"y": T.nest(y, shape), "grad": T.nest(gr, shape)})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {"SM.M1": _m("softmax"), "SM.M2": _m("tau_over_K"), "SM.M3": _m("no_mean_subtraction"),
           "SM.M4": _m("mean_over_all"), "SM.M5": _m("support_ge_tau"), "SM.M6": _m("no_clip")}
READINGS = {}
