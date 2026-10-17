"""Adapter for contracts/CKPT.yaml. The reference is kernel_spec.e2_ckpt (the un-checkpointed program in R
arithmetic); the mutants are an independent mpmath-interval evaluator, each switching on one misreading."""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import e2_ckpt
from kernel_spec.expr import iv_erf
from kernel_spec.num import Invalid, Result, to_iv


def reference(profile, inp):
    return Result("value", e2_ckpt.evaluate(profile, inp))


READINGS = {}


def _I(v):
    return to_iv(Fraction(v)) if not isinstance(v, type(iv.mpf(0))) else v


def _independent(profile, inp, mis):
    P = inp["program"]
    x = [[_I(v) for v in r] for r in P["x"]]
    prm = {n: ([[_I(v) for v in r] for r in m] if isinstance(m[0], list) else [_I(v) for v in m])
           for n, m in P["params"].items()}
    steps = P["steps"]
    ck = inp["checkpoint"]
    a, b = ck["start"], ck["end"]
    if b > len(steps):
        if mis != "truncate":
            raise Invalid("region")
        b = len(steps)
    xg = inp.get("x_requires_grad", True)
    if ck["variant"] == "reentrant" and not xg and a == 1 and mis != "ignore_reentrant":
        raise Invalid("reentrant precondition")
    phi = lambda t: (1 + iv_erf(t / iv.sqrt(2))) / 2                       # noqa: E731
    dens = lambda t: iv.exp(-t * t / 2) / iv.sqrt(2 * iv.pi)               # noqa: E731
    V = [x]
    for st in steps:
        if st["op"] == "linear":
            v, W, bb = V[st["in"]], prm[st["w"]], prm[st["b"]]
            V.append([[sum((v[i][j] * W[o][j] for j in range(len(v[0]))), _I(0)) + bb[o] for o in range(len(W))]
                      for i in range(len(v))])
        elif st["op"] == "gelu":
            V.append([[t * phi(t) for t in r] for r in V[st["in"]]])
        elif st["op"] == "tanh":
            V.append([[1 - 2 / (iv.exp(2 * t) + 1) for t in r] for r in V[st["in"]]])
        else:
            V.append([[s * t for s, t in zip(p, q)] for p, q in zip(V[st["lhs"]], V[st["rhs"]])])
    D = [None] * len(V)
    D[-1] = [[_I(v) for v in r] for r in inp["upstream"]]
    G = {n: ([[_I(0)] * len(m[0]) for _ in m] if isinstance(m[0], list) else [_I(0)] * len(m)) for n, m in prm.items()}

    def add(i, m):
        D[i] = m if D[i] is None else [[s + t for s, t in zip(p, q)] for p, q in zip(D[i], m)]
    for k in range(len(steps), 0, -1):
        st, d = steps[k - 1], D[k]
        if d is None:
            continue
        inside = a <= k <= b
        if st["op"] == "linear":
            v, W = V[st["in"]], prm[st["w"]]
            keep = not (mis == "region_detached" and inside) and not (mis == "shared_inside_only" and not inside)
            if keep:
                for o in range(len(W)):
                    for j in range(len(W[0])):
                        G[st["w"]][o][j] += sum((d[i][o] * v[i][j] for i in range(len(v))), _I(0))
                    G[st["b"]][o] += sum((d[i][o] for i in range(len(v))), _I(0))
            if not (mis == "no_input_grad" and k == a):
                add(st["in"], [[sum((d[i][o] * W[o][j] for o in range(len(W))), _I(0)) for j in range(len(W[0]))]
                               for i in range(len(v))])
        elif st["op"] == "gelu":
            add(st["in"], [[dd * (phi(t) + t * dens(t)) for dd, t in zip(p, q)] for p, q in zip(d, V[st["in"]])])
        elif st["op"] == "tanh":
            add(st["in"], [[dd * (1 - t * t) for dd, t in zip(p, q)] for p, q in zip(d, V[k])])
        else:
            add(st["lhs"], [[dd * t for dd, t in zip(p, q)] for p, q in zip(d, V[st["rhs"]])])
            if mis != "reuse_one_path":
                add(st["rhs"], [[dd * t for dd, t in zip(p, q)] for p, q in zip(d, V[st["lhs"]])])
    grads = dict(G)
    if xg:
        grads["x"] = D[0] if D[0] is not None else [[_I(0)] * len(x[0]) for _ in x]
    return Result("value", {"y": V[-1], "grads": grads})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {"CK.M1": _m("region_detached"), "CK.M2": _m("shared_inside_only"), "CK.M3": _m("reuse_one_path"),
           "CK.M4": _m("truncate"), "CK.M5": _m("ignore_reentrant"), "CK.M6": _m("no_input_grad")}
