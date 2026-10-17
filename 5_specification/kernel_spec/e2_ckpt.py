"""Reference for contracts/CKPT.yaml: forward value and gradients of a small program with a checkpointed region,
evaluated as the same program without checkpointing (exact R arithmetic, erf / tanh as rigorous enclosures).

Program: values v0 = x (rows x features); each step appends one value:
  {op: linear, in: i, w: name, b: name | null}   v = v_i W^T + b
  {op: gelu, in: i}                               v = v_i * Phi(v_i), Phi(t) = (1 + erf(t / sqrt 2)) / 2
  {op: tanh, in: i}
  {op: mul, lhs: i, rhs: j}                       element-wise product
The output is the last value; gradients are the VJP against the explicit upstream.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from .num import Invalid, Unsupported, integer, matrix, vector
from .real import R, erf, exp, sqrt, tanh

VARIANTS = ("non_reentrant", "reentrant", "hf_gradient_checkpointing")


def _phi(t):
    return (1 + erf(t / sqrt(R(2)))) / 2


def _dens(t):
    return exp(-(t * t) / 2) / sqrt(R(2) * R(iv.pi * 1))


def _mat(m):
    return [[R(v) for v in row] for row in m]


def evaluate(profile, inp):
    if profile != "identity":
        raise Unsupported(f"profile {profile!r}")
    prog = inp["program"]
    x = _mat(matrix(prog["x"], "x"))
    params = {}
    for name, val in (prog.get("params") or {}).items():
        params[name] = _mat(matrix(val, name)) if isinstance(val[0], list) else [R(v) for v in vector(val, name)]
    steps = prog["steps"]
    if not isinstance(steps, list) or not steps:
        raise Invalid("steps: a non-empty list")
    ck = inp.get("checkpoint") or {}
    variant = ck.get("variant", "non_reentrant")
    if variant not in VARIANTS:
        raise Invalid(f"variant {variant!r}")
    a, b = integer(ck.get("start", 1), "start", 1), integer(ck.get("end", len(steps)), "end", 1)
    if not a <= b <= len(steps):
        raise Invalid(f"checkpoint region [{a}, {b}] outside steps 1..{len(steps)}")
    xg = inp.get("x_requires_grad", True)
    if not isinstance(xg, bool):
        raise Invalid("x_requires_grad: boolean")
    vals = [x]
    req = [xg]
    for k, st in enumerate(steps, 1):
        op = st.get("op")
        refs = [st[f] for f in ("in", "lhs", "rhs") if f in st]
        for r in refs:
            if not isinstance(r, int) or not 0 <= r < k:
                raise Invalid(f"step {k}: reference {r!r} to a value not yet computed")
        if op == "linear":
            v = vals[st["in"]]
            W = params.get(st["w"])
            bias = params.get(st["b"]) if st.get("b") is not None else None
            if W is None or any(len(row) != len(v[0]) for row in W) or (bias is not None and len(bias) != len(W)):
                raise Invalid(f"step {k}: linear shapes do not fit")
            out = [[sum((v[i][j] * W[o][j] for j in range(len(v[0]))), R(0)) + (bias[o] if bias else R(0))
                    for o in range(len(W))] for i in range(len(v))]
            req.append(True)
        elif op in ("gelu", "tanh"):
            if op == "gelu" and st.get("approximate", "none") != "none":
                raise Unsupported("gelu approximate='tanh' is outside this contract's program language")
            f = (lambda t: t * _phi(t)) if op == "gelu" else tanh
            out = [[f(t) for t in row] for row in vals[st["in"]]]
            req.append(req[st["in"]])
        elif op == "mul":
            p, q = vals[st["lhs"]], vals[st["rhs"]]
            if len(p) != len(q) or len(p[0]) != len(q[0]):
                raise Invalid(f"step {k}: mul shapes differ")
            out = [[s * t for s, t in zip(rp, rq)] for rp, rq in zip(p, q)]
            req.append(req[st["lhs"]] or req[st["rhs"]])
        else:
            raise Unsupported(f"op {op!r}")
        vals.append(out)
    if variant == "reentrant":
        ext = {r for st in steps[a - 1:b] for r in [st.get("in"), st.get("lhs"), st.get("rhs")] if isinstance(r, int) and r < a}
        if not any(req[r] for r in ext):
            raise Invalid("reentrant checkpoint: no input of the region requires grad (documented precondition; the "
                          "region would get no gradients)")
    y = vals[-1]
    u = _mat(matrix(inp["upstream"], "upstream", len(y), len(y[0])))
    adj = [None] * len(vals)
    adj[-1] = u
    g = {n: ([[R(0)] * len(p[0]) for _ in p] if isinstance(p[0], list) else [R(0)] * len(p)) for n, p in params.items()}

    def acc(i, m):
        adj[i] = m if adj[i] is None else [[s + t for s, t in zip(r1, r2)] for r1, r2 in zip(adj[i], m)]

    for k in range(len(steps), 0, -1):
        st, d = steps[k - 1], adj[k]
        if d is None:
            continue
        op = st["op"]
        if op == "linear":
            v, W = vals[st["in"]], params[st["w"]]
            if True:
                for o in range(len(W)):
                    for j in range(len(W[0])):
                        g[st["w"]][o][j] = g[st["w"]][o][j] + sum((d[i][o] * v[i][j] for i in range(len(v))), R(0))
                if st.get("b") is not None:
                    for o in range(len(W)):
                        g[st["b"]][o] = g[st["b"]][o] + sum((d[i][o] for i in range(len(v))), R(0))
            acc(st["in"], [[sum((d[i][o] * W[o][j] for o in range(len(W))), R(0)) for j in range(len(W[0]))]
                           for i in range(len(v))])
        elif op == "gelu":
            v = vals[st["in"]]
            acc(st["in"], [[dd * (_phi(t) + t * _dens(t)) for dd, t in zip(rd, rv)] for rd, rv in zip(d, v)])
        elif op == "tanh":
            o = vals[k]
            acc(st["in"], [[dd * (1 - t * t) for dd, t in zip(rd, ro)] for rd, ro in zip(d, o)])
        else:
            p, q = vals[st["lhs"]], vals[st["rhs"]]
            acc(st["lhs"], [[dd * t for dd, t in zip(rd, rq)] for rd, rq in zip(d, q)])
            if True:
                acc(st["rhs"], [[dd * t for dd, t in zip(rd, rp)] for rd, rp in zip(d, p)])
    conv = lambda m: [[t.v for t in row] for row in m] if isinstance(m[0], list) else [t.v for t in m]   # noqa: E731
    grads = {n: conv(m) for n, m in g.items()}
    if xg:
        grads["x"] = conv(adj[0] if adj[0] is not None else [[R(0)] * len(x[0]) for _ in x])
    return {"y": conv(y), "grads": grads}
