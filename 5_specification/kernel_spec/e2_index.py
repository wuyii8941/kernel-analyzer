"""Reference for contracts/INDEXREDUCE.yaml: index_add, index_reduce, scatter_reduce (out-of-place values), the
extremum (tie) sets of amax / amin, and the uniquely determined part of the VJP with respect to self and src.

Participants of an output position p: the old self[p] (when include_self) and every src element reduced into p.
Tie sets encode the self participant as the coordinate tuple (-1,) and a src participant by its src coordinates.
"""
from __future__ import annotations

import numpy as np

from .e2_common import flag, is_nan, nested, tensor
from .num import Invalid, Undefined, Unsupported, integer, rational
from .real import R

REDUCES = {"index_add": ("sum",), "index_reduce": ("prod", "mean", "amax", "amin"),
           "scatter_reduce": ("sum", "prod", "mean", "amax", "amin")}
SELF = (-1,)


def _prod(vals):
    out = R(1)
    for v in vals:
        out = out * v
    return out


def evaluate(profile, inp, tie_reading=None):
    if profile != "real":
        raise Unsupported(f"profile {profile!r}")
    op = inp.get("op")
    if op not in REDUCES:
        raise Unsupported(f"operation {op!r}")
    st = tensor(inp["self"], "self", allow_nan=True)
    src = tensor(inp["src"], "src", allow_nan=True)
    nd = st.ndim
    if nd < 1 or src.ndim != nd:
        raise Invalid("self and src must have the same number (>= 1) of dimensions")
    dim = integer(inp["dim"], "dim")
    if not -nd <= dim < nd:
        raise Invalid(f"dim {dim} out of range for {nd} dimensions")
    dim %= nd
    if op == "index_add":
        red, inc, alpha = "sum", True, R(rational(inp.get("alpha", 1), "alpha"))
    else:
        red = inp.get("reduce")
        if red not in REDUCES[op]:
            raise Invalid(f"reduce {red!r} not in {REDUCES[op]}")
        inc, alpha = flag(inp.get("include_self", True), "include_self"), R(1)
    if any(is_nan(v) for v in list(st.reshape(-1)) + list(src.reshape(-1))):
        raise Undefined("NaN input: outside the real-number profile (IR.NAN)")
    size = st.shape[dim]
    if op in ("index_add", "index_reduce"):
        idx = tensor(inp["index"], "index", ndim=1, item=lambda x, w: integer(x, w))
        if src.shape[dim] != len(idx) or any(src.shape[d] != st.shape[d] for d in range(nd) if d != dim):
            raise Invalid(f"source shape {src.shape} does not fit self {st.shape}, dim {dim}, len(index) {len(idx)}")
        target = lambda q: q[:dim] + (int(idx[q[dim]]),) + q[dim + 1:]          # noqa: E731
        flat = list(idx)
    else:
        idx = tensor(inp["index"], "index", ndim=nd, item=lambda x, w: integer(x, w))
        if idx.shape != src.shape:
            if all(idx.shape[d] <= src.shape[d] for d in range(nd)):
                raise Unsupported("scatter_reduce with index smaller than src (the documented backward needs equal shapes)")
            raise Invalid("index.size(d) <= src.size(d) is required")
        if any(idx.shape[d] > st.shape[d] for d in range(nd) if d != dim):
            raise Invalid("index.size(d) <= self.size(d) for d != dim is required")
        target = lambda q: q[:dim] + (int(idx[q]),) + q[dim + 1:]               # noqa: E731
        flat = list(idx.reshape(-1))
    for j in flat:
        if not 0 <= j < size:
            raise Invalid(f"index {j} outside [0, {size})")
    if "upstream" not in inp:
        raise Invalid("upstream (cotangent of the output) is required")
    u = tensor(inp["upstream"], "upstream")
    if u.shape != st.shape or any(not v.finite for v in u.reshape(-1)):
        raise Invalid(f"upstream must be finite with self's shape {st.shape}")
    groups = {}
    for q in np.ndindex(src.shape):
        groups.setdefault(target(q), []).append(q)
    out = st.copy()
    gs = u.copy()
    gq = np.empty(src.shape, dtype=object)
    for q in np.ndindex(src.shape):
        gq[q] = R(0)
    ties = np.empty(st.shape, dtype=object)
    for p in np.ndindex(st.shape):
        ties[p] = frozenset()
    for p, qs in groups.items():
        parts = ([(SELF, st[p])] if inc else []) + [(q, alpha * src[q]) for q in qs]
        vals = [v for _, v in parts]
        g = u[p]
        grads = {}
        if red == "sum":
            out[p] = sum(vals[1:], vals[0])
            grads = {key: g for key, _ in parts}
        elif red == "prod":
            out[p] = _prod(vals)
            grads = {key: g * _prod([v for j, (_, v) in enumerate(parts) if j != i]) for i, (key, _) in enumerate(parts)}
        elif red == "mean":
            out[p] = sum(vals[1:], vals[0]) / len(vals)
            grads = {key: g / len(vals) for key, _ in parts}
        else:
            best = vals[0]
            for v in vals[1:]:
                if (v > best) if red == "amax" else (v < best):
                    best = v
            S = [key for key, v in parts if v.same(best)]
            out[p] = best
            ties[p] = frozenset(S)
            if len(S) == 1:
                grads = {S[0]: g}
            elif tie_reading == "even":
                grads = {key: g / len(S) for key in S}
        gs[p] = grads.get(SELF, R(0))
        for q in qs:
            gq[q] = gq[q] + grads.get(q, R(0)) * alpha
    fields = {"out": nested(out), "grad_self": nested(gs), "grad_src": nested(gq)}
    fields["ties"] = nested(ties, lambda a: a) if red in ("amax", "amin") else []
    return fields
