"""Pre-registered conditions of phase 1 of the essential-bug round (docs/protocol_essential_bugs_20261007.md).

Each condition is a dict with an ``id``; ``make_inputs(cond, seed)`` returns the inputs for one seed (seeds 0, 1, 2)
as Python / NumPy float64 values from a generator seeded by (condition id, seed).  Candidates round these values to
their own dtype; the specification is then evaluated on the values the candidate actually received.

    python scripts/essential/conditions.py          # prints the counts per operator family
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np

SEEDS = (0, 1, 2)


def _rng(cond_id, seed):
    h = int.from_bytes(hashlib.sha256(f"{cond_id}/{seed}".encode()).digest()[:8], "little")
    return np.random.default_rng(h)


# ---------------------------------------------------------------------------------------------- cross-entropy

def ce_conditions():
    out = []
    for (N, C), ign, red, w, eps in itertools.product(((4, 5), (7, 1000), (3, 4099)),
                                                      ("none", "partial", "all", "head_tail"),
                                                      ("none", "sum", "mean"), ("none", "ints", "zeros"), (0.0, 0.1)):
        out.append(dict(op="ce", N=N, C=C, ignore=ign, ignore_index=-100, reduction=red, weight=w, eps=eps,
                        logits="gauss", target="index", layout="contiguous"))
    for red, eps in itertools.product(("none", "sum", "mean"), (0.0, 0.1)):   # ignore_index inside the class range
        out.append(dict(op="ce", N=4, C=5, ignore="in_range", ignore_index=0, reduction=red, weight="none", eps=eps,
                        logits="gauss", target="index", layout="contiguous"))
    out.append(dict(op="ce", N=4, C=5, ignore="partial", ignore_index=-1, reduction="mean", weight="none", eps=0.0,
                    logits="gauss", target="index", layout="contiguous"))
    for lg, red, eps in itertools.product(("large", "huge", "equal"), ("none", "mean"), (0.0, 0.1)):
        out.append(dict(op="ce", N=4, C=5, ignore="partial", ignore_index=-100, reduction=red, weight="none", eps=eps,
                        logits=lg, target="index", layout="contiguous"))
    for red in ("none", "sum", "mean"):                                        # non-contiguous logits
        out.append(dict(op="ce", N=7, C=1000, ignore="partial", ignore_index=-100, reduction=red, weight="ints",
                        eps=0.1, logits="gauss", target="index", layout="transposed"))
    for (N, C), red, eps in itertools.product(((1, 1), (1, 5)), ("none", "sum", "mean"), (0.0, 0.1)):  # size-1 dims
        out.append(dict(op="ce", N=N, C=C, ignore="none", ignore_index=-100, reduction=red, weight="none", eps=eps,
                        logits="gauss", target="index", layout="contiguous"))
    for (N, C), red, w, eps in itertools.product(((4, 5), (7, 1000)), ("none", "sum", "mean"), ("none", "ints"),
                                                 (0.0, 0.1)):
        out.append(dict(op="ce", N=N, C=C, ignore="none", ignore_index=-100, reduction=red, weight=w, eps=eps,
                        logits="gauss", target="prob", layout="contiguous"))
    for i, c in enumerate(out):
        c["id"] = f"ce{i:03d}"
    return out


def flce_conditions():
    out = []
    for (N, V), ign, red, eps, w in itertools.product(((4, 1000), (3, 4099)), ("none", "partial", "all"),
                                                      ("none", "sum", "mean"), (0.0, 0.1), ("none", "ints")):
        out.append(dict(op="flce", N=N, V=V, H=64, ignore=ign, ignore_index=-100, reduction=red, eps=eps, weight=w))
    for i, c in enumerate(out):
        c["id"] = f"flce{i:03d}"
    return out


def _ce_targets(rng, c, N, C):
    ii = c["ignore_index"]
    t = rng.integers(0, C, N)
    if c["ignore"] == "partial":
        mask = rng.random(N) < 0.3
        if N >= 2:                         # at least one ignored row and one kept row
            i, j = rng.choice(N, 2, replace=False)
            mask[i], mask[j] = True, False
        t = np.where(mask, ii, t)
    elif c["ignore"] == "all":
        t[:] = ii
    elif c["ignore"] == "head_tail":
        t[0] = ii
        t[-1] = ii
    elif c["ignore"] == "in_range":       # ignore_index = 0 is a class; make sure it occurs and something else does
        t[0] = 0
        t[-1] = rng.integers(1, C)
    return [int(v) for v in t]


def _ce_weights(rng, c, C, targets):
    if c["weight"] == "none":
        return None
    if c["weight"] == "ints":
        return [float(v) for v in rng.integers(1, 6, C)]
    w = rng.choice([0.0, 1.0, 3.0], C)
    present = [t for t in targets if 0 <= t < C]
    if present:                            # a zero weight on a class that is actually a target
        w[present[0]] = 0.0
    return [float(v) for v in w]


def _logits(rng, kind, N, C):
    x = rng.normal(0.0, 2.0, (N, C))
    if kind == "large":
        x = x * 1e4
    elif kind == "huge":
        x[np.arange(N), rng.integers(0, C, N)] = 1e30
    elif kind == "equal":
        x = np.repeat(rng.normal(0.0, 2.0, (N, 1)), C, axis=1)
    return x


def make_ce_inputs(c, seed):
    rng = _rng(c["id"], seed)
    N, C = c["N"], c["C"]
    logits = _logits(rng, c["logits"], N, C)
    if c["target"] == "prob":
        raw = rng.random((N, C)) + 1e-3
        target = raw / raw.sum(1, keepdims=True)            # float64 probabilities (rows sum to 1 up to rounding)
        weights = _ce_weights(rng, c, C, [])
    else:
        target = _ce_targets(rng, c, N, C)
        weights = _ce_weights(rng, c, C, target)
    return dict(logits=logits, target=target, weights=weights)


def make_flce_inputs(c, seed):
    rng = _rng(c["id"], seed)
    N, V, H = c["N"], c["V"], c["H"]
    hidden = rng.normal(0.0, 1.0, (N, H))
    weight = rng.normal(0.0, 0.1, (V, H))
    target = _ce_targets(rng, c, N, V)
    w = _ce_weights(rng, c, V, target)
    return dict(hidden=hidden, weight=weight, target=target, weights=w)


# ---------------------------------------------------------------------------------------------- pooling

def pool_conditions():
    out = []
    for L, k, s, p, ceil, cip, vals in itertools.product((4, 7), (2, 3), (1, 2, 3), (0, 1), (False, True),
                                                         (True, False), ("ints", "gauss")):
        out.append(dict(op="avg_pool", nd=1, size=(L,), kernel=(k,), stride=(s,), padding=(p,), ceil_mode=ceil,
                        count_include_pad=cip, divisor_override=None, values=vals))
    for k, s, p, ceil, cip, div, vals in itertools.product(((2, 2), (3, 2)), ((2, 2), (1, 3)), (0, 1), (False, True),
                                                           (True, False), (None, 5), ("ints", "gauss")):
        out.append(dict(op="avg_pool", nd=2, size=(5, 6), kernel=k, stride=s, padding=(p, p), ceil_mode=ceil,
                        count_include_pad=cip, divisor_override=div, values=vals))
    for k, p, ceil, cip, div, vals in itertools.product(((2, 2, 2), (3, 2, 2)), (0, 1), (False, True), (True, False),
                                                        (None, 7), ("ints", "gauss")):
        out.append(dict(op="avg_pool", nd=3, size=(4, 5, 3), kernel=k, stride=(2, 2, 2), padding=(p, p, p),
                        ceil_mode=ceil, count_include_pad=cip, divisor_override=div, values=vals))
    for L, k, s, p, d, ceil, vals in itertools.product((4, 7), (2, 3), (1, 2, 3), (0, 1), (1, 2), (False, True),
                                                       ("ties", "gauss", "special")):
        if 2 * p > d * (k - 1) + 1 or L + 2 * p < d * (k - 1) + 1:
            continue
        out.append(dict(op="max_pool", nd=1, size=(L,), kernel=(k,), stride=(s,), padding=(p,), dilation=(d,),
                        ceil_mode=ceil, values=vals))
    for vals in ("ties", "gauss"):          # a legal window that samples only padding (spec: -inf, empty argmax set)
        out.append(dict(op="max_pool", nd=1, size=(1,), kernel=(2,), stride=(1,), padding=(1,), dilation=(2,),
                        ceil_mode=False, values=vals))
    for k, s, p, d, ceil, vals in itertools.product(((2, 2), (3, 2)), ((2, 2), (1, 1)), (0, 1), (1, 2), (False, True),
                                                    ("ties", "gauss", "special")):
        if any(2 * p > d * (kk - 1) + 1 for kk in k):
            continue
        out.append(dict(op="max_pool", nd=2, size=(5, 6), kernel=k, stride=s, padding=(p, p), dilation=(d, d),
                        ceil_mode=ceil, values=vals))
    for s, p, ceil, vals in itertools.product(((1, 1, 1), (2, 2, 2)), (0, 1), (False, True), ("ties", "gauss", "special")):
        out.append(dict(op="max_pool", nd=3, size=(4, 5, 3), kernel=(2, 2, 2), stride=s, padding=(p, p, p),
                        dilation=(1, 1, 1), ceil_mode=ceil, values=vals))
    for i, c in enumerate(out):
        c["id"] = f"pool{i:03d}"
    return out


def make_pool_inputs(c, seed):
    rng = _rng(c["id"], seed)
    shape = (2,) + tuple(c["size"])                    # 2 channels, no batch dimension in the spec
    v = c["values"]
    if v == "ints":
        x = rng.integers(-5, 6, shape).astype(np.float64)
    elif v == "ties":
        x = rng.integers(-2, 3, shape).astype(np.float64)
    else:
        x = rng.normal(0.0, 1.0, shape)
    if v == "special":                                 # one NaN, one +inf, one -inf at random positions
        flat = x.reshape(-1)
        pos = rng.choice(flat.size, size=min(3, flat.size), replace=False)
        for j, val in zip(pos, (np.nan, np.inf, -np.inf)):
            flat[j] = val
    return dict(x=x, upstream_seed=int(rng.integers(0, 2 ** 31)))


# ---------------------------------------------------------------------------------------------- index / scatter

_SHAPES = {"1d": ((5,), 0), "1d_single": ((1,), 0), "2d_dim0": ((4, 3), 0), "2d_dim1": ((3, 4), 1)}


def index_conditions():
    out = []
    for shape, m, dup, alpha, vals in itertools.product(_SHAPES, (3, 8), ("some", "all_to_one"), (1.0, 2.5),
                                                        ("ints", "gauss")):
        if shape == "1d_single" and dup == "some":
            continue
        out.append(dict(op="index_add", shape=shape, m=m, dup=dup, alpha=alpha, values=vals))
    for red, inc, shape, m, dup, vals in itertools.product(("prod", "mean", "amax", "amin"), (True, False), _SHAPES,
                                                           (3, 8), ("some", "all_to_one"), ("ints", "special_ints", "gauss")):
        if shape == "1d_single" and dup == "some":
            continue
        out.append(dict(op="index_reduce", reduce=red, include_self=inc, shape=shape, m=m, dup=dup, values=vals))
    for red, inc, shape, dup, vals in itertools.product(("sum", "prod", "mean", "amax", "amin"), (True, False), _SHAPES,
                                                        ("some", "all_to_one"), ("ints", "special_ints", "gauss")):
        if shape == "1d_single" and dup == "some":
            continue
        out.append(dict(op="scatter_reduce", reduce=red, include_self=inc, shape=shape, dup=dup, values=vals))
    for i, c in enumerate(out):
        c["id"] = f"idx{i:03d}"
    return out


def _values(rng, kind, shape, red):
    if kind == "gauss":
        return rng.normal(0.0, 1.0, shape)
    if kind == "ints":
        return rng.integers(-3, 4, shape).astype(np.float64)
    # special_ints: zeros for prod (one or two per array), ties for amax/amin, plain small ints otherwise
    if red == "prod":
        v = rng.choice([-2.0, -1.0, 1.0, 2.0, 3.0], shape)
        flat = v.reshape(-1)
        flat[rng.choice(flat.size, size=min(2, flat.size), replace=False)] = 0.0
        return v
    return rng.integers(-1, 2, shape).astype(np.float64)


def _index_vec(rng, dup, size, m):
    if dup == "all_to_one":
        return [int(rng.integers(0, size))] * m
    idx = rng.integers(0, size, m)
    if m >= 2:
        idx[1] = idx[0]                                # at least one duplicate
    return [int(v) for v in idx]


def make_index_inputs(c, seed):
    rng = _rng(c["id"], seed)
    shape, dim = _SHAPES[c["shape"]]
    red = c.get("reduce", "sum")
    self_t = _values(rng, c["values"], shape, red)
    size = shape[dim]
    if c["op"] in ("index_add", "index_reduce"):
        m = c["m"]
        index = _index_vec(rng, c["dup"], size, m)
        src_shape = (m,) if len(shape) == 1 else ((m, shape[1]) if dim == 0 else (shape[0], m))
        source = _values(rng, c["values"], src_shape, red)
    else:                                             # scatter_reduce: index has src's shape
        if len(shape) == 1:
            src_shape = (8,)
            index = _index_vec(rng, c["dup"], size, 8)
        elif dim == 0:
            src_shape = (5, shape[1])
            index = [_index_vec(rng, c["dup"], size, 5) for _ in range(shape[1])]
            index = [list(r) for r in zip(*index)]     # (5, C): column c holds the targets of column c
        else:
            src_shape = (shape[0], 5)
            index = [_index_vec(rng, c["dup"], size, 5) for _ in range(shape[0])]
        source = _values(rng, c["values"], src_shape, red)
    return dict(self=self_t, index=index, source=source, dim=dim, upstream_seed=int(rng.integers(0, 2 ** 31)))


def all_conditions():
    return {"ce": ce_conditions(), "flce": flce_conditions(), "pool": pool_conditions(), "index": index_conditions()}


if __name__ == "__main__":
    total = 0
    for fam, conds in all_conditions().items():
        by_op = {}
        for c in conds:
            by_op[c["op"]] = by_op.get(c["op"], 0) + 1
        total += len(conds)
        print(fam, len(conds), by_op)
    print("total conditions", total, "x seeds", len(SEEDS), "=", total * len(SEEDS))
