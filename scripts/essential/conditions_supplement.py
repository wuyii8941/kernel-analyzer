"""Phase-1 supplement S3 (docs/protocol_essential_bugs_phase2_20261007.md section 1): boundary inputs for pooling and
index / scatter that phase 1 did not cover - extreme and mixed magnitudes, non-contiguous strides ("strided",
"transposed"), a non-zero storage offset ("offset"), and larger sizes that are not multiples of block sizes.
Same interfaces as conditions.py (pool_conditions / make_pool_inputs, index_conditions / make_index_inputs).

    python scripts/essential/conditions_supplement.py
"""
from __future__ import annotations

import itertools

import numpy as np

from conditions import SEEDS, _index_vec, _rng  # noqa: F401  (SEEDS re-exported)

VALUES = ("large", "small", "mixed")
LAYOUTS = ("strided", "offset", "transposed")


def _scaled(rng, kind, shape):
    x = rng.normal(0.0, 1.0, shape)
    if kind == "large":
        return x * 1e30
    if kind == "small":
        return x * 1e-30
    return x * 10.0 ** rng.integers(-30, 31, shape)          # mixed magnitudes, 1e-30 .. 1e30


def pool_conditions():
    out = []
    sizes = {1: (1027,), 2: (33, 37), 3: (9, 10, 11)}
    avg_settings = (dict(kernel=3, stride=2, padding=1, ceil_mode=True, count_include_pad=True),
                    dict(kernel=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False))
    max_settings = (dict(kernel=3, stride=2, padding=1, dilation=1, ceil_mode=True),
                    dict(kernel=2, stride=1, padding=1, dilation=2, ceil_mode=False))
    for nd, vals, lay, st in itertools.product((1, 2, 3), VALUES, LAYOUTS, avg_settings):
        out.append(dict(op="avg_pool", nd=nd, size=sizes[nd], kernel=(st["kernel"],) * nd, stride=(st["stride"],) * nd,
                        padding=(st["padding"],) * nd, ceil_mode=st["ceil_mode"], count_include_pad=st["count_include_pad"],
                        divisor_override=None, values=vals, layout=lay))
    for nd, vals, lay, st in itertools.product((1, 2, 3), VALUES, LAYOUTS, max_settings):
        out.append(dict(op="max_pool", nd=nd, size=sizes[nd], kernel=(st["kernel"],) * nd, stride=(st["stride"],) * nd,
                        padding=(st["padding"],) * nd, dilation=(st["dilation"],) * nd, ceil_mode=st["ceil_mode"],
                        values=vals, layout=lay))
    for i, c in enumerate(out):
        c["id"] = f"s3pool{i:03d}"
    return out


def make_pool_inputs(c, seed):
    rng = _rng(c["id"], seed)
    x = _scaled(rng, c["values"], (2,) + tuple(c["size"]))
    return dict(x=x, upstream_seed=int(rng.integers(0, 2 ** 31)))


_SHAPES = {"1d_large": ((1029,), 0), "2d_dim1_large": ((37, 33), 1)}
_OPS = (("index_add", None), ("index_reduce", "amax"), ("index_reduce", "mean"), ("index_reduce", "prod"),
        ("scatter_reduce", "sum"), ("scatter_reduce", "amax"), ("scatter_reduce", "mean"))


def index_conditions():
    out = []
    for (op, red), vals, lay, shape in itertools.product(_OPS, VALUES, LAYOUTS, _SHAPES):
        if red == "prod" and vals != "small":                 # products of 1e30-scale values leave the float range
            continue
        c = dict(op=op, shape=shape, dup="some", values=vals, layout=lay)
        if op == "index_add":
            c.update(m=2051, alpha=2.5)
        elif op == "index_reduce":
            c.update(reduce=red, include_self=True, m=2051)
        else:
            c.update(reduce=red, include_self=False)
        out.append(c)
    for i, c in enumerate(out):
        c["id"] = f"s3idx{i:03d}"
    return out


def make_index_inputs(c, seed):
    rng = _rng(c["id"], seed)
    shape, dim = _SHAPES[c["shape"]]
    self_t = _scaled(rng, c["values"], shape)
    size = shape[dim]
    if c["op"] in ("index_add", "index_reduce"):
        m = c["m"]
        index = _index_vec(rng, c["dup"], size, m)
        src_shape = (m,) if len(shape) == 1 else (shape[0], m)
        source = _scaled(rng, c["values"], src_shape)
    else:
        if len(shape) == 1:
            src_shape = (2053,)
            index = _index_vec(rng, c["dup"], size, 2053)
        else:
            src_shape = (shape[0], 41)
            index = [_index_vec(rng, c["dup"], size, 41) for _ in range(shape[0])]
        source = _scaled(rng, c["values"], src_shape)
    return dict(self=self_t, index=index, source=source, dim=dim, upstream_seed=int(rng.integers(0, 2 ** 31)))


if __name__ == "__main__":
    print("pool", len(pool_conditions()), "index", len(index_conditions()))
