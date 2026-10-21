"""Independent review tests for path A closure package C (LAYOUT, POOL, INTERP drafts of 2026-10-20).

Reviewer-written. Expected values are computed by the reviewer's own index loops (Fraction / plain Python),
derived from the retained sources: T_PAD, LPI_DERIVATIVES.SLICE + T_SLICE_SCATTER, LPI_T_REPEAT_INTERLEAVE,
T_TRIL, T_AVGPOOL2D (+ the PA.DIVISOR task choice R2), LPI_DECOMP.UPSAMPLE_NEAREST_SCALES. Neither the draft
adapters' helpers nor kernel_spec.aten are used to compute expectations; the drafts are only called through
kernel_spec.registry.load on a fresh stage under tmp_path.
"""
from fractions import Fraction
from itertools import product
from pathlib import Path
import math
import random
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import evaluate  # noqa: E402

DRAFT = ROOT / "contracts" / "draft"


def draft_tag(cid):
    # LAYOUT_DRAFT selects a later LAYOUT draft pair (e.g. path_a_A2_negzero_20261022); default: the A2 package C pair.
    import os
    return os.environ.get("LAYOUT_DRAFT", "path_a_A2_20261020") if cid == "LAYOUT" else "path_a_A2_20261020"
INT64_MAX = 9223372036854775807
IDS = ("LAYOUT", "POOL", "INTERP")


@pytest.fixture(scope="module")
def C(tmp_path_factory):
    root = tmp_path_factory.mktemp("a2_review_C")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    for cid in IDS:
        shutil.copy(DRAFT / f"{cid}_{draft_tag(cid)}.yaml", root / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / f"{cid}_adapter_{draft_tag(cid)}.py", root / "adapters" / f"{cid}.py")
    return {cid: registry.load(cid, contracts=root / "contracts", adapters=root / "adapters") for cid in IDS}


def run(contract, profile, op, *args, **kwargs):
    return evaluate(contract.adapter.reference, profile, {"op": op, "args": list(args), "kwargs": kwargs})


def rec(dtype, nested, shape=None):
    arr = np.array(nested, dtype=object)
    return {"t": dtype, "v": nested if shape is not None else arr.tolist(),
            "shape": list(shape) if shape is not None else list(arr.shape)}


def same(observed, expected):
    """Coordinate-wise exact comparison of nested lists (shape included)."""
    if isinstance(expected, list):
        assert isinstance(observed, list) and len(observed) == len(expected)
        for o, e in zip(observed, expected):
            same(o, e)
        return
    if isinstance(expected, bool):
        assert observed is expected or observed == expected and isinstance(observed, (bool, np.bool_))
        return
    assert Fraction(observed) == Fraction(expected), (observed, expected)


# ------------------------------------------------------------------ reviewer's own n-d helpers (nested lists)
def shape_of(x):
    s = []
    while isinstance(x, list):
        s.append(len(x))
        x = x[0] if x else None
        if x is None:
            break
    return s


def build(shape, f, prefix=()):
    if not shape:
        return f(prefix)
    return [build(shape[1:], f, prefix + (i,)) for i in range(shape[0])]


def get(x, idx):
    for i in idx:
        x = x[i]
    return x


def make(shape, gen):
    return build(list(shape), lambda idx: gen())


def norm_dim(d, rank):
    return d + rank if d < 0 else d


def my_pad(x, shape, pad, fill):
    out_shape = list(shape)
    lo = [0] * len(shape)
    for k in range(len(pad) // 2):
        d = len(shape) - 1 - k
        lo[d] = pad[2 * k]
        out_shape[d] = shape[d] + pad[2 * k] + pad[2 * k + 1]

    def f(idx):
        src = [i - l for i, l in zip(idx, lo)]
        if all(0 <= s < n for s, n in zip(src, shape)):
            return get(x, src)
        return fill
    return build(out_shape, f)


def my_slice_norm(n, start, end):
    start = start + n if start < 0 else start
    end = end + n if end < 0 else end
    start = min(max(start, 0), n)
    end = min(max(end, 0), n)
    return start, max(end, start)


def my_slice_backward(g, sizes, dim, start, end, step, zero):
    dim = norm_dim(dim, len(sizes))
    s, e = my_slice_norm(sizes[dim], start, end)

    def f(idx):
        k = idx[dim] - s
        if idx[dim] >= s and idx[dim] < e and k % step == 0:
            j = list(idx)
            j[dim] = k // step
            return get(g, j)
        return zero
    return build(list(sizes), f)


def my_repeat_interleave(x, shape, r, dim):
    if dim is None:
        flat = []

        def walk(v):
            if isinstance(v, list):
                for w in v:
                    walk(w)
            else:
                flat.append(v)
        walk(x)
        return [flat[j // r] for j in range(len(flat) * r)]
    d = norm_dim(dim, len(shape))
    out_shape = list(shape)
    out_shape[d] *= r

    def f(idx):
        j = list(idx)
        j[d] //= r
        return get(x, j)
    return build(out_shape, f)


def my_tril(x, shape, k, zero):
    return build(list(shape), lambda idx: get(x, idx) if idx[-1] - idx[-2] <= k else zero)


# ======================================================================================== LAYOUT
def test_pad_recorded_int64_labels(C):
    rng = random.Random(1)
    vals = [[rng.choice([-100, 0, 1, 50256, -(2 ** 63), 2 ** 63 - 1, rng.randrange(-2 ** 40, 2 ** 40)])
             for _ in range(128)]]
    r = run(C["LAYOUT"], "real", "aten.pad", rec("i64", vals), [0, 1], "constant", -100.0)
    assert r.outcome == "value"
    exp = my_pad(vals, [1, 128], [0, 1], -100)
    same(r.fields["out"], exp)
    assert shape_of(r.fields["out"]) == [1, 129]


@pytest.mark.parametrize("dtype,shape,pad,value", [
    ("f32", [2, 3], [1, 2, 0, 1], -100.0), ("i32", [2, 3], [-1, 1, 1, -1], 7), ("i16", [3], [2, -3], None),
    ("u8", [1, 2], [0, 0, 2, 0], 255), ("i8", [0, 3], [1, 1], -128), ("f64", [2, 2], [-2, 0], 0.5),
    ("i64", [2, 1, 2], [1, 0, 0, 0, 1, 1], 3.0), ("bf16", [1, 1], [0, 0], None)])
def test_pad_geometry_and_integer_fills(C, dtype, shape, pad, value):
    rng = random.Random(len(shape) * 31 + len(pad))
    if dtype.startswith(("f", "b")):
        x = make(shape, lambda: Fraction(rng.randrange(-16, 16), 2))
    else:
        lo, hi = {"i32": (-2 ** 31, 2 ** 31 - 1), "i16": (-2 ** 15, 2 ** 15 - 1), "u8": (0, 255),
                  "i8": (-128, 127), "i64": (-2 ** 63, 2 ** 63 - 1)}[dtype]
        x = make(shape, lambda: rng.choice([lo, hi, 0, rng.randrange(lo, hi)]))
    xv = build(shape, lambda idx: (str(get(x, idx)) if isinstance(get(x, idx), Fraction) else get(x, idx)))
    args = [rec(dtype, xv, shape), pad] + ([] if value is None else ["constant", value])
    r = run(C["LAYOUT"], "real", "aten.pad", *args)
    assert r.outcome == "value", r
    fill = 0 if value is None else Fraction(value)
    same(r.fields["out"], my_pad(x, shape, pad, fill))


@pytest.mark.parametrize("dtype,value,outcome", [
    ("i64", 0.5, "unsupported"), ("i64", -2.7, "unsupported"), ("i64", 9.223372036854776e18, "unsupported"),
    ("i64", 9007199254740992.0, "value"), ("i64", -9.223372036854776e18, "value"), ("u8", -1, "unsupported"),
    ("u8", 256, "unsupported"), ("i16", 32767.0, "value"), ("i16", 32768, "unsupported"),
    ("i32", float("nan"), "invalid_input")])
def test_integer_fill_boundaries(C, dtype, value, outcome):
    r = run(C["LAYOUT"], "real", "aten.pad", rec(dtype, [1, 2]), [1, 0], "constant", value)
    assert r.outcome == outcome
    if outcome == "value":
        same(r.fields["out"], [int(value), 1, 2])


@pytest.mark.parametrize("profile,dtype,value", [
    ("real", "i32", float("inf")),
    ("real", "i64", float("-inf")),
    ("boolean", "b8", float("inf")),
    ("real", "f32", float("inf"))])  # review issue LAYOUT-1, fixed 2026-10-21
def test_infinite_fill_is_unsupported(C, profile, dtype, value):
    x = rec(dtype, [True] if dtype == "b8" else [1])
    assert run(C["LAYOUT"], profile, "aten.pad", x, [1, 0], "constant", value).outcome == "unsupported"


@pytest.mark.parametrize("value,outcome,fill", [
    (None, "value", False), (0.0, "value", False), (1.0, "value", True), (0, "value", False), (1, "value", True),
    (True, "value", True), (False, "value", False), (0.5, "unsupported", None), (-1.0, "unsupported", None),
    (2, "unsupported", None)])
def test_bool_pad_fills(C, value, outcome, fill):
    args = [rec("b8", [[True, False]]), [1, 1]] + ([] if value is None else ["constant", value])
    r = run(C["LAYOUT"], "boolean", "aten.pad", *args)
    assert r.outcome == outcome
    if outcome == "value":
        same(r.fields["bool_out"], [[fill, True, False, fill]])


@pytest.mark.parametrize("mode,outcome", [("reflect", "unsupported"), ("replicate", "unsupported"),
                                          ("circular", "unsupported"), ("zeros", "invalid_input"),
                                          ("CONSTANT", "invalid_input")])
def test_pad_modes(C, mode, outcome):
    assert run(C["LAYOUT"], "real", "aten.pad", rec("f32", [[1, 2, 3]]), [1, 1], mode).outcome == outcome


def test_pad_negative_result_size_refused(C):
    r = run(C["LAYOUT"], "real", "aten.pad", rec("i64", [1, 2]), [-2, -1], "constant", 0.0)
    assert r.outcome == "invalid_input"


@pytest.mark.parametrize("dtype,gshape,sizes,start,end", [
    ("bf16", [1, 2, 128, 32], [1, 2, 128, 64], 32, INT64_MAX),
    ("bf16", [1, 2, 128, 32], [1, 2, 128, 64], 0, 32),
    ("f32", [1, 14, 128, 32], [1, 14, 128, 64], 32, INT64_MAX),
    ("f32", [1, 14, 128, 32], [1, 14, 128, 64], 0, 32)])
def test_slice_backward_recorded(C, dtype, gshape, sizes, start, end):
    rng = random.Random(start)
    g = make(gshape, lambda: rng.randrange(-256, 256))
    r = run(C["LAYOUT"], "real", "aten.slice_backward", rec(dtype, g, gshape), sizes, 3, start, end, 1)
    assert r.outcome == "value"
    same(r.fields["out"], my_slice_backward(g, sizes, 3, start, end, 1, 0))


@pytest.mark.parametrize("gshape,sizes,dim,start,end,step", [
    ([2, 2], [2, 7], -1, -5, INT64_MAX, 3), ([3, 0], [3, 9], 1, 8, -INT64_MAX, 2),  # end clamps to 0 < start
    ([0, 4], [5, 4], 0, 7, 9, 1), ([2, 4], [5, 4], 0, -100, 2, 1), ([3], [7], 0, 1, 7, 2),
    ([1, 2, 0], [1, 2, 0], 2, 0, INT64_MAX, 1), ([4, 1], [4, 3], 1, -1, INT64_MAX, 5)])
def test_slice_backward_edges(C, gshape, sizes, dim, start, end, step):
    rng = random.Random(7)
    g = make(gshape, lambda: rng.randrange(-9, 9)) if 0 not in gshape else build(gshape, lambda i: 0)
    r = run(C["LAYOUT"], "real", "aten.slice_backward", rec("f64", g, gshape), sizes, dim, start, end, step)
    assert r.outcome == "value", r
    same(r.fields["out"], my_slice_backward(g, sizes, dim, start, end, step, 0))


@pytest.mark.parametrize("profile,dtype,g,zero", [("real", "i64", [5, -(2 ** 63)], 0),
                                                  ("boolean", "b8", [True, True], False)])
def test_slice_backward_exact_data(C, profile, dtype, g, zero):
    r = run(C["LAYOUT"], profile, "aten.slice_backward", rec(dtype, g), [5], 0, 1, 5, 2)
    key = "out" if profile == "real" else "bool_out"
    same(r.fields[key], my_slice_backward(g, [5], 0, 1, 5, 2, zero))


@pytest.mark.parametrize("gshape,sizes,dim,start,end,step", [
    ([2], [4], 0, 0, 4, 1), ([2], [4], 0, 0, 4, 0), ([2], [4], 0, 2, 0, -1), ([2], [4, 1], 0, 0, 2, 1),
    ([2], [-2], 0, 0, 2, 1), ([2], [4], 1, 0, 2, 1)])
def test_slice_backward_invalid(C, gshape, sizes, dim, start, end, step):
    r = run(C["LAYOUT"], "real", "aten.slice_backward", rec("f32", [1] * gshape[0]), sizes, dim, start, end, step)
    assert r.outcome == "invalid_input"


def test_repeat_interleave_recorded(C):
    rng = random.Random(3)
    shape = [1, 2, 128, 64]
    x = make(shape, lambda: Fraction(rng.randrange(-64, 64), 4))
    xv = build(shape, lambda i: str(get(x, i)))
    r = run(C["LAYOUT"], "real", "aten.repeat_interleave", rec("f32", xv, shape), 7, -3)
    assert r.outcome == "value"
    assert shape_of(r.fields["out"]) == [1, 14, 128, 64]
    same(r.fields["out"], my_repeat_interleave(x, shape, 7, -3))


@pytest.mark.parametrize("dtype,shape,r,dim,osize", [
    ("f32", [2, 3], 2, None, None), ("i8", [2, 3], 3, -1, 9), ("u8", [3, 1, 2], 1, 1, 1), ("f16", [2, 2], 0, 0, 0),
    ("i64", [0, 2], 4, 0, 0), ("f32", [2, 2], 2, None, 8), ("i32", [1, 3], 5, 0, None)])
def test_repeat_interleave_edges(C, dtype, shape, r, dim, osize):
    rng = random.Random(r)
    x = make(shape, lambda: rng.randrange(0, 100)) if 0 not in shape else build(shape, lambda i: 0)
    args = [rec(dtype, x, shape), r] + ([] if dim is None else [dim])
    if dim is None and osize is not None:
        args.append(None)
    kw = {} if osize is None else {"output_size": osize}
    res = run(C["LAYOUT"], "real", "aten.repeat_interleave", *args, **kw)
    assert res.outcome == "value", res
    same(res.fields["out"], my_repeat_interleave(x, shape, r, dim))


def test_repeat_interleave_bool(C):
    x = [[True, False]]
    r = run(C["LAYOUT"], "boolean", "aten.repeat_interleave", rec("b8", x), 2, 1)
    same(r.fields["bool_out"], my_repeat_interleave(x, [1, 2], 2, 1))


@pytest.mark.parametrize("args,kw,outcome", [
    ([-1, 0], {}, "invalid_input"), ([2, 0], {"output_size": 5}, "invalid_input"), ([2, 2], {}, "invalid_input"),
    ([{"t": "i64", "v": [1, 2], "shape": [2]}, 0], {}, "unsupported")])
def test_repeat_interleave_refusals(C, args, kw, outcome):
    r = run(C["LAYOUT"], "real", "aten.repeat_interleave", rec("f32", [[1, 2], [3, 4]]), *args, **kw)
    assert r.outcome == outcome


def test_tril_recorded_bool_mask(C):
    rng = random.Random(5)
    for x in (build([128, 128], lambda i: True), make([128, 128], lambda: rng.random() < 0.5)):
        r = run(C["LAYOUT"], "boolean", "aten.tril", rec("b8", x))
        assert r.outcome == "value"
        same(r.fields["bool_out"], my_tril(x, [128, 128], 0, False))


@pytest.mark.parametrize("profile,dtype,shape,k", [
    ("boolean", "b8", [2, 3, 4], -1), ("boolean", "b8", [3, 2], 200), ("boolean", "b8", [3, 3], -200),
    ("real", "i64", [3, 4], 1), ("real", "u8", [2, 2, 2], 0), ("real", "i16", [0, 3], 0)])
def test_tril_edges(C, profile, dtype, shape, k):
    rng = random.Random(k)
    gen = (lambda: rng.random() < 0.7) if dtype == "b8" else (lambda: rng.randrange(0, 200))
    x = make(shape, gen) if 0 not in shape else build(shape, lambda i: 0)
    zero = False if dtype == "b8" else 0
    r = run(C["LAYOUT"], profile, "aten.tril", rec(dtype, x, shape), k)
    assert r.outcome == "value"
    same(r.fields["out" if profile == "real" else "bool_out"], my_tril(x, shape, k, zero))


@pytest.mark.parametrize("profile,op,args,outcome", [
    ("real", "aten.tril", [{"t": "b8", "v": [[True]], "shape": [1, 1]}], "unsupported"),
    ("boolean", "aten.tril", [{"t": "f32", "v": [[1]], "shape": [1, 1]}], "unsupported"),
    ("boolean", "aten.tril", [{"t": "b8", "v": [True], "shape": [1]}], "invalid_input"),
    ("boolean", "aten.tril", [{"t": "b8", "v": [[1]], "shape": [1, 1]}], "invalid_input"),
    ("real", "aten.clone", [{"t": "i64", "v": [True], "shape": [1]}], "invalid_input"),
    ("real", "aten.clone", [{"t": "u8", "v": [256], "shape": [1]}], "invalid_input"),
    ("real", "aten.cat", [[{"t": "i64", "v": [1], "shape": [1]}, {"t": "f32", "v": [1], "shape": [1]}]],
     "unsupported")])
def test_layout_data_refusals(C, profile, op, args, outcome):
    assert run(C["LAYOUT"], profile, op, *args).outcome == outcome


def test_layout_recorded_calls_reach_exactly_one_automatic_profile(C):
    c = C["LAYOUT"]
    auto = [p for p, d in c.data["profiles"].items() if not d.get("task_choice_required")]
    calls = [("aten.pad", [rec("i64", [[1, 2]]), [0, 1], "constant", -100.0]),
             ("aten.slice_backward", [rec("f32", [[1.0]]), [1, 2], 1, 1, INT64_MAX, 1]),
             ("aten.repeat_interleave", [rec("f32", [[[1.0]]]), 7, -3]),
             ("aten.tril", [rec("b8", [[True, True], [True, True]])])]
    for op, args in calls:
        ok = [p for p in auto if run(c, p, op, *args).outcome == "value"]
        assert len(ok) == 1, (op, ok)


# ======================================================================================== POOL (R2 divisor)
def my_out_len(n, k, s, p, ceil):
    num = n + 2 * p - k
    o = (-(-num // s) if ceil else num // s) + 1
    if ceil and (o - 1) * s >= n + p:
        o -= 1
    return o


def my_avg_windows(H, W, k, s, p, ceil, cip, dov):
    """For each output (oh, ow): list of real input coordinates and the R2 divisor (PA.DIVISOR)."""
    OH, OW = my_out_len(H, k[0], s[0], p[0], ceil), my_out_len(W, k[1], s[1], p[1], ceil)
    win = {}
    for oh, ow in product(range(OH), range(OW)):
        h0, w0 = oh * s[0] - p[0], ow * s[1] - p[1]
        h1, w1 = min(h0 + k[0], H + p[0]), min(w0 + k[1], W + p[1])
        region = (h1 - h0) * (w1 - w0)
        real = [(h, w) for h in range(max(h0, 0), min(h1, H)) for w in range(max(w0, 0), min(w1, W))]
        div = dov if dov is not None else (region if cip else len(real))
        win[(oh, ow)] = (real, div)
    return OH, OW, win


def my_avg_fwd(x, k, s, p, ceil, cip, dov):
    lead = shape_of(x)[:-2]
    H, W = shape_of(x)[-2:]
    OH, OW, win = my_avg_windows(H, W, k, s, p, ceil, cip, dov)

    def f(idx):
        plane = get(x, idx[:-2])
        real, div = win[(idx[-2], idx[-1])]
        return sum((Fraction(plane[h][w]) for h, w in real), Fraction(0)) / div
    return build(lead + [OH, OW], f)


def my_avg_bwd(g, xshape, k, s, p, ceil, cip, dov):
    lead, (H, W) = xshape[:-2], xshape[-2:]
    OH, OW, win = my_avg_windows(H, W, k, s, p, ceil, cip, dov)
    grad = build(list(xshape), lambda i: Fraction(0))
    for idx in product(*[range(n) for n in lead]):
        gp, xp = get(g, idx), get(grad, idx)
        for (oh, ow), (real, div) in win.items():
            for h, w in real:
                xp[h][w] += Fraction(gp[oh][ow]) / div
    return grad


def test_avg_pool2d_recorded_D8(C):
    rng = random.Random(8)
    x = make([1, 1, 4, 4], lambda: rng.randrange(-32, 32))
    r = run(C["POOL"], "torch210_aten_avg2d", "aten.avg_pool2d", rec("f32", x), [3, 3], [2, 2], [0, 0], True)
    assert r.outcome == "value"
    same(r.fields["out"], my_avg_fwd(x, (3, 3), (2, 2), (0, 0), True, True, None))
    g = make([1, 1, 2, 2], lambda: rng.randrange(-8, 8))
    r = run(C["POOL"], "torch210_aten_avg2d", "aten.avg_pool2d_backward", rec("f32", g), rec("f32", x),
            [3, 3], [2, 2], [0, 0], True, True, None)
    assert r.outcome == "value"
    same(r.fields["out"], my_avg_bwd(g, [1, 1, 4, 4], (3, 3), (2, 2), (0, 0), True, True, None))


def _pool_cases():
    rng = random.Random(2026)
    cases = []
    for _ in range(160):
        k = (rng.randint(1, 3), rng.randint(1, 3))
        s = (rng.randint(1, 3), rng.randint(1, 3))
        p = (rng.randint(0, k[0] // 2), rng.randint(0, k[1] // 2))
        H, W = rng.randint(max(1, k[0] - 2 * p[0]), 6), rng.randint(max(1, k[1] - 2 * p[1]), 6)
        ceil, cip = rng.random() < 0.6, rng.random() < 0.5
        dov = rng.choice([None, None, None, 2, 5, -3])
        lead = rng.choice([[2], [1, 2], [2, 1]])
        stride_arg = [] if s == k and rng.random() < 0.5 else list(s)
        cases.append((lead + [H, W], k, s, stride_arg, p, ceil, cip, dov, rng.choice(["f32", "f64", "bf16", "f16"])))
    return cases


@pytest.mark.parametrize("shape,k,s,stride_arg,p,ceil,cip,dov,dtype", _pool_cases())
def test_avg_pool2d_grid(C, shape, k, s, stride_arg, p, ceil, cip, dov, dtype):
    rng = random.Random(str(shape) + str(k))
    x = make(shape, lambda: rng.randrange(-20, 20))
    exp = my_avg_fwd(x, k, s, p, ceil, cip, dov)
    r = run(C["POOL"], "torch210_aten_avg2d", "aten.avg_pool2d", rec(dtype, x), list(k), stride_arg, list(p),
            ceil, cip, dov)
    assert r.outcome == "value", r
    same(r.fields["out"], exp)
    g = make(shape_of(exp), lambda: rng.randrange(-5, 5))
    r = run(C["POOL"], "torch210_aten_avg2d", "aten.avg_pool2d_backward", rec(dtype, g), rec(dtype, x),
            list(k), stride_arg, list(p), ceil, cip, dov)
    assert r.outcome == "value", r
    same(r.fields["out"], my_avg_bwd(g, shape, k, s, p, ceil, cip, dov))


X44 = [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]


@pytest.mark.parametrize("op,args,outcome", [
    ("aten.avg_pool2d", [rec("f32", X44), [2, 2], [], [0, 0], False, True, 0], "invalid_input"),
    ("aten.avg_pool2d", [rec("f32", X44), [2, 2], [1, 1], [2, 2]], "invalid_input"),
    ("aten.avg_pool2d", [rec("f32", X44), [0, 2]], "invalid_input"),
    ("aten.avg_pool2d", [rec("f32", X44), [5, 5]], "invalid_input"),
    ("aten.avg_pool2d", [rec("i64", X44), [2, 2]], "unsupported"),
    ("aten.avg_pool2d", [rec("f32", [[1, 2], [3, 4]]), [2, 2]], "unsupported"),
    ("aten.avg_pool2d_backward", [rec("f32", [[[[1, 2]]]]), rec("f32", X44), [2, 2], [2, 2], [0, 0], False, True,
                                  None], "invalid_input"),
    ("aten.avg_pool2d_backward", [rec("f64", [[[[1, 2], [3, 4]]]]), rec("f32", X44), [2, 2], [2, 2], [0, 0], False,
                                  True, None], "invalid_input")])
def test_avg_pool2d_refusals(C, op, args, outcome):
    assert run(C["POOL"], "torch210_aten_avg2d", op, *args).outcome == outcome


def test_avg_calls_reach_only_task_choice_profiles(C):
    c = C["POOL"]
    assert c.data["profiles"]["torch210_aten_avg2d"]["task_choice_required"] is True
    for p, d in c.data["profiles"].items():
        if d.get("task_choice_required"):
            continue
        r = run(c, p, "aten.avg_pool2d", rec("f32", X44), [3, 3], [2, 2], [0, 0], True)
        assert r.outcome == "unsupported", (p, r.outcome)


# ======================================================================================== INTERP
def my_nearest_idx(isize, osize, s):
    if s is None:
        return [(o * isize) // osize for o in range(osize)]
    q = 1 / Fraction(s)
    return [math.floor(o * q) for o in range(osize)]


def my_upsample(x, oh, ow, sh=None, sw=None):
    N, Cc, H, W = shape_of(x)
    rows, cols = my_nearest_idx(H, oh, sh), my_nearest_idx(W, ow, sw)
    return build([N, Cc, oh, ow], lambda i: x[i[0]][i[1]][rows[i[2]]][cols[i[3]]])


def my_upsample_bwd(g, insize, oh, ow, sh=None, sw=None):
    rows, cols = my_nearest_idx(insize[2], oh, sh), my_nearest_idx(insize[3], ow, sw)
    out = build(list(insize), lambda i: Fraction(0))
    for n, c, a, b in product(range(insize[0]), range(insize[1]), range(oh), range(ow)):
        out[n][c][rows[a]][cols[b]] += Fraction(g[n][c][a][b])
    return out


def test_upsample_recorded_D18(C):
    rng = random.Random(18)
    x = make([2, 3, 10, 8], lambda: rng.randrange(-1000, 1000))
    r = run(C["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d", rec("f32", x), [20, 16],
            2.0, 2.0)
    assert r.outcome == "value"
    exp = my_upsample(x, 20, 16, 2.0, 2.0)
    same(r.fields["out"], exp)
    same(r.fields["out"], build([2, 3, 20, 16], lambda i: x[i[0]][i[1]][i[2] // 2][i[3] // 2]))


def f32_exact_no_scale(isize, osize):
    """After review issue INTERP-1: a value only where fl32(isize/osize) and every o*q are exact binary32 values."""
    q = Fraction(isize, osize)
    exact = lambda v: Fraction(float(np.float32(float(v)))) == v
    return exact(q) and all(exact(o * q) for o in range(osize))


@pytest.mark.parametrize("H,W,oh,ow", [(h, w, a, b) for h, w, a, b in product([1, 3, 5], [2, 4], [1, 4, 7], [3, 8])])
def test_upsample_without_scales_small_grid(C, H, W, oh, ow):
    x = build([1, 2, H, W], lambda i: i[1] * 100 + i[2] * 10 + i[3])
    r = run(C["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d", rec("f64", x), [oh, ow])
    if not (f32_exact_no_scale(H, oh) and f32_exact_no_scale(W, ow)):
        assert r.outcome == "unsupported", r
        return
    assert r.outcome == "value"
    same(r.fields["out"], my_upsample(x, oh, ow))
    assert my_nearest_idx(H, oh, None) == source_index_no_scale(H, oh)
    assert my_nearest_idx(W, ow, None) == source_index_no_scale(W, ow)


@pytest.mark.parametrize("H,W,oh,ow,sh,sw", [
    (1, 11, 1, 2, 1.0, 0.25), (8, 8, 2, 2, 0.5, 0.25), (3, 3, 12, 12, 4.0, 4.0), (4, 4, 3, 3, 1.0, 1.0),
    (4, 6, 8, 3, 2.0, None), (5, 4, 2, 8, None, 2.0), (3, 3, 5, 5, 2.0, 2.0)])
def test_upsample_exact_scales(C, H, W, oh, ow, sh, sw):
    x = build([1, 1, H, W], lambda i: i[2] * 10 + i[3])
    r = run(C["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d", rec("f32", x), [oh, ow],
            sh, sw)
    assert r.outcome == "value", r
    same(r.fields["out"], my_upsample(x, oh, ow, sh, sw))


@pytest.mark.parametrize("H,W,oh,ow,sh,sw", [
    (1, 3, 1, 6, 1.0, 1.0),          # index reaches isize
    (1, 4, 1, 2, 1.0, 0.3),          # 1/0.3 not exact in binary32
    (1, 9, 1, 3, 1.0, 1 / 3),        # 1/(binary64 1/3) not exact in binary32
    (3, 3, 9, 9, 3.0, 3.0),          # 1/3 not exact in binary32 (conservative refusal)
    (2, 2, 3, 3, 1.5, 1.5),          # 2/3 not exact in binary32
    (2, 2, 4, 4, float("inf"), 2.0), (2, 2, 4, 4, 2.0, 0.0), (2, 2, 4, 4, -2.0, 2.0)])
def test_upsample_scale_refusals(C, H, W, oh, ow, sh, sw):
    x = build([1, 1, H, W], lambda i: i[3])
    r = run(C["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d", rec("f32", x), [oh, ow],
            sh, sw)
    assert r.outcome == "unsupported"


@pytest.mark.parametrize("insize,oh,ow,sh,sw", [([1, 2, 3, 2], 5, 4, None, None), ([2, 1, 10, 8], 20, 16, 2.0, 2.0),
                                                ([1, 1, 1, 11], 1, 2, 1.0, 0.25), ([1, 1, 4, 4], 2, 3, None, None),
                                                ([1, 2, 3, 2], 6, 8, None, None), ([2, 1, 12, 8], 3, 2, None, None)])
def test_upsample_backward(C, insize, oh, ow, sh, sw):
    rng = random.Random(oh * ow)
    g = make([insize[0], insize[1], oh, ow], lambda: Fraction(rng.randrange(-40, 40), 8))
    gv = build([insize[0], insize[1], oh, ow], lambda i: str(get(g, i)))
    args = [rec("f64", gv, [insize[0], insize[1], oh, ow]), [oh, ow], insize] + ([sh, sw] if sh or sw else [])
    r = run(C["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d_backward", *args)
    if sh is None and sw is None and not (f32_exact_no_scale(insize[2], oh) and f32_exact_no_scale(insize[3], ow)):
        assert r.outcome == "unsupported", r   # review issue INTERP-1 fix
        return
    assert r.outcome == "value", r
    same(r.fields["out"], my_upsample_bwd(g, insize, oh, ow, sh, sw))


@pytest.mark.parametrize("op,args,outcome", [
    ("aten.upsample_nearest2d", [rec("f32", [[[[1, 2]]]]), [0, 2]], "invalid_input"),
    ("aten.upsample_nearest2d", [rec("f32", [[[[1, 2]]]]), None, [2.0, 2.0]], "unsupported"),
    ("aten.upsample_nearest2d", [rec("i64", [[[[1, 2]]]]), [2, 2]], "unsupported"),
    ("aten.upsample_nearest2d", [rec("f32", [[1, 2]]), [2, 2]], "unsupported"),
    ("aten.upsample_nearest2d_backward", [rec("f32", [[[[1, 2, 3]]]]), [1, 4], [1, 1, 1, 2]], "invalid_input"),
    ("aten.upsample_nearest2d_backward", [rec("f32", [[[[1, 2, 3, 4]]]]), [1, 4], [1, 2, 1, 2]], "invalid_input"),
    ("aten.upsample_nearest2d_backward", [rec("f32", [[[[1, 2, 3, 4]]]]), [1, 4], [1, 1, 2]], "invalid_input")])
def test_upsample_refusals(C, op, args, outcome):
    r = run(C["INTERP"], "torch210_aten_upsample_nearest2d", op, *args)
    assert r.outcome == outcome, r.outcome


def test_interp_only_aten_profile_is_automatic(C):
    c = C["INTERP"]
    auto = [p for p, d in c.data["profiles"].items() if not d.get("task_choice_required")]
    assert auto == ["torch210_aten_upsample_nearest2d"]
    assert run(c, "torch210_nearest2d_size_autograd", "aten.upsample_nearest2d", rec("f32", [[[[1]]]]),
               [2, 2]).outcome == "unsupported"


# ---------------------------------------------------------------- review issue INTERP-1 (no-scale arithmetic)
def source_index_no_scale(isize, osize):
    """LPI_DECOMP.UPSAMPLE_NEAREST_SCALES without a scale: trunc(float32(o) * float32(isize/osize))."""
    o = np.arange(osize, dtype=np.float32)
    return np.trunc(o * np.float32(isize / osize)).astype(np.int64).tolist()


def test_no_scale_source_arithmetic_differs_from_exact_rule():
    # Reviewer's evidence: the retained decomposition's floating index is not floor(o*isize/osize) everywhere.
    assert source_index_no_scale(2, 82)[41] == 0
    assert my_nearest_idx(2, 82, None)[41] == 1


# review issue INTERP-1, fixed 2026-10-21
def test_no_scale_branch_refuses_or_agrees_with_source(C):
    x = build([1, 1, 1, 2], lambda i: i[3])
    r = run(C["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d", rec("f32", x), [1, 82])
    if r.outcome == "value":
        assert [int(v) for v in r.fields["out"][0][0][0]] == source_index_no_scale(2, 82)
    else:
        assert r.outcome == "unsupported"


def test_exact_predicate_implies_source_agreement():
    # Reviewer re-check of the INTERP-1 fix: wherever the admission condition holds, the exact rule equals the
    # retained source's float32 index (so admitted values never contradict the source).
    admitted = 0
    for isize in range(1, 65):
        for osize in range(1, 129):
            if f32_exact_no_scale(isize, osize):
                admitted += 1
                assert my_nearest_idx(isize, osize, None) == source_index_no_scale(isize, osize), (isize, osize)
    assert admitted > 100


# ---------------------------------------------------------------- re-check of the tensor-parameter decoder (2026-10-21)
@pytest.mark.parametrize("profile,dtype,v1,v2", [("real", "i64", [1, 2], [3]), ("real", "f32", [1, 2], [3]),
                                                 ("boolean", "b8", [True, False], [True])])
def test_tensor_parameters_still_accepted(C, profile, dtype, v1, v2):
    key = "out" if profile == "real" else "bool_out"
    c = C["LAYOUT"]
    r = run(c, profile, "aten.cat", [rec(dtype, v1), rec(dtype, v2)], 0)
    same(r.fields[key], v1 + v2)
    r = run(c, profile, "aten.stack", [rec(dtype, v1), rec(dtype, v1)], 0)
    same(r.fields[key], [v1, v1])
    base = [v1, v1]
    r = run(c, profile, "aten.select_scatter", rec(dtype, base), rec(dtype, [v2[0], v2[0]]), 0, 1)
    same(r.fields[key], [v1, [v2[0], v2[0]]])
    r = run(c, profile, "aten.slice_scatter", rec(dtype, v1), rec(dtype, v2), 0, 1, INT64_MAX, 1)
    same(r.fields[key], [v1[0], v2[0]])
    zero = False if dtype == "b8" else 0
    r = run(c, profile, "aten.slice_backward", rec(dtype, v2), [2], 0, 1, INT64_MAX, 1)
    same(r.fields[key], [zero, v2[0]])
    r = run(c, profile, "aten.constant_pad_nd", rec(dtype, v1), [1, 0])
    same(r.fields[key], [zero] + v1)


T0 = {"t": "i64", "v": 0, "shape": []}


@pytest.mark.parametrize("op,args", [
    ("aten.narrow", [rec("f32", [1, 2, 3]), 0, T0, 1]),
    ("aten.select", [rec("f32", [1, 2, 3]), 0, T0]),
    ("aten.slice", [rec("f32", [1, 2, 3]), T0, 0, 2, 1]),
    ("aten.view", [rec("f32", [1, 2]), [T0, 2]]),
    ("aten.constant_pad_nd", [rec("f32", [1, 2]), [T0, 1]]),
    ("aten.repeat_interleave", [rec("f32", [1, 2]), rec("i64", [1, 2]), 0]),
    ("aten.tril", [rec("f32", [[1, 2], [3, 4]]), T0])])
def test_tensor_for_non_tensor_parameter_is_unsupported(C, op, args):
    assert run(C["LAYOUT"], "real", op, *args).outcome == "unsupported"
