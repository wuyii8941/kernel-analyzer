"""Rule ``reduce.welford`` (tool 3.0): multi-value reduction with the Welford merge.

Containment argument (also in results/general/rule_registry.json): for weights w_i >= 0 with W > 0, every merge tree of
the pairwise Welford merge equals the closed form mean = sum w m / W, M2 = sum s + sum w (m - mean)^2, weight = W; the
evaluator encloses the closed form, so the tree Triton chooses does not matter.  Counterexample: a merge that drops a
factor is not Welford and must be rejected; negative control: a three-value region of independent sums is not
matched; premise check: possibly negative weights -> not established.

DSL v2 (rc3 02 6.2): a combine region without an order-free fast path is no longer rejected.  It is interpreted step by
step along the lowering's combination order read from the TTGIR, and the target is order-specific.  The tests below
that used to expect "unrecognized reduction combiner" now check that order-specific reference against an exact
rational fold written independently here (64 elements, 4 warps: one element per thread, a butterfly over the 32 lanes
of each of 2 warps, then over the 2 warps)."""
from __future__ import annotations

import random
from fractions import Fraction as F

import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from kernel_analyzer import check  # noqa: E402


def merge(a, b):
    (ma, sa, wa), (mb, sb, wb) = a, b
    W = wa + wb
    r = F(0) if W == 0 else wb / W
    d = mb - ma
    return (ma + d * r, sa + sb + d * d * wa * r, W)


def closed_form(items):
    W = sum(w for _, _, w in items)
    mean = sum(w * m for m, _, w in items) / W
    return (mean, sum(s for _, s, _ in items) + sum(w * (m - mean) ** 2 for m, _, w in items), W)


def random_tree(items, rng):
    items = list(items)
    while len(items) > 1:
        i = rng.randrange(len(items) - 1)
        items[i:i + 2] = [merge(items[i], items[i + 1])]
    return items[0]


def test_closed_form_equals_every_merge_tree_in_exact_arithmetic():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randrange(2, 12)
        items = [(F(rng.randrange(-50, 50), rng.randrange(1, 9)), F(rng.randrange(0, 5)), F(rng.choice([0, 0, 1, 2, 3])))
                 for _ in range(n)]
        if sum(w for *_, w in items) == 0:
            continue
        ref = closed_form(items)
        for _ in range(3):
            assert random_tree(items, rng) == ref


pytestmark_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@triton.jit
def _welford_combine(m1, s1, w1, m2, s2, w2):
    delta = m2 - m1
    nw = w1 + w2
    r = tl.where(nw == 0.0, 0.0, w2 / nw)
    return m1 + delta * r, s1 + s2 + delta * delta * w1 * r, nw


@triton.jit
def _broken_combine(m1, s1, w1, m2, s2, w2):
    delta = m2 - m1
    nw = w1 + w2
    r = tl.where(nw == 0.0, 0.0, w2 / nw)
    return m1 + delta * r, s1 + s2 + delta * delta * r, nw          # w1 factor dropped


@triton.jit
def _three_sums(a1, b1, c1, a2, b2, c2):
    return a1 + a2, b1 + b2, c1 + c2


def _make(combine):
    @triton.jit
    def kern(x_ptr, w_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        w = tl.load(w_ptr + offs)
        m, s, ww = tl.reduce((x, tl.zeros_like(x), w), 0, combine)
        tl.store(out_ptr, m)
        tl.store(out_ptr + 1, s)
        tl.store(out_ptr + 2, ww)
    return kern


_KERNELS = {"welford": _make(_welford_combine), "broken": _make(_broken_combine), "three_sums": _make(_three_sums)}


def _run(kind, weights, keep=None):
    kern = _KERNELS[kind]

    class Case(check.Case):
        name = f"welford_{kind}"

        def inputs(self, seed):
            g = torch.Generator().manual_seed(seed)
            return {"x": torch.randn(64, generator=g).cuda(), "w": torch.tensor(weights(seed), dtype=torch.float32).cuda()}

        def launch(self, inp):
            out = torch.empty(3, device="cuda")
            kern[(1,)](inp["x"], inp["w"], out, N=64)
            return {"out": out}

    return check.run(Case(), dev=[0], conf=[1, 2], keep=keep)


def _inputs(seed, weights):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, generator=g).double().numpy()
    return [F(float(v)) for v in x], [F(float(np.float32(w))) for w in weights(seed)]


def _tree_fold(combine, leaves):
    """The documented lowering order for 64 elements with 4 warps (tpw = 32, wpc = 2), in exact arithmetic."""
    lanes = [[leaves[w * 32 + l] for l in range(32)] for w in range(2)]
    for w in range(2):
        stride = 16
        while stride >= 1:
            lanes[w] = [combine(lanes[w][l], lanes[w][l ^ stride]) for l in range(32)]
            stride //= 2
    part = [lanes[0][0], lanes[1][0]]
    return combine(part[0], part[1])


def _exact_merge_broken(a, b):
    (m1, s1, w1), (m2, s2, w2) = a, b
    nw = w1 + w2
    r = F(0) if nw == 0 else w2 / nw
    d = m2 - m1
    return (m1 + d * r, s1 + s2 + d * d * r, nw)


def _encloses(keep_rows, seed_value):
    for r in keep_rows:
        for i, v in enumerate(seed_value(r["seed"])):
            assert F(float(r["r_lo"][i])) <= v <= F(float(r["r_hi"][i])), (r["seed"], i)


@pytestmark_cuda
def test_handwritten_welford_merge_has_a_complete_reference():
    rep = _run("welford", lambda s: [1.0] * 64)
    o = rep["outputs"]["out"]
    assert rep["ttir_coverage_complete"] == [True]
    assert o["reference_classes"]["finite_complete_fraction"] == 1.0


@pytestmark_cuda
def test_merge_without_the_weight_factor_gets_an_order_specific_reference():
    keep = {}
    w = lambda s: [1.0] * 64  # noqa: E731
    rep = _run("broken", w, keep)
    o = rep["outputs"]["out"]
    assert o["reference_classes"]["finite_complete_fraction"] == 1.0
    assert any("tpw=32, wpc=2" in k for k in (o.get("not_established_reasons_seed0") or {}))

    def exact(seed):
        x, ww = _inputs(seed, w)
        return _tree_fold(_exact_merge_broken, [(x[i], F(0), ww[i]) for i in range(64)])
    _encloses(keep["out"], exact)


@pytestmark_cuda
def test_three_independent_sums_get_a_complete_reference():
    keep = {}
    w = lambda s: [1.0] * 64  # noqa: E731
    rep = _run("three_sums", w, keep)
    assert rep["outputs"]["out"]["reference_classes"]["finite_complete_fraction"] == 1.0

    def exact(seed):
        x, ww = _inputs(seed, w)
        return (sum(x), F(0), sum(ww))
    _encloses(keep["out"], exact)


@pytestmark_cuda
def test_possibly_negative_weights_are_not_established():
    rep = _run("welford", lambda s: [1.0] * 63 + [-1.0])
    o = rep["outputs"]["out"]
    assert o["reference_classes"]["finite_complete_fraction"] < 1.0
    assert any("weight may be negative" in k for k in (o.get("not_established_reasons_seed0") or {}))


@pytestmark_cuda
def test_inductor_bfloat16_variance_encloses_the_exact_variance():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "essential"))
    import common
    st = {}

    def setup():
        torch._dynamo.reset()
        st["fn"] = torch.compile(lambda x: x.var(-1), dynamic=False)
        launch(inputs(0))
        torch.cuda.synchronize()

    def inputs(seed):
        g = torch.Generator().manual_seed(seed)
        return {"x": torch.randn(4, 1027, generator=g, dtype=torch.float64).float().to(torch.bfloat16).cuda(), "_seed": seed}

    def launch(t):
        return {"out": st["fn"](t["x"])}

    rep, keep = common.fr_run("welford_var_bf16", setup, inputs, launch, lambda _t: None)
    assert rep["ttir_coverage_complete"] == [True]
    for r in keep["out"]:                              # the exact (rational) variance of the bf16 inputs
        x = inputs(r["seed"])["x"].double().cpu().numpy()
        for i, row in enumerate(x):
            q = [F(float(v)) for v in row]
            mean = sum(q) / len(q)
            var = sum((v - mean) ** 2 for v in q) / (len(q) - 1)
            assert F(float(r["r_lo"][i])) <= var <= F(float(r["r_hi"][i]))


@triton.jit
def _unguarded_combine(m1, s1, w1, m2, s2, w2):
    delta = m2 - m1
    nw = w1 + w2
    r = w2 / nw                                                    # no guard: 0 / 0 when two zero weights meet
    return m1 + delta * r, s1 + s2 + delta * delta * w1 * r, nw


_KERNELS["unguarded"] = _make(_unguarded_combine)


@pytestmark_cuda
def test_unguarded_ratio_with_zero_weights_is_not_established():
    # in this tree the 32 zero weights all sit in warp 0, so the lane butterfly divides 0 by 0: mean and M2 have no
    # real value (the closed form cannot decide it; the actual tree does), the total weight does
    rep = _run("unguarded", lambda s: [0.0] * 32 + [1.0] * 32)
    o = rep["outputs"]["out"]
    assert abs(o["reference_classes"]["finite_complete_fraction"] - 1 / 3) < 1e-9
    assert any("domain of div" in k for k in (o.get("not_established_reasons_seed0") or {}))


@pytestmark_cuda
def test_all_zero_weights_follow_the_actual_merge_tree():
    # the closed form leaves the mean open (it depends on the merge tree); the actual tree keeps the left operand at
    # every merge (r = 0), so the mean is the first element; M2 = sum of s = 0 and W = 0
    keep = {}
    w = lambda s: [0.0] * 64  # noqa: E731
    rep = _run("welford", w, keep)
    o = rep["outputs"]["out"]
    assert o["reference_classes"]["finite_complete_fraction"] == 1.0

    def exact(seed):
        x, ww = _inputs(seed, w)
        return _tree_fold(merge, [(x[i], F(0), ww[i]) for i in range(64)])
    _encloses(keep["out"], exact)
