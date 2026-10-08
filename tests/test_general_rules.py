"""Rule-registry tests added in the general-capability round (results/general/rule_registry.json): nested loops and
branches with reference conditions, aliasing (one storage through two pointers), bit-level round trips.  Each has a
positive case checked against exact rational values and a negative control."""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from kernel_analyzer import check  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
N = 64


@triton.jit
def _nested(x_ptr, out_ptr, thr, N: tl.constexpr, ITERS: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    acc = tl.zeros_like(x)
    for i in range(ITERS):
        for j in range(2):
            if i % 2 == 0:
                acc += tl.where(x > thr, x * 0.5, -x)
            else:
                acc -= x * 0.25
    tl.store(out_ptr + offs, acc)


def _exact_nested(x, thr, iters):
    out = []
    for v in x:
        v = F(float(v))
        acc = F(0)
        for i in range(iters):
            for _ in range(2):
                acc += (v * F(1, 2) if v > thr else -v) if i % 2 == 0 else -v * F(1, 4)
        out.append(acc)
    return out


def _run(launch_fn, inputs_fn):
    class Case(check.Case):
        name = "general_rules"

        def inputs(self, seed):
            return inputs_fn(seed)

        def launch(self, inp):
            return launch_fn(inp)

    keep = {}
    rep = check.run(Case(), dev=[0], conf=[1, 2], keep=keep)
    return rep, keep


def test_nested_loops_and_data_branch_are_exact_where_decided():
    thr = 0.1

    def inputs(seed):
        g = torch.Generator().manual_seed(seed)
        return {"x": torch.randn(N, generator=g).cuda()}

    def launch(inp):
        out = torch.empty(N, device="cuda")
        _nested[(1,)](inp["x"], out, thr, N=N, ITERS=3)
        return {"out": out}

    rep, keep = _run(launch, inputs)
    assert rep["outputs"]["out"]["reference_classes"]["finite_complete_fraction"] == 1.0
    for r in keep["out"]:
        ex = _exact_nested(inputs(r["seed"])["x"].cpu().numpy(), F(float(np.float32(thr))), 3)
        for e, lo, hi in zip(ex, r["r_lo"], r["r_hi"]):
            assert F(float(lo)) <= e <= F(float(hi))


@triton.jit
def _select_at_threshold(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = tl.exp(tl.log(x))                                # equals x in the reals; its enclosure contains x
    tl.store(out_ptr + offs, tl.where(y > x, y, -y))


@triton.jit
def _if_at_threshold(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = tl.exp(tl.log(x))
    if tl.sum(y) > tl.sum(x):                            # control flow on an undecided predicate
        tl.store(out_ptr + offs, y)
    else:
        tl.store(out_ptr + offs, -y)


def _threshold_inputs(seed):
    g = torch.Generator().manual_seed(seed)
    return {"x": (torch.rand(N, generator=g) + 1.0).cuda()}


def test_undecided_select_takes_the_union_which_still_encloses_the_true_branch():
    """negative control (select): y > x is false in the reals but undecided on the enclosure: the reference is the
    union of both branches -- an enclosure of the true value -x that is about 2|x| wide (class: enclosure too wide),
    never the wrong branch."""

    def launch(inp):
        out = torch.empty(N, device="cuda")
        _select_at_threshold[(1,)](inp["x"], out, N=N)
        return {"out": out}

    rep, keep = _run(launch, _threshold_inputs)
    for r in keep["out"]:
        x = _threshold_inputs(r["seed"])["x"].cpu().numpy().astype(np.float64)
        assert np.all((r["r_lo"] <= -x) & (-x <= r["r_hi"]))
        assert np.all(r["r_hi"] - r["r_lo"] >= 1.9 * x)


def test_undecided_control_flow_takes_the_union_of_both_branches():
    """negative control (scf.if): an undecided predicate does not pick a branch; the stores of both branches are
    joined (an enclosure of the true -x, about 2|x| wide -- class: enclosure too wide).  kappa = conditional does not
    arise here: in check.run it comes only from loads pinned to captured values (the pin_loads option)."""

    def launch(inp):
        out = torch.empty(N, device="cuda")
        _if_at_threshold[(1,)](inp["x"], out, N=N)
        return {"out": out}

    rep, keep = _run(launch, _threshold_inputs)
    for r in keep["out"]:
        x = _threshold_inputs(r["seed"])["x"].cpu().numpy().astype(np.float64)
        assert np.all((r["r_lo"] <= -x) & (-x <= r["r_hi"]))
        assert np.all(r["r_hi"] - r["r_lo"] >= 1.9 * x)


@triton.jit
def _alias(a_ptr, b_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(a_ptr + offs)
    tl.store(a_ptr + offs, x * 3.0)
    tl.debug_barrier()
    y = tl.load(b_ptr + offs)                                # same storage through a second pointer
    tl.store(a_ptr + N + offs, y + 1.0)


def test_store_then_load_through_an_alias_carries_the_reference_value():
    def inputs(seed):
        g = torch.Generator().manual_seed(seed)
        buf = torch.zeros(2 * N)
        buf[:N] = torch.randn(N, generator=g)
        return {"buf": buf.cuda()}

    def launch(inp):
        buf = inp["buf"].clone()
        _alias[(1,)](buf, buf, N=N)
        return {"buf": buf}

    rep, keep = _run(launch, inputs)
    o = rep["outputs"]["buf"]
    assert o["reference_classes"]["finite_complete_fraction"] == 1.0
    for r in keep["buf"]:
        x0 = inputs(r["seed"])["buf"][:N].cpu().numpy()
        ex = [F(float(v)) * 3 + 1 for v in x0]
        for e, lo, hi in zip(ex, r["r_lo"][N:], r["r_hi"][N:]):
            assert F(float(lo)) <= e <= F(float(hi))


@triton.jit
def _bits(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    i = x.to(tl.int32, bitcast=True)
    i = i ^ 0                                                 # bit-level identity
    tl.store(out_ptr + offs, i.to(tl.float32, bitcast=True) * 2.0)


def test_bitcast_round_trip_of_point_values_is_exact():
    def inputs(seed):
        g = torch.Generator().manual_seed(seed)
        return {"x": torch.randn(N, generator=g).cuda()}

    def launch(inp):
        out = torch.empty(N, device="cuda")
        _bits[(1,)](inp["x"], out, N=N)
        return {"out": out}

    rep, keep = _run(launch, inputs)
    assert rep["outputs"]["out"]["reference_classes"]["finite_complete_fraction"] == 1.0
    for r in keep["out"]:
        x = inputs(r["seed"])["x"].cpu().numpy()
        assert np.array_equal(r["r_lo"], x.astype(np.float64) * 2) and np.array_equal(r["r_hi"], r["r_lo"])
