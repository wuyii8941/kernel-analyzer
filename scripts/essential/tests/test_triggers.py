"""Unit tests of the D trigger checks: each check passes on an independent correct reference implementation and flags the
corresponding known defect (B020 / B021 on torch 2.10; B022 is a nightly regression, so on 2.10 the D3 check passes).

    PYTHONPATH=src /data1/tzh/envs/ka_main/bin/python -m pytest -q scripts/essential/tests/test_triggers.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triggers as T  # noqa: E402


class _RefExtremum(torch.autograd.Function):
    """independent reference: extremum over the participants only (include_self=False), gradient to the unique
    extremum (ties split evenly among participants)."""

    @staticmethod
    def forward(ctx, s, idx, src, reduce):
        out = s.clone()
        win = {}
        for i, j in enumerate(idx.tolist()):
            v = float(src[i])
            if j not in win or (v > win[j][0] if reduce == "amax" else v < win[j][0]):
                win[j] = (v, [i])
            elif v == win[j][0]:
                win[j][1].append(i)
        for j, (v, _) in win.items():
            out[j] = v
        ctx.win, ctx.n = win, src.numel()
        return out

    @staticmethod
    def backward(ctx, g):
        gs = g.clone()
        gsrc = torch.zeros(ctx.n, dtype=g.dtype)
        for j, (_, members) in ctx.win.items():
            gs[j] = 0.0
            for i in members:
                gsrc[i] = g[j] / len(members)
        return gs, None, gsrc, None


def _ref(s, idx, src, reduce, include_self):
    assert include_self is False
    return _RefExtremum.apply(s, idx, src, reduce)


@pytest.mark.parametrize("reduce", ["amax", "amin"])
def test_d1_passes_on_reference_and_flags_b020(reduce):
    r = T.d1_extremum_excluded(_ref, reduce)
    assert r["passed"], r
    assert not T.d1_extremum_excluded(T.torch_scatter, reduce)["passed"]           # B020
    assert not T.d1_extremum_excluded(T.torch_index_reduce, reduce)["passed"]      # B020


def test_d2_flags_b021_on_cpu():
    for nd in (1, 2, 3):
        assert not T.d2_no_contributor_window(nd, "cpu")["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_d2_cuda_1d_2d_pass_and_3d_flags_b021():
    assert T.d2_no_contributor_window(1, "cuda")["details"]["grad"] == [0.0] * 4
    assert not T.d2_no_contributor_window(3, "cuda")["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_d3_passes_on_2_10_inductor():
    assert T.d3_sentinel_backward(1, "cuda")["passed"]


def test_d4_forces_equalities():
    x = T.tie_inputs(np.random.default_rng(0), (8,), force_equal=[(0, 5), (2, 3)])
    assert x[0] == x[5] and x[2] == x[3] and np.all(x == np.round(x))
