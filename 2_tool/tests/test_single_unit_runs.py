"""A run with development units only (confirmation empty, path A2 deviation D-A2-9 for vocabulary-sized calls) must
produce a report: the seed summary used to index the first and last confirmation seed and raised IndexError."""
from __future__ import annotations

import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from kernel_analyzer import check  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
N = 64


@triton.jit
def _add(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))


class _Add(check.Case):
    name = "single_unit_add"

    def inputs(self, seed):
        g = torch.Generator().manual_seed(seed)
        return {"x": torch.randn(N, generator=g).cuda(), "y": torch.randn(N, generator=g).cuda()}

    def launch(self, inp):
        out = torch.empty(N, device="cuda")
        _add[(1,)](inp["x"], inp["y"], out, N=N)
        return {"out": out}


def test_program_reference_run_with_no_confirmation_units():
    rep = check.run(_Add(), dev=[0], conf=[])
    assert rep["seeds"]["confirmation"] == []
    assert rep["outputs"]["out"]["reference_classes"]["finite_complete_fraction"] == 1.0


class _Tanh(check.Case):
    name = "single_unit_tanh"

    def inputs(self, seed):
        return {"x": np.random.default_rng(seed).normal(size=N)}

    def launch(self, inp):
        return {"y": torch.from_numpy(np.asarray(np.tanh(inp["x"]), dtype=np.float32))}

    def spec(self, inp):
        return {"y": check.f64_point_spec(np.tanh(inp["x"]))}


def test_black_box_run_with_no_confirmation_units():
    rep = check.run_black_box(_Tanh(), dev=[0], conf=[])
    assert rep["seeds"]["confirmation"] == []
