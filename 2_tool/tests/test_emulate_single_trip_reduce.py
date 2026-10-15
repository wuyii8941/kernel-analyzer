"""The emulator on Inductor's sum-of-squares reduction (RMSNorm / LayerNorm variance) when the reduction loop runs once:
the accumulator starts at +0, LLVM folds ``0 + x*x`` to ``x*x`` (a square is never -0) and contracts the unrounded
squares into the in-thread part of the sum (fma chain).  Found on Qwen3 RMSNorm kernels (three_way protocol, D1)."""

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("gmpy2")
import triton.language as tl  # noqa: E402

from kernel_analyzer.reference_eval import emulate as E  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@triton.jit
def _sumsq_k(x_ptr, y_ptr, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    base = tl.arange(0, RB)[None, :]
    acc = tl.full([1, RB], 0, tl.float32)
    for off in range(0, N, RB):
        idx = off + base
        m = idx < N
        x = tl.load(x_ptr + row * N + idx, m, other=0.0)
        acc = tl.where(m, acc + x * x, acc)
    tl.store(y_ptr + row + tl.arange(0, 1)[:, None], tl.sum(acc, 1)[:, None])


def _capture(rows, N, RB):
    x = torch.randn(rows, N, device="cuda", generator=torch.Generator(device="cuda").manual_seed(40))
    y = torch.empty(rows, device="cuda")
    with TritonLaunchRecorder() as R:
        _sumsq_k[(rows,)](x, y, N=N, RB=RB)
        torch.cuda.synchronize()
    return R.launches[0]


@cuda
def test_single_trip_sum_of_squares_is_bit_identical():
    L = _capture(128, 2048, 2048)
    v = E.verify(L, ungrouped=[], swapped=[], programs=[(r, 0, 0) for r in range(128)])
    assert v["status"] == "bit_identical", v["buffers"]


@cuda
def test_multi_trip_loop_is_unchanged():
    L = _capture(32, 2048, 512)            # four trips: the accumulator is a real running sum
    v = E.verify(L, ungrouped=[], swapped=[], programs=[(r, 0, 0) for r in range(32)])
    assert v["status"] == "bit_identical", v["buffers"]


@triton.jit
def _masked_sumsq_k(x_ptr, y_ptr, R: tl.constexpr, N: tl.constexpr, XB: tl.constexpr):
    rows = tl.program_id(0) * XB + tl.arange(0, XB)[:, None]
    rmask = rows < R
    cols = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + rows * N + cols, rmask, other=0.0)
    s = tl.sum(tl.where(rmask, x * x, 0), 1)[:, None]
    tl.store(y_ptr + rows, s, rmask)


@cuda
def test_row_masked_sum_of_squares_is_bit_identical():
    """Inductor's persistent reductions: sum(where(xmask, x*x, 0)).  The mask is the same along the reduced axis, so
    LLVM moves the select after the in-thread partial sum and the squares are still contracted into it."""
    x = torch.randn(1000, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(41))
    y = torch.empty(1000, device="cuda")
    with TritonLaunchRecorder() as R:
        _masked_sumsq_k[(125,)](x, y, R=1000, N=128, XB=8)
        torch.cuda.synchronize()
    L = R.launches[0]
    v = E.verify(L, ungrouped=[], swapped=[], programs=[(p, 0, 0) for p in range(125)])
    assert v["status"] == "bit_identical", v["buffers"]
