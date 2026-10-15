"""Textbook premises (reference_eval.premises): each premise replaced gives the textbook computation, nothing replaced
gives the device's bits, and the device differs from the textbook computation where the premise matters (needs CUDA)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("gmpy2")
import triton.language as tl  # noqa: E402

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.premises import PREMISES, textbook_emulate  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capture(fn):
    with TritonLaunchRecorder() as R:
        fn()
        torch.cuda.synchronize()
    return R.launches[0]


def _values(result, name):
    b = next(b for b in result.buffers.values() if b.name == name)
    w = b.written
    assert np.all(b.st[w] == 0)
    return b.lo[w], b.actual_after[w]


@triton.jit
def _div_k(x_ptr, d_ptr, y_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    tl.store(y_ptr + r, tl.load(x_ptr + r) / tl.load(d_ptr + r))


@cuda
def test_functions_premise_gives_ieee_division():
    g = torch.Generator(device="cuda").manual_seed(30)
    x, d = torch.randn(4096, device="cuda", generator=g), torch.rand(4096, device="cuda", generator=g) + 0.5
    y = torch.empty_like(x)
    L = _capture(lambda: _div_k[(1,)](x, d, y, B=4096))
    ieee = (x.cpu() / d.cpu()).double().numpy()
    text, device = _values(textbook_emulate(L, premises=["functions"])[0], "y_ptr")
    assert np.array_equal(text, ieee)
    assert np.count_nonzero(device != ieee) > 0           # div.full is not the IEEE division
    same, device2 = _values(textbook_emulate(L, premises=())[0], "y_ptr")
    assert np.array_equal(same, device2)                   # no premise replaced: the device's bits


@triton.jit
def _rowsum_k(x_ptr, y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    tl.store(y_ptr + row, tl.sum(tl.load(x_ptr + row * N + tl.arange(0, N)), axis=0))


@cuda
def test_order_premise_gives_the_sequential_sum():
    x = torch.randn(64, 4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(31))
    y = torch.empty(64, device="cuda")
    L = _capture(lambda: _rowsum_k[(64,)](x, y, N=4096))
    seq = []
    for row in x.cpu().numpy():
        acc = np.float32(0)
        for v in row:
            acc = np.float32(acc + v)
        seq.append(float(acc))
    text, device = _values(textbook_emulate(L, premises=["order"])[0], "y_ptr")
    assert np.array_equal(text, np.array(seq))
    assert np.count_nonzero(device != np.array(seq)) > 0   # the device adds in its layout's tree order


@triton.jit
def _muladd_k(a_ptr, b_ptr, c_ptr, y_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    tl.store(y_ptr + r, tl.load(a_ptr + r) * tl.load(b_ptr + r) + tl.load(c_ptr + r))


@cuda
def test_fusion_premise_rounds_the_product_separately():
    g = torch.Generator(device="cuda").manual_seed(32)
    a, b, c = (torch.randn(4096, device="cuda", generator=g) for _ in range(3))
    y = torch.empty_like(a)
    L = _capture(lambda: _muladd_k[(1,)](a, b, c, y, B=4096))
    an, bn, cn = (t.cpu().numpy() for t in (a, b, c))
    two = (np.float32(an * bn) + cn).astype(np.float32).astype(np.float64)
    fused = (an.astype(np.float64) * bn + cn).astype(np.float32).astype(np.float64)   # exact product: one rounding
    text, device = _values(textbook_emulate(L, premises=["fusion"])[0], "y_ptr")
    assert np.array_equal(text, two)
    assert np.array_equal(device, fused) and np.count_nonzero(fused != two) > 0


@triton.jit
def _dot_k(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))


@cuda
def test_tensor_core_premise_removes_the_tf32_shrink():
    """tf32 operands truncated toward zero shrink the product (about -3.5e-4 per operand); rounded to nearest they
    do not.  The textbook result equals the tf32-rounded operands' product summed left to right in fp32."""
    g = torch.Generator(device="cuda").manual_seed(33)
    a = 0.1 * torch.randn(64, 128, device="cuda", generator=g)
    b = 0.1 * torch.randn(128, 64, device="cuda", generator=g)
    c = torch.empty(64, 64, device="cuda")
    L = _capture(lambda: _dot_k[(1,)](a, b, c, M=64, N=64, K=128))
    exact = (a.double() @ b.double()).cpu().numpy().reshape(-1)
    text, device = _values(textbook_emulate(L, premises=["tensor_core"])[0], "c_ptr")

    def align(v):
        return float(((v - exact) * exact).sum() / (exact * exact).sum())
    assert align(device) < -5e-4 and abs(align(text)) < 5e-5
    assert set(PREMISES) == {"functions", "fusion", "order", "tensor_core", "subnormals"}
