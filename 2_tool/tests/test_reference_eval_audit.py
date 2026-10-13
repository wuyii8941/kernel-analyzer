"""The audit entry (reference_eval.audit): a significant bias is found and attributed to its node with a fix; a kernel
without one is reported as such (needs CUDA)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("gmpy2")
import triton.language as tl  # noqa: E402

from kernel_analyzer.reference_eval.audit import audit  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@triton.jit
def _dot_k(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, IEEE: tl.constexpr):
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if IEEE:
        c = tl.dot(a, b, input_precision="ieee")
    else:
        c = tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@triton.jit
def _scale_k(x_ptr, y_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    tl.store(y_ptr + r, tl.abs(tl.load(x_ptr + r)) * 2.0)


@cuda
def test_audit_finds_and_attributes_a_tf32_shrink():
    g = torch.Generator(device="cuda").manual_seed(14)
    a, b = torch.randn(64, 128, device="cuda", generator=g), torch.randn(128, 64, device="cuda", generator=g)
    c = torch.empty(64, 64, device="cuda")
    rep = audit(lambda: _dot_k[(1,)](a, b, c, M=64, N=64, K=128, IEEE=False),
                spec={"c_ptr": lambda: a.double() @ b.double()})
    m = rep["measured"]["c_ptr"]
    assert m["significant"] and m["alignment"] < -5e-4 and m.get("prediction_agrees")
    top = rep["attribution"]["c_ptr"]["nodes"][0]
    assert top["node"].startswith("tt.dot") and "tf32" in top["mechanism"]
    assert any("ieee" in f["fix"] for f in rep["findings"])
    assert abs(rep["spec"]["c_ptr"]["semantic_alignment"]) < 1e-9          # the program is the specification


@cuda
def test_audit_reports_no_finding_for_an_exact_kernel():
    x = torch.randn(4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(15))
    y = torch.empty_like(x)
    rep = audit(lambda: _scale_k[(1,)](x, y, B=4096), spec={"y_ptr": lambda: x.abs().double() * 2})
    m = rep["measured"]["y_ptr"]
    assert not m["significant"] and m["alignment"] == 0 and not rep["findings"]
    assert "未发现显著偏差" in rep["text"]
