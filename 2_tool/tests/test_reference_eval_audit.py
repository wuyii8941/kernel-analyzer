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
    # scaled so that the largest |K - S| (~4e-4) is under the atol 1e-2 of the Triton tutorial's test
    a = 0.1 * torch.randn(64, 128, device="cuda", generator=g)
    b = 0.1 * torch.randn(128, 64, device="cuda", generator=g)
    c = torch.empty(64, 64, device="cuda")
    rep = audit(lambda: _dot_k[(1,)](a, b, c, M=64, N=64, K=128, IEEE=False),
                spec={"c_ptr": lambda: a.double() @ b.double()})
    m = rep["measured"]["c_ptr"]
    assert m["significant"] and m["alignment"] < -5e-4 and m.get("prediction_agrees")
    top = rep["attribution"]["c_ptr"]["nodes"][0]
    assert top["node"].startswith("tt.dot") and "tf32" in top["mechanism"]
    group = rep["fix_groups"][0]
    assert any("ieee" in text for text, _ in group["fixes"])
    assert abs(group["outputs"]["c_ptr"]["after_all"]) < 0.1 * abs(m["alignment"])
    assert abs(group["outputs"]["c_ptr"]["before"] - m["alignment"]) < 1e-3 * abs(m["alignment"])   # same elements
    assert abs(rep["spec"]["c_ptr"]["semantic_alignment"]) < 1e-9          # the program is the specification
    loose, strict = rep["tolerance_checks"]["c_ptr"][0], rep["tolerance_checks"]["c_ptr"][2]
    assert loose["passes"] and not strict["passes"]                        # atol 1e-2 lets the shrink through
    assert "会被漏掉" in rep["text"] and "偏小" in rep["text"].splitlines()[0]
    assert [p for p, *_ in m["per_program"]] == [[0, 0, 0]] and "program_range" not in m     # one program


@cuda
def test_audit_reports_no_finding_for_an_exact_kernel():
    x = torch.randn(4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(15))
    y = torch.empty_like(x)
    rep = audit(lambda: _scale_k[(1,)](x, y, B=4096), spec={"y_ptr": lambda: x.abs().double() * 2})
    m = rep["measured"]["y_ptr"]
    assert not m["significant"] and m["alignment"] == 0 and not rep["findings"]
    assert "未发现显著偏差" in rep["text"]


@triton.jit
def _rows_k(x_ptr, y_ptr, B: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * B + tl.arange(0, B)
    x = tl.load(x_ptr + r)
    y = x.to(tl.bfloat16).to(tl.float32) if pid == 0 else x               # only program 0 rounds
    tl.store(y_ptr + r, y)


@cuda
def test_audit_reports_the_spread_between_programs():
    """x = 1 + k 2^-7 + 2^-10 always rounds down to bf16 (2^-10 is under half its ulp): program 0 shrinks, the others
    are exact; the spread between programs is reported."""
    k = torch.randint(0, 128, (4 * 1024,), device="cuda", generator=torch.Generator(device="cuda").manual_seed(16))
    x = 1 + k.float() * 2 ** -7 + 2 ** -10
    y = torch.empty_like(x)
    rep = audit(lambda: _rows_k[(4,)](x, y, B=1024), programs="all", predict=False)
    m = rep["measured"]["y_ptr"]
    assert [p for p, *_ in m["per_program"]] == [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    assert all(a == 0 for _, a, _ in m["per_program"][1:]) and m["per_program"][0][1] < -5e-4
    assert m["significant"] and m["programs_differ"] and m["program_range"][1] == 0
    assert "各 program 的偏差差别大" in rep["text"] and "全部 4 个 program" in rep["text"]


@triton.jit
def _round_k(x_ptr, y_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    tl.store(y_ptr + r, tl.load(x_ptr + r).to(tl.bfloat16).to(tl.float32))


@cuda
def test_tolerance_checks_tell_the_bias_from_the_random_error():
    """Every element of x = 1 + k 2^-7 + 2^-10 (in [1, 2)) rounds down by 2^-10 to bf16.  With the systematic
    component a x removed (a = <e, x> / <x, x> ~ -0.64 2^-10) the residual -2^-10 - a x stays within
    [-0.36, 0.29] 2^-10: atol 0.5 2^-10 is failed because of the bias, atol 0.1 2^-10 either way, atol 2 2^-10
    passes."""
    k = torch.randint(0, 128, (4096,), device="cuda", generator=torch.Generator(device="cuda").manual_seed(17))
    x = 1 + k.float() * 2 ** -7 + 2 ** -10
    y = torch.empty_like(x)
    u = 2.0 ** -10
    rep = audit(lambda: _round_k[(1,)](x, y, B=4096), spec={"y_ptr": lambda: x.double()}, predict=False,
                tolerance_checks=[("loose", 2 * u, 0.0), ("tight", 0.5 * u, 0.0), ("noisy", 0.1 * u, 0.0)])
    loose, tight, noisy = rep["tolerance_checks"]["y_ptr"]
    assert loose["passes"] and loose["passes_without_bias"]
    assert not tight["passes"] and tight["passes_without_bias"]
    assert not noisy["passes"] and not noisy["passes_without_bias"]
    assert "loose 会通过" in rep["text"] and "tight 会因为它报错" in rep["text"] and "noisy 去掉这个偏差也通不过" in rep["text"]
