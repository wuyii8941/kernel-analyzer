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


@triton.jit
def _leaky_k(x_ptr, y_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    x = tl.load(x_ptr + r)
    tl.store(y_ptr + r, tl.where(x > 0, x, 0.01 * x))


@cuda
def test_reference_check_finds_an_error_orthogonal_to_the_reference():
    """A relu written as leaky relu: G - S is nonzero only where S = 0, so the alignment <G - S, S> / <S, S> is exactly 0;
    the element-wise check finds every negative input."""
    x = torch.randn(4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(18))
    y = torch.empty_like(x)
    rep = audit(lambda: _leaky_k[(1,)](x, y, B=4096), spec={"y_ptr": lambda: x.double().clamp(min=0)}, predict=False,
                reference="relu（torch，float64）")
    s = rep["spec"]["y_ptr"]
    assert s["semantic_alignment"] == 0
    assert s["verdict"] == "不一致" and s["exceeding"] == int((x < 0).sum()) and s["compared"] == 4096
    assert s["exceeding_by_program"] == [[[0, 0, 0], int((x < 0).sum())]]
    assert any(f["kind"] == "reference_difference" for f in rep["findings"])
    assert not any(f["kind"] == "semantic" for f in rep["findings"])
    assert s["semantic_verdict"] == "无法判断"
    assert "与参照程序不一致" in rep["text"].splitlines()[0] and "relu（torch，float64）" in rep["text"]
    assert "assert_close 的 dtype 默认值 不通过" in rep["text"] and "随机误差已超出" not in rep["text"]


def test_float64_reference_does_not_certify_semantics():
    """A large G-S difference alone cannot decide which program implements the task."""
    from types import SimpleNamespace
    from kernel_analyzer.reference_eval.audit import _compare_reference, _semantic_text, render
    buf = SimpleNamespace(global_indices=lambda: np.arange(2), written=np.ones(2, dtype=bool),
                          actual_after_st=np.zeros(2, dtype=int), writer=np.zeros(2, dtype=int), elem="f32")
    s = _compare_reference(buf, np.ones(2, dtype=bool), np.array([2., 2.]), np.zeros(2),
                           np.array([2., 2.]), np.ones(2), [(0, 0, 0)], 2**-24)
    assert s["verdict"] == "不一致"  # retain the useful implementation comparison
    assert s["semantic_verdict"] == "无法判断"
    assert s["reference_kind"] == "implementation_reference"
    assert "不能据此确认语义错误" in _semantic_text(s)
    assert "数值舍入解释不了" not in _semantic_text(s)
    rep = {"names": {}, "prediction": None, "verify": {"status": "bit_identical"},
           "measured": {"out": dict(alignment=0., se=0., energy_relative=0., max_abs=0.,
                                    elements=2, written=2, format_resolution=2**-24)},
           "spec": {"out": s}, "findings": [{"output": "out", "kind": "reference_difference",
                                               "text": _semantic_text(s)}],
           "attribution": {}, "fix_groups": [], "tolerance_checks": {}, "cannot_decide": [],
           "scope": dict(kernel="example", target="test", rule="all")}
    text = render(rep)
    assert "与参照程序不一致" in text.splitlines()[0]
    assert "语义正确性尚无法判断" in text
    assert "G − S（语义）" not in text
    assert "来自公式、常数或 mask，不是数值误差" not in text


def test_matching_callback_does_not_certify_task_or_inputs():
    from types import SimpleNamespace
    from kernel_analyzer.reference_eval.audit import _compare_reference, render
    buf = SimpleNamespace(global_indices=lambda: np.arange(2), written=np.ones(2, dtype=bool),
                          actual_after_st=np.zeros(2, dtype=int), writer=np.zeros(2, dtype=int), elem="f32")
    s = _compare_reference(buf, np.ones(2, dtype=bool), np.ones(2), np.zeros(2),
                           np.full(2, 1.1), np.ones(2), [(0, 0, 0)], 2**-24)
    rep = {"names": {}, "prediction": None, "verify": {"status": "bit_identical"},
           "measured": {"out": dict(alignment=0.1, se=0., energy_relative=0.1, max_abs=0.1,
                                    elements=2, written=2, format_resolution=2**-24, significant=True)},
           "spec": {"out": s}, "findings": [], "attribution": {}, "fix_groups": [],
           "tolerance_checks": {}, "cannot_decide": [],
           "scope": dict(kernel="example", target="test", rule="all")}
    text = render(rep)
    assert "问题出在数值实现" not in text
    assert "在 kernel 实际收到的输入上" not in text
    assert "语义正确性尚未建立" in text
    assert "输入与前态对应尚未核验" in text


def test_callback_values_follow_output_view_addresses():
    from types import SimpleNamespace
    from kernel_analyzer.reference_eval.audit import _reference_storage
    arg = SimpleNamespace(shape=(2, 2), stride=(1, 3), data_ptr=104, storage_ptr=100,
                          element_size=4, storage_nbytes=28)
    storage = _reference_storage(np.array([[10., 20.], [30., 40.]]), arg)
    assert np.isnan(storage[[0, 3, 6]]).all()
    assert storage[[1, 2, 4, 5]].tolist() == [10., 30., 20., 40.]
    with pytest.raises(ValueError, match="shape"):
        _reference_storage(np.array([10., 20., 30., 40.]), arg)
    for changes in ({"stride": (1, 1)}, {"storage_nbytes": 16}, {"data_ptr": 102}):
        with pytest.raises(ValueError):
            _reference_storage(np.ones((2, 2)), SimpleNamespace(**(vars(arg) | changes)))


@pytest.mark.parametrize("shape,stride,offset", [((2, 3), (5, 1), 2), ((3, 2), (1, 5), 1),
                                                ((), (), 2), ((0, 2), (2, 1), 0)])
def test_callback_storage_matches_independent_coordinate_enumeration(shape, stride, offset):
    from types import SimpleNamespace
    from kernel_analyzer.reference_eval.audit import _reference_storage
    values = np.arange(np.prod(shape, dtype=int), dtype=np.float64).reshape(shape)
    arg = SimpleNamespace(shape=shape, stride=stride, data_ptr=100 + offset * 4, storage_ptr=100,
                          element_size=4, storage_nbytes=64)
    expected = np.full(16, np.nan)
    for coordinate in np.ndindex(shape):
        expected[offset + sum(i * s for i, s in zip(coordinate, stride))] = values[coordinate]
    np.testing.assert_equal(_reference_storage(values, arg), expected)


@cuda
def test_reference_check_leaves_out_nonfinite_callback_values():
    x = torch.randn(4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(19))
    y = torch.empty_like(x)
    nan = torch.tensor(float("nan"), dtype=torch.float64, device="cuda")
    rep = audit(lambda: _scale_k[(1,)](x, y, B=4096), predict=False,
                spec={"y_ptr": lambda: torch.where(x < -1, nan, x.abs().double() * 2)})
    s, undefined = rep["spec"]["y_ptr"], int((x < -1).sum())
    assert undefined > 0 and s["undefined"] == undefined and s["compared"] == 4096 - undefined
    assert s["verdict"] == "未发现差异" and s["exceeding"] == 0 and s["semantic_alignment"] == 0
    assert rep["tolerance_checks"]["y_ptr"][0]["passes"] and not rep["findings"]
    assert f"参照程序在 {undefined} 个元素上返回非有限值" in rep["text"]
    assert "不等于数学目标无定义" in rep["text"]


@triton.jit
def _sqrt_k(x_ptr, y_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    tl.store(y_ptr + r, tl.sqrt_rn(tl.load(x_ptr + r)))


@cuda
def test_reference_check_counts_non_finite_kernel_outputs():
    """sqrt without the abs of the reference: NaN where x < 0, the reference is finite there."""
    x = torch.randn(4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(20))
    y = torch.empty_like(x)
    rep = audit(lambda: _sqrt_k[(1,)](x, y, B=4096), spec={"y_ptr": lambda: x.abs().double().sqrt()}, predict=False)
    s = rep["spec"]["y_ptr"]
    assert s["kernel_non_finite"] == int((x < 0).sum()) and s["exceeding"] == 0 and s["verdict"] == "不一致"
    assert "kernel 输出不是有限值" in rep["text"]
