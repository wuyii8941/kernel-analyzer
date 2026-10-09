"""Bitwise emulation of captured launches and in-kernel localization (needs CUDA)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gmpy2")
pytest.importorskip("triton")

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.emulate import HardwareOracle, _compare, emulate, localize, verify  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capture(fn):
    rec = TritonLaunchRecorder()
    with rec:
        fn()
        torch.cuda.synchronize()
    return rec.launches[-1]


def test_oracle_renames_operands_before_the_result():
    text = """module {
  tt.func public @k(%y: f32) {
    %y_11 = math.rsqrt %y : f32 loc(#loc52)
    tt.return
  }
}"""
    op = next(o for o in parse_ttir(text).entry().walk() if o.name == "math.rsqrt")
    line = HardwareOracle.rewrite(op, {"%y": "%x0"}, "%y")
    assert line == "%y = math.rsqrt %x0 : tensor<1024xf32>"


@cuda
def test_softmax_and_layernorm_reproduce_bitwise():
    from scripts import reference_eval_kernels as k

    g = torch.Generator(device="cuda").manual_seed(5)
    rows = torch.randn(8, 100, device="cuda", generator=g) * 3
    softmax = _capture(lambda: k.softmax_rows[(8,)](rows, torch.empty_like(rows), 100, 100, BLOCK=128))
    w, b = torch.randn(100, device="cuda", generator=g), torch.randn(100, device="cuda", generator=g)
    layernorm = _capture(lambda: k.layernorm_rows[(8,)](rows, w, b, torch.empty_like(rows), 100, 1e-5, BLOCK=128))
    for launch in (softmax, layernorm):
        r = verify(launch)
        assert r["bitwise_reproduced"]
        assert all(v["emulated"] == v["written"] for v in r["buffers"].values())
        assert not r["lowering_choices"]["uncontracted_from_output"]
        assert not r["lowering_choices"]["swapped_from_output"]
    # layernorm exercises contraction into the reduction and the div.full -> sub contraction
    nodes = verify(layernorm)["nodes"]
    assert nodes.get("reduce_tree_fused") and nodes.get("oracle_group")


@cuda
def test_tensor_core_dot_is_not_emulable():
    from scripts import reference_eval_kernels as k

    A = torch.randn(64, 48, device="cuda")
    B = torch.randn(48, 32, device="cuda")
    C = torch.empty(64, 32, device="cuda")
    launch = _capture(lambda: k.matmul[(2, 2)](A, B, C, 64, 32, 48, *A.stride(), *B.stride(), *C.stride(),
                                               BM=32, BN=16, BK=16, PRECISION="tf32"))
    r = verify(launch)
    assert r["buffers"]["C"]["emulated"] == 0
    assert any("tensor-core" in n for n in r["not_emulable_nodes"])


@cuda
def test_directed_rounding_node_is_localized():
    from scripts import mutation_kernels as mk

    x = torch.randn(2048, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    y = torch.empty_like(x)
    launch = _capture(lambda: mk.m_scale[(8,)](x, y, 0.1, 2048, BLOCK=256, MUT=1))
    assert verify(launch)["bitwise_reproduced"]
    rows = localize(launch)["nodes"]
    top = max(rows, key=lambda r: abs(r["buffers"]["Y"]["mean"]))
    assert "__nv_fmul_rd" in top["text"]
    b = top["buffers"]["Y"]
    assert b["mean"] < 0 and b["positive"] == 0


def test_comparison_uses_storage_bits():
    written = np.ones(3, dtype=bool)
    lo = np.array([-0.0, 1.0, 0.0])
    st = np.array([0, 0, 1], dtype=np.int8)  # the third element is emulated as NaN
    actual = np.array([0.0, 1.0, np.nan])
    bits = np.array([0.0, 1.0, np.nan], dtype=np.float32).view(np.uint32).astype(np.uint64)
    m = _compare(written, lo, lo, st, actual, "f32", bits)
    assert not m["bit_equal"][0] and m["value_equal"][0]  # -0 against +0: equal values, different bits
    assert m["bit_equal"][1]
    assert m["nan_both"][2] and not m["bit_equal"][2]  # NaN only as a class


@cuda
def test_atomic_kernel_is_not_established():
    from scripts import reference_eval_kernels as k

    x = torch.randn(1000, device="cuda")
    launch = _capture(lambda: k.atomic_accumulate[(4,)](x, torch.zeros(4, device="cuda"), torch.empty_like(x), 1000,
                                                        BLOCK=256, USE_OLD=False))
    r = verify(launch)
    assert r["status"] == "not_established" and not r["bitwise_reproduced"]


@cuda
def test_rounded_substitution_predicts_the_div_rn_kernel():
    """Model intervention (division node correctly rounded) against the real kernel change."""

    from scripts import mutation_kernels as mk

    x = torch.randn(16, 200, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4)) * 3
    outs = {}
    for mut in (0, 1):
        y = torch.empty_like(x)
        outs[mut] = (_capture(lambda: mk.m_softmax[(16,)](x, y, 200, 200, BLOCK=256, MUT=mut)), y)
    launch = outs[0][0]
    div = next(o.node_id for o in parse_ttir(launch.asm["ttir"]).entry().walk() if o.name == "arith.divf")
    res, _ = emulate(launch, exact_nodes=[div], substitution="rn")
    buf = next(b for b in res.buffers.values() if b.name == "Y")
    real = outs[1][1].reshape(-1).double().cpu().numpy()
    assert np.array_equal(buf.lo[buf.written], real[buf.written[: real.size]])


def test_zero_sign_follows_ieee():
    from kernel_analyzer.reference_eval.emulate import _zero_sign
    from kernel_analyzer.reference_eval.ttir_eval import _ftv

    def tv(v):
        v = np.array(v, dtype=np.float64)
        return _ftv("f32", v, v.copy(), np.zeros(v.shape, dtype=np.int8), np.zeros(v.shape, dtype=bool), frozenset())

    out = tv([0.0, 0.0, 0.0, 0.0])
    a, b = tv([1.5, -0.0, -0.0, 0.0]), tv([-1.5, -0.0, 0.0, -0.0])
    r = _zero_sign(out, "add", [a, b])
    assert list(np.signbit(r.lo)) == [False, True, False, False]  # x + (-x) = +0; -0 + -0 = -0
    r = _zero_sign(out, "fma", [tv([-2.0, 0.0, -1.0, 2.0]), tv([0.0, 3.0, 0.0, 0.5]), tv([-0.0, -0.0, 0.0, -1.0])])
    assert list(np.signbit(r.lo)) == [True, False, False, False]  # (-2*0) + -0 = -0; 0*3 + -0 = +0; 2*0.5 - 1 = +0


def test_locations_link_ttir_ops_to_ptx_and_sass():
    from kernel_analyzer.reference_eval.emulate import (op_location, ptx_instructions_by_location,
                                                        sass_instructions_by_line, ttir_locations)

    ttir = """module {
  tt.func public @k(%a: f32) {
    %t = arith.mulf %a, %a : f32 loc(#loc3)
    tt.return
  }
}
#loc1 = loc("/x/kern.py":12:7)
#loc3 = loc("t"(#loc1))
"""
    op = next(o for o in parse_ttir(ttir).entry().walk() if o.name == "arith.mulf")
    loc = op_location(op, ttir_locations(ttir), {"%t": "    %t = arith.mulf %a, %a : f32 loc(#loc3)"})
    assert loc == ("kern.py", 12, 7)
    ptx = "\t.loc\t1 12 7\n\tmul.f32 \t%r1, %r2, %r2;\n\t.loc\t1 13 0\n\tadd.f32 \t%r3, %r1, %r4;\n\t.file\t1 \"/x/kern.py\"\n"
    assert ptx_instructions_by_location(ptx)[("kern.py", 12, 7)] == ["mul.f32"]
    sass = '\t//## File "/x/kern.py", line 13\n        /*0010*/                   FADD R3, R1, R4 ;\n'
    assert sass_instructions_by_line(sass)[("kern.py", 13)] == ["FADD"]
