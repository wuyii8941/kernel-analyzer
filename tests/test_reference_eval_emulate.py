"""Bitwise emulation of captured launches and in-kernel localization (needs CUDA)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gmpy2")
pytest.importorskip("triton")

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.emulate import HardwareOracle, localize, verify  # noqa: E402
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
