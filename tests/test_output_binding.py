"""Detector 2.2 regression (docs/protocol_essential_bugs_phase2_20261007.md, item A1): an output produced by an ATen
op at the address of a freed Triton intermediate must not be bound to that intermediate's writes.

Three scenarios (the intermediate A is written by a Triton kernel and freed; the returned output B is produced by
ATen): (1) B has a different size, (2) the same size and different content, (3) the same size and the same content.
Each runs twice: with the recorder's default storage keep-alive (the address cannot be reused inside the recorded
region: the output must come out as not written by Triton), and with the keep-alive disabled (the address IS reused;
identities of freed storages are not unique, so the binding must come out as not established - never as bound).

    PYTHONPATH=src python -m pytest -q tests/test_output_binding.py
"""
import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from kernel_analyzer import check  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
N = 1024


@triton.jit
def _double(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)


class _Case(check.Case):
    name = "binding"

    def __init__(self, scenario):
        self.scenario = scenario
        self.addresses = []

    def inputs(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return {"x": torch.randn(N, generator=g).cuda()}

    def launch(self, inp):
        x = inp["x"]
        a = torch.empty_like(x)
        _double[(triton.cdiv(N, 256),)](x, a, N, BLOCK=256)
        a_addr = a.untyped_storage().data_ptr()
        del a                                                   # the Triton intermediate is freed
        if self.scenario == "different_size":
            b = torch.full((N + 5,), 1.0, device="cuda")
        elif self.scenario == "same_size_other_content":
            b = torch.full((N,), 3.0, device="cuda")
        else:                                                   # same size, same content as the intermediate
            b = x * 2.0
        self.addresses.append(b.untyped_storage().data_ptr() == a_addr)
        return {"out": b}


def _run(scenario, keep):
    original = TritonLaunchRecorder.__init__

    def init(self, *a, **k):
        k.setdefault("keep_storages", keep)
        original(self, *a, **k)

    TritonLaunchRecorder.__init__ = init
    try:
        case = _Case(scenario)
        # one launch per input: the repeated launches of tool 4.0 run outside the recorded region and would add
        # address reuses this test does not count (it is about binding inside the recorded region)
        report = check.run(case, dev=[0], conf=[1, 2], repeats=1)
    finally:
        TritonLaunchRecorder.__init__ = original
    return case, report


@pytest.mark.parametrize("scenario", ["different_size", "same_size_other_content", "same_size_same_content"])
@pytest.mark.parametrize("keep", [True, False])
def test_aten_output_at_freed_triton_address_is_not_bound(scenario, keep):
    case, report = _run(scenario, keep)
    assert float(report["tool_version"]) >= 2.2          # binding by storage identity since 2.2
    assert "out" not in report["outputs"]                      # never bound to the intermediate's writes
    if keep:
        assert not any(case.addresses), "keep-alive must prevent address reuse inside the recorded region"
        assert "out" in report["outputs_not_written_by_triton"]
    else:
        # without keep-alive the address (and the StorageImpl address) may be reused: binding is not provable
        assert any(case.addresses), "the scenario must actually reuse the freed address"
        assert "out" in report["outputs_binding_not_established"]


def test_triton_written_output_is_still_bound():
    class Written(_Case):
        def launch(self, inp):
            x = inp["x"]
            y = torch.empty_like(x)
            _double[(triton.cdiv(N, 256),)](x, y, N, BLOCK=256)
            return {"out": y}

    report = check.run(Written("written"), dev=[0], conf=[1, 2])
    assert report["outputs_not_written_by_triton"] == []
    assert report["outputs_binding_not_established"] == []
    assert "out" in report["outputs"]
