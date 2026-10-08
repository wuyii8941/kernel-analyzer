"""Detector 2.3: an output whose Triton producer reads a buffer made by a torch / ATen op inside the launch must be marked
as depending on a non-Triton intermediate (its e_sem is mixed), also when the output buffer already holds the identical
bytes before the launch -- the caching allocator hands back the buffer of a warm-up on the same inputs.  2.2 decided
"written" by a byte change and missed that case (closure F/FR run, 2026-10-08, composition C1)."""
from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from kernel_analyzer import check  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@triton.jit
def _plus_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)


N = 512


def _capture(fn, inp):
    rec = TritonLaunchRecorder()
    with rec:
        out = fn()
        torch.cuda.synchronize()
    seq = evaluate_sequence(rec.launches, masked_fill_zero=True)
    return out, check.torch_intermediates(rec.launches, seq, inp)


def test_output_with_identical_stale_bytes_still_depends_on_the_aten_intermediate():
    x = torch.randn(N, device="cuda", generator=None)
    inp = {"x": x}
    y = torch.empty(N, device="cuda")
    h = x * 3.0                                              # ATen intermediate
    _plus_one[(N // 128,)](h, y, N, BLOCK=128)               # warm-up: y already holds the result

    def run():
        h2 = x * 3.0
        _plus_one[(N // 128,)](h2, y, N, BLOCK=128)          # rewrites identical bytes
        return y
    out, deps = _capture(run, inp)
    assert deps.get(out.untyped_storage().data_ptr()), "the ATen intermediate must be reported upstream of y"


def test_output_reading_only_case_inputs_has_no_upstream_intermediate():
    x = torch.randn(N, device="cuda")
    inp = {"x": x}
    y = torch.empty(N, device="cuda")
    _plus_one[(N // 128,)](x, y, N, BLOCK=128)

    def run():
        _plus_one[(N // 128,)](x, y, N, BLOCK=128)
        return y
    out, deps = _capture(run, inp)
    assert not deps.get(out.untyped_storage().data_ptr())


def test_kernel_reference_records_the_storages_it_stored():
    x = torch.randn(N, device="cuda")
    y = torch.empty(N, device="cuda")
    rec = TritonLaunchRecorder()
    with rec:
        _plus_one[(N // 128,)](x, y, N, BLOCK=128)
        torch.cuda.synchronize()
    seq = evaluate_sequence(rec.launches, masked_fill_zero=True)
    assert y.untyped_storage().data_ptr() in seq.launches[0].stored
    assert x.untyped_storage().data_ptr() not in seq.launches[0].stored


def test_input_modified_in_place_by_aten_inside_the_launch_is_an_intermediate():
    """the case input is renormalised in place by an ATen op inside the launch (as F.embedding(max_norm=...) does to
    the weight) before the Triton kernel reads it: the read buffer is no longer the input the case gave."""
    class Case(check.Case):
        name = "inplace_input"

        def inputs(self, seed):
            return {"w": torch.randn(N, device="cuda", generator=None)}

        def launch(self, inp):
            inp["w"].mul_(0.5)                                   # ATen, in place on the input
            y = torch.empty(N, device="cuda")
            _plus_one[(N // 128,)](inp["w"], y, N, BLOCK=128)
            return {"y": y}

    rep = check.run(Case(), dev=[0], conf=[1, 2])
    assert rep["outputs"]["y"]["depends_on_non_triton_intermediates"], "the in-place modified input must count as upstream"
