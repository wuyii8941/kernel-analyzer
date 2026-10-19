"""Dispatch can hide a library implementation's internal ATen producers."""
from types import SimpleNamespace

import torch

from kernel_analyzer.provenance import _Trace


def test_dispatch_with_nested_launches_reports_the_unobserved_region():
    recorder = SimpleNamespace(launches=[])
    trace = _Trace(recorder)
    library = torch.library.Library("ka_trace_visibility", "DEF")
    library.define("forward(Tensor x) -> Tensor")

    def implementation(x):
        scratch = x.clone()
        recorder.launches.extend([object(), object()])
        return scratch + 1

    library.impl("forward", implementation, "CPU")
    x = torch.tensor([1.0, 2.0])
    with trace.mode:
        y = torch.ops.ka_trace_visibility.forward(x)
        z = y.clone()
    assert torch.equal(z, x + 1)
    # The dispatcher implementation really executes both operations; the mode
    # sees neither. The ordinary clone after the call remains visible.
    assert [str(e[2]) for e in trace.events] == [
        "ka_trace_visibility.forward.default", "aten.clone.default"]
    assert trace.enclosed_launches == [{
        "operator": "ka_trace_visibility.forward.default",
        "first_launch": 0, "stop_launch": 2,
    }]


def test_visible_aten_producers_between_launches_are_not_hidden():
    recorder = SimpleNamespace(launches=[object()])
    trace = _Trace(recorder)
    x = torch.tensor([1.0, 2.0])
    with trace.mode:
        y = x.clone()
        recorder.launches.append(object())
        z = y + 1
    assert torch.equal(z, x + 1)
    assert [e[0] for e in trace.events] == [1, 2]
    assert trace.enclosed_launches == []
