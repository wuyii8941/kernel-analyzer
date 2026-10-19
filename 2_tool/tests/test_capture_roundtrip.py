"""Saved captures must retain execution evidence and indirectly addressed state."""
import json

import numpy as np
import pytest
import torch

from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch, load_launch, save_launch


def launch():
    implicit = CapturedArg(-1, "implicit0", "tensor", False, None,
                          dtype="uint8", shape=(2,), stride=(1,), storage_ptr=100,
                          data_ptr=101, element_size=1, storage_nbytes=4,
                          before=torch.tensor([7, 8], dtype=torch.uint8),
                          after=torch.tensor([9, 10], dtype=torch.uint8),
                          window=np.array([1, 2], dtype=np.int64))
    return CapturedLaunch(0, "kernel", "", (1, 1, 1), [], {}, None, {}, None,
                          implicit=[implicit], executed=False)


def test_capture_preserves_indirect_memory_and_execution(tmp_path):
    original = launch()
    original.call_stack = [{"file": "/example/generated.py", "line": 12, "function": "call"}]
    save_launch(original, tmp_path)
    restored = load_launch(tmp_path)
    assert restored.executed is False
    assert restored.call_stack == original.call_stack
    assert len(restored.implicit) == 1
    before, after = original.implicit[0], restored.implicit[0]
    for field in ("name", "index", "shape", "stride", "data_ptr", "storage_ptr", "storage_nbytes"):
        assert getattr(before, field) == getattr(after, field)
    assert torch.equal(before.before, after.before)
    assert torch.equal(before.after, after.after)
    assert np.array_equal(before.window, after.window)


def test_old_capture_does_not_invent_execution_evidence(tmp_path):
    save_launch(launch(), tmp_path)
    path = tmp_path / "launch.json"
    data = json.loads(path.read_text())
    data.pop("executed", None)
    data.pop("implicit", None)
    data.pop("call_stack", None)
    path.write_text(json.dumps(data))
    restored = load_launch(tmp_path)
    assert restored.executed is None
    assert restored.implicit_recorded is False
    assert restored.call_stack is None
    save_launch(restored, tmp_path / "resaved")
    assert load_launch(tmp_path / "resaved").implicit_recorded is False


def test_replay_refuses_unmapped_indirect_addresses_before_launch():
    from kernel_analyzer.reference_eval.capture import replay
    with pytest.raises(ValueError, match="indirect"):
        replay(launch(), kernel=object(), device="cpu")


def test_unexecuted_capture_cannot_certify_replay():
    from kernel_analyzer.reference_eval.capture import check_replay
    for state in (False, None):
        capture = launch()
        capture.executed = state
        with pytest.raises(ValueError, match="executed"):
            check_replay(capture, kernel=object())


def test_direct_and_indirect_snapshots_have_separate_storage_keys(tmp_path):
    original = launch()
    original.executed = True
    original.args = [CapturedArg(-1, "x", "tensor", False, None, dtype="uint8", shape=(1,), stride=(1,),
                                before=None, after=torch.tensor([42], dtype=torch.uint8))]
    save_launch(original, tmp_path)
    restored = load_launch(tmp_path)
    assert restored.executed is True
    assert restored.implicit_recorded is True
    assert restored.args[0].before is None
    assert restored.args[0].after.tolist() == [42]
    assert restored.implicit[0].after.tolist() == [9, 10]


def test_replay_cannot_report_success_without_any_execution():
    from kernel_analyzer.reference_eval.capture import check_replay
    original = launch()
    original.executed = True
    original.implicit = []
    for times in (0, -1, False, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            check_replay(original, kernel=object(), times=times)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_capture_roundtrip_preserves_views_and_actual_execution(tmp_path):
    import triton
    import triton.language as tl
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, check_replay

    @triton.jit
    def copy_view(x, y, SX: tl.constexpr, SY: tl.constexpr, N: tl.constexpr):
        offsets = tl.arange(0, N)
        tl.store(y + offsets * SY, tl.abs(tl.load(x + offsets * SX)) * 2)

    x_base = torch.arange(-32, 32, dtype=torch.float32, device="cuda")
    y_base = torch.full((64,), -7, dtype=torch.float32, device="cuda")
    x, y = x_base[3:35:2], y_base[5:53:3]
    compiled = copy_view[(1,)](x, y, 2, 3, 16)
    y_base.fill_(-7)
    before = y_base.cpu().clone()
    with TritonLaunchRecorder() as recorder:
        copy_view[(1,)](x, y, 2, 3, 16)
    assert len(recorder.launches) == 1
    original = recorder.launches[0]
    assert any(frame["function"] == "test_device_capture_roundtrip_preserves_views_and_actual_execution"
               for frame in original.call_stack)
    assert original.executed is True
    assert original.implicit_recorded is True
    assert original.implicit == []
    expected = before.clone()
    expected[5:53:3] = x.cpu().abs() * 2
    assert torch.equal(y_base.cpu(), expected)
    save_launch(original, tmp_path)
    restored = load_launch(tmp_path)
    assert restored.call_stack == original.call_stack
    for live, saved in zip(original.args, restored.args):
        if live.kind != "tensor":
            continue
        assert saved.shape == live.shape
        assert saved.stride == live.stride
        assert saved.data_ptr - saved.storage_ptr == live.data_ptr - live.storage_ptr
        assert torch.equal(saved.before, live.before)
        assert torch.equal(saved.after, live.after)
        if live.name == "y":
            assert torch.equal(saved.before, before.view(torch.uint8))
            assert torch.equal(saved.after, expected.view(torch.uint8))
    assert check_replay(restored, kernel=compiled, times=2)["bitwise_reproduced"] is True
