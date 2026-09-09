import pytest
import torch
from kernel_analyzer.softmax_same_call_capture import call_and_capture


def buffers():
    return dict(in_out_ptr0=torch.ones(2, 2, dtype=torch.bfloat16),
                in_ptr0=torch.zeros(2, dtype=torch.int64),
                out_ptr0=torch.empty(2), out_ptr1=torch.empty(2),
                out_ptr2=torch.empty(2, 2, dtype=torch.bfloat16))


def test_one_invocation_preserves_both_sides_without_mutation():
    p = buffers()
    calls, records = [], []
    def run():
        calls.append(1)
        p['in_out_ptr0'].zero_()
        p['out_ptr0'].zero_()
        p['out_ptr1'].fill_(2)
        p['out_ptr2'].fill_(.5)
        return 'original-result'
    result = call_and_capture(run, p, rows=2, width=2, scale=.5,
                              invoke=lambda f: f(), sink=records.append)
    assert result == 'original-result' and calls == [1]
    assert records[0]['pre_call_inputs']['in_out_ptr0'].sum() == 4
    assert records[0]['post_call_outputs']['in_out_ptr0'].sum() == 0
    assert torch.count_nonzero(records[0]['diagnostic']['normalization_defect']) == 0
    p['out_ptr2'].zero_()
    assert records[0]['post_call_outputs']['out_ptr2'].sum() == 2


def test_failed_call_never_emits_success_record():
    records = []
    def fail():
        raise RuntimeError('kernel failed')
    with pytest.raises(RuntimeError):
        call_and_capture(fail, buffers(), rows=2, width=2, scale=.5,
                         invoke=lambda f: f(), sink=records.append)
    assert records == []


@pytest.mark.parametrize('count', [0, 2])
def test_missing_or_repeated_invocation_rejected(count):
    records = []
    with pytest.raises(ValueError):
        call_and_capture(lambda: None, buffers(), rows=2, width=2, scale=.5,
                         invoke=lambda f: [f() for _ in range(count)], sink=records.append)
    assert not records
