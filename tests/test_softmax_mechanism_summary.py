import pytest
import torch
from scripts.summarize_softmax_mechanism import summarize_record
from kernel_analyzer.softmax_saved_state_diagnostic import evaluate


def test_recompute_not_trust_saved_numbers():
    outputs = dict(in_out_ptr0=torch.zeros(2, 2, dtype=torch.bfloat16),
                   out_ptr0=torch.zeros(2), out_ptr1=torch.ones(2)*4)
    record = dict(post_call_outputs=outputs, checked_scale=.5, symbol='k', invocation_index=0,
                  diagnostic=evaluate(outputs['in_out_ptr0'], outputs['out_ptr0'], outputs['out_ptr1'], scale=.5))
    summary = summarize_record(record)
    assert summary['normalization_defect_mean'] == -.5
    assert not summary['mechanism_novelty_established']
    record['diagnostic']['row_mass'].zero_()
    with pytest.raises(ValueError): summarize_record(record)
    report = summarize_record(record, report_recompute_differences=True)
    assert report['recompute_max_absolute_differences']['row_mass'] == .5
    assert not report['recompute_bitwise_identical']
    assert report['normalization_defect_mean'] == -.5
