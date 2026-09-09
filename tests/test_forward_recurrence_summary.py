import pytest
import torch
from kernel_analyzer.forward_recurrence_diagnostic import analyze
from scripts.summarize_forward_recurrence_mechanism import summarize_record


def test_recomputes_and_reports_modified_diagnostic():
    dims = dict(steps=2, channels=1, state_width=1, packed_width=3, state_offset=1)
    inputs = {'in_ptr0': torch.zeros(1)}
    inputs.update({f'in_ptr{i}': torch.ones(n, dtype=torch.bfloat16)
                   for i, n in ((1, 2), (2, 1), (3, 6), (4, 2))})
    outputs = dict(out_ptr0=torch.ones(1), out_ptr1=torch.ones(1, dtype=torch.bfloat16))
    diagnostic = analyze(inputs, outputs, **dims)
    record = dict(dimensions=dims, function_ast_sha256='fixture', symbol='recurrence',
                  invocation_index=0, pre_call_inputs=inputs, post_call_outputs=outputs,
                  diagnostic=diagnostic)
    contract = dict(dims, function_ast_sha256='fixture')
    result = summarize_record(record, contract)
    assert all(x == 0 for x in result['recompute_max_absolute_differences'].values())
    diagnostic['state_difference'][0] += 1
    changed = summarize_record(record, contract)
    assert changed['recompute_max_absolute_differences']['state_difference'] == 1
    assert changed['step_relative_rms'] == result['step_relative_rms']
    assert not changed['new_mechanism_confirmed']
    with pytest.raises(ValueError):
        summarize_record(record, dict(contract, function_ast_sha256='wrong'))
