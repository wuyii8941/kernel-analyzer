import pytest
import torch
from kernel_analyzer.forward_recurrence_diagnostic import analyze
from kernel_analyzer.forward_state_recurrence_reference import decode_inputs, evaluate


def fixture():
    dimensions = dict(steps=3, channels=2, state_width=2, packed_width=6, state_offset=2)
    inputs = {'in_ptr0': torch.zeros(2, 2)}
    for i, size in [(1, 6), (2, 2), (3, 18), (4, 6)]:
        inputs[f'in_ptr{i}'] = torch.ones(size, dtype=torch.bfloat16)
    ref = evaluate(**decode_inputs(inputs, **dimensions, dtype=torch.float64))['states']
    outputs = {f'out_ptr{i}': ref[i].to(torch.bfloat16 if i == 2 else torch.float32)
               for i in range(3)}
    return inputs, outputs, dimensions


def test_stored_output_decomposition_does_not_claim_candidate_cast():
    inputs, outputs, dimensions = fixture()
    result = analyze(inputs, outputs, **dimensions)
    assert result['reconstruction_max_abs'] < 1e-15
    assert torch.count_nonzero(result['final_reference_cast_difference']) > 0
    assert torch.count_nonzero(result['final_difference_from_rounded_reference']) == 0
    assert not result['actual_final_candidate_cast_error_identified']
    assert not result['new_mechanism_confirmed']


def test_injected_difference_is_recorded_and_reconstructed():
    inputs, outputs, dimensions = fixture()
    outputs['out_ptr1'][0, 0] += 1
    result = analyze(inputs, outputs, **dimensions)
    assert result['state_difference'][1, 0, 0] > .99
    assert result['reconstruction_max_abs'] < 1e-15


@pytest.mark.parametrize('fault', ['missing', 'dtype', 'nonfinite'])
def test_reject_invalid_outputs(fault):
    inputs, outputs, dimensions = fixture()
    if fault == 'missing': del outputs['out_ptr1']
    if fault == 'dtype': outputs['out_ptr2'] = outputs['out_ptr2'].float()
    if fault == 'nonfinite': outputs['out_ptr0'][0, 0] = float('nan')
    with pytest.raises(ValueError): analyze(inputs, outputs, **dimensions)
