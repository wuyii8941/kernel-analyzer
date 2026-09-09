import torch
import pytest
from kernel_analyzer.forward_state_recurrence_reference import evaluate
from kernel_analyzer.forward_state_recurrence_reference import decode_inputs


def pointer_fixture():
    return dict(in_ptr0=torch.zeros(2, 2),
                in_ptr1=torch.arange(6).to(torch.bfloat16),
                in_ptr2=torch.zeros(2, dtype=torch.bfloat16),
                in_ptr3=torch.arange(18).to(torch.bfloat16),
                in_ptr4=torch.arange(6).to(torch.bfloat16))


def decode(pointers):
    return decode_inputs(pointers, steps=3, channels=2, state_width=2,
                         packed_width=6, state_offset=2, dtype=torch.float64)


def test_pointer_layout_and_snapshot():
    pointers = pointer_fixture()
    decoded = decode(pointers)
    assert decoded['time_inputs'].tolist() == [[0, 1], [2, 3], [4, 5]]
    assert decoded['input_signal'].tolist() == [[0, 3], [1, 4], [2, 5]]
    assert decoded['input_state'].tolist() == [[2, 3], [8, 9], [14, 15]]
    pointers['in_ptr4'].zero_()
    assert decoded['input_signal'][-1].tolist() == [2, 5]


@pytest.mark.parametrize('fault', ['dtype', 'size', 'stride', 'nonfinite'])
def test_reject_invalid_runtime_storage(fault):
    pointers = pointer_fixture()
    if fault == 'dtype': pointers['in_ptr1'] = pointers['in_ptr1'].float()
    if fault == 'size': pointers['in_ptr3'] = pointers['in_ptr3'][:-1]
    if fault == 'stride': pointers['in_ptr4'] = torch.zeros(12, dtype=torch.bfloat16)[::2]
    if fault == 'nonfinite': pointers['in_ptr0'][0, 0] = float('nan')
    with pytest.raises(ValueError): decode(pointers)


def test_shared_dt_controls_decay_and_input():
    rate = torch.zeros(2, 3, dtype=torch.float64)
    time = torch.zeros(4, 2, dtype=torch.float64)
    bias = torch.zeros(2, dtype=torch.float64)
    b = torch.ones(4, 3, dtype=torch.float64)
    signal = torch.ones(4, 2, dtype=torch.float64)
    result = evaluate(rate, time, bias, b, signal)
    dt = torch.log(torch.tensor(2., dtype=torch.float64))
    previous = torch.zeros_like(rate)
    for i in range(4):
        previous = .5*previous + dt
        assert torch.allclose(result['states'][i], previous, rtol=0, atol=1e-15)
    assert not result['population_bias_proved']


def test_zero_injection_and_nonfinite_rejection():
    rate = torch.zeros(1, 2)
    time = torch.zeros(3, 1)
    bias = torch.zeros(1)
    b = torch.ones(3, 2)
    signal = torch.zeros(3, 1)
    assert torch.count_nonzero(evaluate(rate,time,bias,b,signal)['states']) == 0
    time[0] = float('nan')
    with pytest.raises(ValueError): evaluate(rate,time,bias,b,signal)
