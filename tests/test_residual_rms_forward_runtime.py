import pytest
import torch
from kernel_analyzer.residual_rms_forward_reference import from_runtime_pointers, select_output
from kernel_analyzer.residual_rms_forward_reference import snapshot_pre_call


def inputs():
    return dict(in_ptr0=torch.ones(6, dtype=torch.bfloat16),
                in_out_ptr0=torch.full((2, 3), .25, dtype=torch.bfloat16),
                in_ptr1=torch.ones(3, dtype=torch.bfloat16))


@pytest.mark.parametrize('pointer,index,dtype,size', [
    ('in_out_ptr0', 0, torch.bfloat16, 6),
    ('in_out_ptr1', 1, torch.float32, 2),
    ('out_ptr0', 2, torch.bfloat16, 6)])
def test_each_output_and_no_input_mutation(pointer, index, dtype, size):
    pointers = inputs()
    before = {key: value.clone() for key, value in pointers.items()}
    outputs = from_runtime_pointers(pointers, rows=2, width=3, epsilon=1e-6)
    selected = select_output(pointers, torch.empty(size, dtype=dtype),
                             formal_pointer=pointer, rows=2, width=3, epsilon=1e-6)
    assert torch.equal(selected, outputs[index].reshape(-1))
    assert all(torch.equal(value, before[key]) for key, value in pointers.items())


@pytest.mark.parametrize('change', ['missing', 'dtype', 'layout', 'size'])
def test_invalid_pre_call_buffers_rejected(change):
    pointers = inputs()
    if change == 'missing': del pointers['in_out_ptr0']
    if change == 'dtype': pointers['in_ptr0'] = pointers['in_ptr0'].float()
    if change == 'layout': pointers['in_ptr0'] = torch.ones(3, 2, dtype=torch.bfloat16).t()
    if change == 'size': pointers['in_ptr1'] = torch.ones(4, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        from_runtime_pointers(pointers, rows=2, width=3, epsilon=1e-6)


def test_wrong_output_type_rejected():
    with pytest.raises(ValueError):
        select_output(inputs(), torch.empty(2, dtype=torch.bfloat16),
                      formal_pointer='in_out_ptr1', rows=2, width=3, epsilon=1e-6)


def full_inputs():
    return dict(inputs(), in_out_ptr1=torch.empty(2),
                out_ptr0=torch.empty(6, dtype=torch.bfloat16))


def test_snapshot_survives_in_place_write():
    pointers = full_inputs()
    saved = snapshot_pre_call(pointers, rows=2, width=3)
    pointers['in_out_ptr0'].zero_()
    assert torch.all(saved['in_out_ptr0'] == .25)
    assert set(saved) == {'in_ptr0', 'in_out_ptr0', 'in_ptr1'}


@pytest.mark.parametrize('target,source', [('out_ptr0', 'in_out_ptr0'),
                                         ('in_ptr0', 'in_out_ptr0')])
def test_aliases_checked_before_clone(target, source):
    pointers = full_inputs()
    pointers[target] = pointers[source].reshape(-1)
    with pytest.raises(ValueError, match='Shared argument storage'):
        snapshot_pre_call(pointers, rows=2, width=3)


def test_disjoint_views_of_same_storage_conservatively_rejected():
    pointers = full_inputs()
    storage = torch.empty(12, dtype=torch.bfloat16)
    pointers['in_ptr0'], pointers['out_ptr0'] = storage[:6], storage[6:]
    with pytest.raises(ValueError, match='Shared argument storage'):
        snapshot_pre_call(pointers, rows=2, width=3)


def test_unknown_pointer_rejected():
    pointers = full_inputs()
    pointers['extra'] = torch.empty(1)
    with pytest.raises(ValueError, match='Pointer ABI'):
        snapshot_pre_call(pointers, rows=2, width=3)
