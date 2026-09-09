import pytest
import torch
from kernel_analyzer.continued_recurrence_reference import from_runtime_pointers
from kernel_analyzer.continued_recurrence_reference import reference, dependency_hashes


def pointers():
    return {'in_ptr0': torch.full((2, 3), 2.),
            'in_ptr1': torch.zeros(2, 3),
            'in_ptr2': torch.tensor([[0., 0.], [1., 2.], [3., 4.], [5., 6.]], dtype=torch.bfloat16),
            'in_ptr3': torch.zeros(2, dtype=torch.bfloat16),
            'in_ptr4': torch.ones(2, dtype=torch.bfloat16),
            'in_ptr5': torch.ones(3, dtype=torch.bfloat16),
            'in_ptr6': torch.full((2,), 2., dtype=torch.bfloat16),
            'in_ptr7': torch.ones(3, dtype=torch.bfloat16)}


def run(p, output=1):
    return from_runtime_pointers(p, channels=2, width=3, outputs=2,
                                 output_index=output, time_start=3, sequence_length=4)


def test_previous_state_and_descending_time():
    p = pointers()
    h = p['in_ptr0'].clone()
    for j, row in enumerate((3, 2)):
        # exp(-softplus(x)) = sigmoid(-x); independent reference expression.
        h = torch.sigmoid(-p['in_ptr2'][row].float())[:, None]*h + (j+1)
        torch.testing.assert_close(run(p, j), h)


@pytest.mark.parametrize('key', ['in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr7'])
def test_storage_dtype_enforced(key):
    p = pointers()
    p[key] = p[key].double()
    with pytest.raises(ValueError):
        run(p)


def test_missing_input_rejected():
    p = pointers()
    del p['in_ptr6']
    with pytest.raises(ValueError):
        run(p)


def test_nonfinite_state_rejected():
    p = pointers()
    p['in_ptr0'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        run(p)


def boundary():
    contract = dict(symbol='kernel', channels=2, state_width=3, outputs=2,
                    time_start=3, sequence_length=4, output_pointer='out_ptr1',
                    output_pointers=['out_ptr0', 'out_ptr1'],
                    segment_input_kind='FP32_PREVIOUS_STATE',
                    reference_dependencies_sha256=dependency_hashes())
    metadata = dict(symbol='kernel', formal_pointer='out_ptr1',
                    input_output_storage_aliases=[], runtime_pointers=pointers())
    return metadata, contract


def test_output_selection():
    metadata, contract = boundary()
    torch.testing.assert_close(reference(metadata, torch.zeros(1, 2, 3), contract),
                               run(metadata['runtime_pointers']).reshape(1, 2, 3))


@pytest.mark.parametrize('key,value', [('symbol', 'different'),
    ('formal_pointer', 'out_ptr0'), ('input_output_storage_aliases', ['in_ptr0'])])
def test_wrong_boundary_rejected(key, value):
    metadata, contract = boundary()
    metadata[key] = value
    with pytest.raises(ValueError):
        reference(metadata, torch.zeros(2, 3), contract)


def test_changed_dependency_rejected():
    metadata, contract = boundary()
    contract['reference_dependencies_sha256'] = {}
    with pytest.raises(ValueError):
        reference(metadata, torch.zeros(2, 3), contract)


def test_wrong_output_dtype():
    metadata, contract = boundary()
    with pytest.raises(ValueError):
        reference(metadata, torch.zeros(2, 3, dtype=torch.bfloat16), contract)
