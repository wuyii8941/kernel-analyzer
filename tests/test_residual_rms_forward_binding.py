import pytest
from scripts.bind_residual_rms_forward import bind


def fixture():
    task = dict(task_id='forward:1:out_ptr0', symbol='triton_x',
                formal_pointer='out_ptr0', exact_aot_endpoint_id='forward:graph0:saved',
                implementation_kind='TRITON', status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT')
    contracts = {'triton_x': dict(output_pointers=['out_ptr0'])}
    mapping = dict(rows=[dict(endpoint='forward:graph0:saved', parameters=[
        dict(name='weight', aot_distance=2, backward_entry='backward:graph0:b')])])
    return task, contracts, mapping


def test_binding_preserves_scope_and_no_runtime_claim():
    task, contracts, mapping = fixture()
    result = bind([task], contracts, mapping)
    assert result['cases'][0]['carrier'] == 'weight'
    assert result['cases'][0]['runtime_parameter_reach'] == 'NOT_YET_MEASURED'
    assert not result['runtime_measurement_complete']


@pytest.mark.parametrize('field,value', [('formal_pointer', 'unknown'),
    ('status', 'UNRESOLVED'), ('implementation_kind', 'OTHER'),
    ('exact_aot_endpoint_id', 'forward:graph0:other')])
def test_invalid_bindings_retained(field, value):
    task, contracts, mapping = fixture()
    task[field] = value
    result = bind([task], contracts, mapping)
    assert not result['cases']
    assert len(result['unresolved']) == 1


def test_duplicate_identity_rejected():
    task, contracts, mapping = fixture()
    with pytest.raises(ValueError): bind([task, task], contracts, mapping)
