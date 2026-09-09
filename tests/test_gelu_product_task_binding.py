import pytest
from scripts.bind_gelu_product_tasks import bind


def test_internal_task_remains_internal_and_scope_is_not_graph_proof():
    scan=dict(sources=[dict(rows=[dict(status='SOURCE_CHECKED',symbol='gelu',
        contract=dict(output_pointer='out_ptr0'))])])
    task=dict(task_id='backward:1:out_ptr0',symbol='gelu',formal_pointer='out_ptr0',
        phase='BACKWARD',implementation_kind='TRITON',exact_aot_endpoint_id=None,
        status='INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT')
    scope=dict(trainable_parameters=['projection.weight'])
    result=bind(scan,dict(rows=[task]),scope)
    assert result['cases'][0]['exact_aot_endpoint_id'] is None
    assert result['cases'][0]['carrier']=='projection.weight'
    assert result['cases'][0]['runtime_parameter_reach']=='NOT_YET_MEASURED'
    assert not result['runtime_measurement_complete']
    with pytest.raises(ValueError):bind(scan,dict(rows=[task,task]),scope)
    with pytest.raises(ValueError):bind(scan,dict(rows=[task]),dict(trainable_parameters=[]))
