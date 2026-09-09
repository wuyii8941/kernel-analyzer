import pytest
from scripts.bind_convolution_parameters import bind


def test_closure_not_mislabelled_as_replaced_output():
    pairs=dict(records=[dict(convolution_task_id='forward:3:output_0',
        closed_task_id='forward:4:in_out_ptr0',exact_aot_endpoint_id='forward:graph0:add')])
    mapping=dict(rows=[dict(endpoint='forward:graph0:add',parameters=[
        dict(name='weight',aot_distance=2),dict(name='other',aot_distance=3)])])
    result=bind(pairs,mapping);case=result['cases'][0]
    assert case['carrier']=='weight'
    assert case['exact_aot_endpoint_id'] is None
    assert case['replacement_boundary']=='EXTERNAL_CONVOLUTION_OUTPUT_BEFORE_SEPARATE_BIAS'
    assert not result['runtime_measurement_complete']
    assert len(bind(pairs,dict(rows=[]))['unresolved'])==1
    with pytest.raises(ValueError):bind(dict(records=pairs['records']*2),mapping)
