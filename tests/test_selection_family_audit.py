import pytest
from scripts.audit_selection_family import audit


def test_selection_uses_actual_overload_and_versioned_tensor():
    selected=dict(invocation_id='forward:1',overload='aten.topk.default',phase='FORWARD',
        module_context=['router'],dispatcher_schema='aten::topk',argument_bindings=[],
        input_tensors=[],output_tensors=[dict(tensor_id=7,source_ordinal=1)])
    consumer=dict(invocation_id='forward:2',overload='aten.add.Tensor',
        input_tensors=[dict(tensor_id=7,source_ordinal=1)])
    reused=dict(invocation_id='forward:3',overload='aten.add.Tensor',
        input_tensors=[dict(tensor_id=7,source_ordinal=2)])
    data=dict(schema='kernel-analyzer-full-architecture-invocation-inventory-v1',
              trace={'events':[selected,consumer,reused]},model={},implementation='eager')
    result=audit(data)
    assert result['records'][0]['output_consumers']==[['forward:2']]
    assert result['new_bias_confirmed'] is False
    assert result['records'][0]['three_stage_measurement'] is False
    data['trace']['events'].append(consumer)
    with pytest.raises(ValueError,match='Duplicate'):audit(data)
