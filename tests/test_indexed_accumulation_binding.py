import pytest
from scripts.bind_indexed_accumulation_cases import bind


def fixture():
    task=dict(task_id='backward:direct_aten:0:mutated_output_0',symbol='index_put_',
              implementation_kind='DIRECT_ATEN',formal_pointer='mutated_output_0',
              status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT',
              exact_aot_endpoint_id='backward:graph0:index_put')
    case=dict(task_id=task['task_id'],case_id='embedding',expected_symbol='index_put_',
              exact_aot_endpoint_id=task['exact_aot_endpoint_id'],carrier='embedding.weight')
    return task,case


def test_exact_mapping_selects_without_claiming_runtime_completion():
    task,case=fixture()
    result=bind([task],[case])
    assert len(result['cases'])==1 and not result['training_support_established']
    assert result['cases'][0]['reference_method']=='INDEXED_INPUT_ORDER_FP32'
    assert result['cases'][0]['runtime_parameter_reach']=='NOT_YET_MEASURED'


def test_internal_or_missing_mapping_kept_unresolved():
    task,case=fixture()
    assert len(bind([task],[])['unresolved'])==1
    internal=dict(task,status='INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT',exact_aot_endpoint_id=None)
    assert len(bind([internal],[])['unresolved'])==1
    with pytest.raises(ValueError): bind([internal],[case])


def test_wrong_semantics_and_pointer_do_not_get_reference():
    task,case=fixture()
    assert bind([dict(task,symbol='copy_')],[])['cases']==[]
    assert len(bind([dict(task,formal_pointer='output_0')],[case])['unresolved'])==1
    with pytest.raises(ValueError): bind([task],[dict(case,expected_symbol='copy_')])
