import pytest
from scripts.bind_internal_reference_cases import bind


def inputs():
    manifest=dict(reference_family='SELECTED_SILU_PRODUCT',sources=[dict(rows=[dict(status='SOURCE_CHECKED',symbol='kernel',
        contract=dict(output_pointer='out_ptr0',function_ast_sha256='digest'))])])
    task=dict(task_id='backward:1:out_ptr0',symbol='kernel',formal_pointer='out_ptr0',
        status='INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT',
        implementation_kind='TRITON',exact_aot_endpoint_id=None,closure_uses_candidate_values=False,
        closed_by_semantic_endpoint_tasks=['a','b'])
    mappings=dict(cases=[dict(task_id='a',carrier='weight_a',mapping_evidence=dict(aot_distance=4)),
                        dict(task_id='b',carrier='weight_b',mapping_evidence=dict(aot_distance=2))])
    return manifest,dict(rows=[task]),mappings


def test_internal_reference_does_not_invent_aot_identity():
    result=bind(*inputs())
    case=result['cases'][0]
    assert case['carrier']=='weight_b'
    assert case['exact_aot_endpoint_id'] is None
    assert case['runtime_parameter_reach']=='NOT_YET_MEASURED'
    assert case['reference_method']=='REGISTERED_SAME_INPUT_REFERENCE'
    assert case['reference_family']=='SELECTED_SILU_PRODUCT'
    assert case['reference_variant']=='FP32_NATIVE'
    assert case['reference_contract']['function_ast_sha256']=='digest'
    assert not result['all_source_positions_supported']


def test_no_downstream_mapping_retained_as_unresolved():
    m,t,p=inputs(); p['cases']=[]
    result=bind(m,t,p)
    assert not result['cases']
    assert result['source_checked_internal_positions']==1
    assert result['unresolved'][0]['reason']=='NO_MAPPED_DOWNSTREAM_PARAMETER'


def test_value_based_closure_not_accepted():
    m,t,p=inputs(); t['rows'][0]['closure_uses_candidate_values']=True
    assert not bind(m,t,p)['cases']


def test_duplicate_task_rejected():
    m,t,p=inputs(); t['rows']*=2
    with pytest.raises(ValueError,match='Duplicate'): bind(m,t,p)


def test_exact_endpoint_uses_direct_parameter_mapping():
    manifest,tasks,mappings=inputs()
    task=tasks['rows'][0]
    task.update(status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT',
                exact_aot_endpoint_id='backward:graph0:sum',
                closed_by_semantic_endpoint_tasks=[])
    mappings['cases']=[dict(task_id=task['task_id'],carrier='direct_weight',
                            mapping_evidence=dict(aot_distance=0))]
    result=bind(manifest,tasks,mappings)
    assert not result['unresolved']
    assert result['schema']=='source-checked-reference-plan-v2'
    assert result['cases'][0]['carrier']=='direct_weight'
    assert result['cases'][0]['parameter_selection_rule']=='EXACT_AOT_ENDPOINT_DIRECT_PARAMETER_MAPPING'


def test_exact_endpoint_without_direct_mapping_is_unresolved():
    manifest,tasks,mappings=inputs()
    tasks['rows'][0]['status']='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'
    mappings['cases']=[]
    result=bind(manifest,tasks,mappings)
    assert not result['cases']
    assert result['unresolved'][0]['reason']=='NO_DIRECT_PARAMETER_MAPPING'


def test_generic_forward_path_selects_shortest_then_named_parameter():
    manifest,tasks,_mappings=inputs()
    task=tasks['rows'][0]
    task.update(status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT',
                exact_aot_endpoint_id='forward:graph0:scale',
                closed_by_semantic_endpoint_tasks=[])
    mappings={
        'schema':'forward-parameter-paths-v1',
        'rows':[dict(endpoint='forward:graph0:scale',parameters=[
            dict(name='z_weight',aot_distance=3),
            dict(name='b_weight',aot_distance=2),
            dict(name='a_weight',aot_distance=2),
        ])],
    }
    result=bind(manifest,tasks,mappings)
    case=result['cases'][0]
    assert case['carrier']=='a_weight'
    assert case['parameter_selection_evidence']['mapping_evidence'][
        'selection_rule'] == 'SHORTEST_AOT_DISTANCE_THEN_PARAMETER_NAME'
