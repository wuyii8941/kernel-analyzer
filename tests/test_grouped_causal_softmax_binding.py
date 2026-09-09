from scripts.bind_grouped_causal_softmax_forward import bind


def test_reuses_parameter_selection_not_numerical_results():
    task = dict(task_id='forward:0:out_ptr2', symbol='softmax',
                exact_aot_endpoint_id='forward:softmax', formal_pointer='out_ptr2',
                status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT',
                implementation_kind='TRITON')
    contracts = {'softmax': dict(output_pointers=['out_ptr2'])}
    mapping = dict(rows=[dict(endpoint='forward:softmax', parameters=[
        dict(name='z', aot_distance=2), dict(name='a', aot_distance=2)])])
    result = bind([task], contracts, mapping)
    case = result['cases'][0]
    assert case['carrier'] == 'a'
    assert case['reference_method'] == 'GROUPED_CAUSAL_SOFTMAX_FORWARD_COMMON_INPUT'
    assert case['case_id'].endswith('-grouped-causal-softmax-forward')
    assert not result['runtime_measurement_complete']
    task['bias'] = 999
    assert bind([task], contracts, mapping) == result
    assert bind([task], contracts, {'rows': []})['unresolved']
