import pytest
from scripts.summarize_grid_parameter_mappings import validate


def inputs():
    case=dict(task_id='t',carrier='w',mapping_evidence=dict(name='w'),
              expected_symbol='kernel',exact_aot_endpoint_id='backward:node')
    task=dict(task_id='t',symbol='kernel',exact_aot_endpoint_id='backward:node',
              status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT')
    return dict(cases=[case],unresolved_backward_outputs=[],exact_backward_endpoints=1),dict(rows=[task])


def test_consistency_does_not_claim_runtime_validation():
    report=validate(*inputs())
    assert report['mapped_backward_positions']==1
    assert report['status'].endswith('NOT_RUNTIME_VERIFIED')


def test_wrong_endpoint_rejected():
    mapping,tasks=inputs(); mapping['cases'][0]['exact_aot_endpoint_id']='other'
    with pytest.raises(ValueError,match='disagrees'): validate(mapping,tasks)


def test_omitted_unresolved_case_rejected():
    mapping,tasks=inputs(); mapping['exact_backward_endpoints']=2
    with pytest.raises(ValueError,match='denominator'): validate(mapping,tasks)
