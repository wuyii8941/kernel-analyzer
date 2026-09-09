import pytest
from kernel_analyzer.numerical_campaign import plan_release


def task(kind='TRITON'):
    return {'rows': [{'task_id': 'forward:1:out', 'implementation_kind': kind,
                      'exact_aot_endpoint_id': 'node',
                      'status': 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'}],
            'reference_cut_tasks': [{'task_id': 'same-dtype:node'}]}


def case():
    return {'task_id': 'forward:1:out', 'case_id': 'example',
            'carrier': 'model.weight', 'reference_method': 'AOT_REPLAY'}


@pytest.mark.parametrize('kind', ['TRITON', 'EXTERNAL', 'ATEN', 'UNKNOWN'])
def test_backend_does_not_decide_eligibility(kind):
    assert plan_release(task(kind), [case()])['counts'] == {'READY_FOR_CAPTURE': 1}


def test_missing_mapping_remains_in_denominator():
    result = plan_release(task(), [])
    assert result['endpoint_count'] == 1
    assert result['counts'] == {'PARAMETER_MAPPING_UNAVAILABLE': 1}


def test_missing_reference_not_negative():
    data = task(); data['reference_cut_tasks'] = []
    assert plan_release(data, [case()])['counts'] == {'REFERENCE_UNAVAILABLE': 1}


def test_ambiguous_comparison_not_silently_selected():
    assert plan_release(task(), [case(), case()])['counts'] == {'AMBIGUOUS_DECLARED_COMPARISON': 1}


def test_duplicates_rejected():
    data = task(); data['rows'] *= 2
    with pytest.raises(ValueError):
        plan_release(data, [case()])


def test_path_escape_rejected():
    row = case(); row['case_id'] = '../escape'
    with pytest.raises(ValueError):
        plan_release(task(), [row])


def test_unresolved_boundary_cannot_be_rescued_by_a_declared_case():
    data = task(); data['rows'][0]['status'] = 'UNKNOWN'
    assert plan_release(data, [case()])['counts'] == {'UNRESOLVED_EXECUTION_BOUNDARY': 1}


def test_custom_reference_is_not_silently_replaced():
    row = case(); row['reference_method'] = 'EXTERNAL_FP32_RECOMPUTE'
    assert plan_release(task(), [row])['counts'] == {'REFERENCE_ADAPTER_REQUIRED': 1}


def test_unmatched_declarations_are_reported():
    row = case(); row['task_id'] = 'absent'
    result = plan_release(task(), [row])
    assert result['declared_tasks_absent_from_release'] == ['absent']
    assert result['endpoint_count'] == 1


def test_aot_reference_does_not_claim_common_operands():
    scope = plan_release(task(), [case()])['rows'][0]['reference_scope']
    assert scope['same_local_operands'] is False
    assert scope['includes_possible_upstream_differences'] is True


@pytest.mark.parametrize('symbol', ['mm', 'bmm', 'addmm'])
def test_external_reference_does_not_require_aot_cut(symbol):
    data = task('EXTERN'); data['reference_cut_tasks'] = []
    data['rows'][0]['symbol'] = 'extern_kernels.' + symbol
    row = case(); row['reference_method'] = 'EXTERNAL_FP32_RECOMPUTE'
    result = plan_release(data, [row])
    assert result['counts'] == {'READY_FOR_CAPTURE': 1}
    assert result['rows'][0]['reference_scope']['same_local_operands'] is True
