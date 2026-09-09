import json
from kernel_analyzer.reference_provenance import recorded_reference_scope


def setup(tmp_path, method='AOT_REPLAY'):
    path = tmp_path / 'run/raw/case.json'; path.parent.mkdir(parents=True)
    payload = {'case_id': 'case', 'carrier': 'weight', 'runtime_boundary': {'task_id': 'backward:1:out'}}
    path.write_text(json.dumps(payload))
    plan = {'cases': [{'case_id': 'case', 'carrier': 'weight', 'task_id': 'backward:1:out', 'reference_method': method}]}
    (tmp_path / 'run/case_plan.json').write_text(json.dumps(plan))
    return path, payload


def test_aot_scope_is_not_single_kernel_bias(tmp_path):
    path, payload = setup(tmp_path)
    result = recorded_reference_scope(payload, path, tmp_path)
    assert result['scope']['same_local_operands'] is False
    assert not result['single_kernel_bias_proved']


def test_matching_name_does_not_override_wrong_runtime_boundary(tmp_path):
    path, payload = setup(tmp_path)
    payload['runtime_boundary']['task_id'] = 'other'
    assert recorded_reference_scope(payload, path, tmp_path)['status'] == 'PLAN_ARTIFACT_SCOPE_MISMATCH'


def test_special_compatibility_slot_requires_review(tmp_path):
    path, payload = setup(tmp_path, 'PARTIAL_REDUCTION_FROM_BOUND_INPUT')
    result = recorded_reference_scope(payload, path, tmp_path)
    assert result['scope']['comparison'] == 'SPECIAL_REFERENCE_REQUIRES_ADAPTER_REVIEW'


def test_explicit_scope_is_recorded_not_independently_proved(tmp_path):
    path, payload = setup(tmp_path)
    payload['reference_comparison_scope'] = {'comparison': 'CUSTOM'}
    path.write_text(json.dumps(payload))
    result = recorded_reference_scope(payload, path, tmp_path)
    assert result['status'] == 'EXPLICIT_ARTIFACT_SCOPE'
    assert not result['single_kernel_bias_proved']


def test_successful_command_links_external_plan_but_failed_command_does_not(tmp_path):
    path, payload = setup(tmp_path)
    plan = tmp_path / 'elsewhere/case_plan.json'; plan.parent.mkdir()
    (tmp_path / 'run/case_plan.json').rename(plan)
    execution = tmp_path / 'run/execution/case.json'; execution.parent.mkdir()
    record = {'returncode': 0, 'command': ['capture', '--case-plan', str(plan),
                                          '--training-bias-profile-v2-output-dir', str(path.parent)]}
    execution.write_text(json.dumps(record))
    result = recorded_reference_scope(payload, path, tmp_path)
    assert result['status'] == 'COMMAND_LINKED_CURRENT_PLAN_DECLARATION'
    assert not result['execution_record']['plan_digest_frozen_by_execution_record']
    record['returncode'] = 1; execution.write_text(json.dumps(record))
    assert recorded_reference_scope(payload, path, tmp_path)['status'] == 'UNKNOWN_REFERENCE_SCOPE'
