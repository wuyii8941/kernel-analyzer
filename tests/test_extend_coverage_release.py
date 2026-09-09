import pytest
from scripts.extend_coverage_release import extend


def test_same_task_id_is_distinct_across_releases(tmp_path):
    old = {'records': [{'release': str(tmp_path/'r1'), 'task_id': 'x',
                        'runtime_measurement_status': 'VERIFIED'}]}
    result = extend(old, tmp_path/'r3', [{'task_id':'x','implementation_kind':'TRITON'}])
    assert result['records'][0] == old['records'][0]
    assert result['records'][1]['runtime_measurement_status'] == 'NOT_ASSESSED_BY_THIS_INVENTORY'
    assert result['positions'] == 2
    with pytest.raises(ValueError, match='already registered'):
        extend(result, tmp_path/'r3', [{'task_id':'x'}])


def test_duplicate_source_tasks_rejected(tmp_path):
    with pytest.raises(ValueError, match='duplicate'):
        extend({'records':[]}, tmp_path, [{'task_id':'x'}, {'task_id':'x'}])
