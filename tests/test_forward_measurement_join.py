import hashlib
import pytest
from scripts.join_coverage_measurements import join, validate_retained_measurements


def test_record_check_is_not_upgraded(tmp_path):
    raw = tmp_path / 'raw.json'
    raw.write_text('{}')
    status = 'RECORDED_MEASUREMENT_CHECKED'
    inventory = {'records': [{'release': str(tmp_path), 'task_id': 'forward:1:x',
                              'runtime_measurement_status': 'NOT_CAPTURED'}]}
    evidence = {'task_id': 'forward:1:x', 'status': status,
                'raw_artifact': str(raw),
                'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}
    result = join(inventory, tmp_path, {'records': [evidence], 'errors': []})
    assert result['runtime_counts'] == {status: 1}
    assert validate_retained_measurements(result) == 1
    raw.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'):
        validate_retained_measurements(result)
