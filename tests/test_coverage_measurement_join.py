import pytest
from scripts.join_coverage_measurements import join, validate_retained_measurements
from scripts.run_numerical_coverage import sha


def inventory():
    return dict(records=[dict(release='/tmp/a', task_id='same', runtime_measurement_status='UNKNOWN'),
                         dict(release='/tmp/b', task_id='same', runtime_measurement_status='UNKNOWN')])


def test_release_identity_and_full_denominator_preserved():
    result = join(inventory(), '/tmp/a', dict(records=[dict(task_id='same', status='VERIFIED')], errors=[]))
    assert result['positions'] == 2
    assert result['runtime_counts'] == {'VERIFIED': 1, 'UNKNOWN': 1}
    assert result['all_kernel_support_established'] is False


def test_pending_not_promoted():
    result = join(inventory(), '/tmp/a', dict(records=[dict(task_id='same', status='NOT_SCHEDULED')], errors=[]))
    assert 'VERIFIED' not in result['runtime_counts']


@pytest.mark.parametrize('tasks', [['missing'], ['same', 'same']])
def test_unknown_and_duplicate_rejected(tasks):
    with pytest.raises(ValueError):
        join(inventory(), '/tmp/a', dict(records=[dict(task_id=t, status='VERIFIED') for t in tasks], errors=[]))


def test_stale_or_missing_retained_evidence_rejected(tmp_path):
    path = tmp_path/'raw.json'
    path.write_text('{}')
    data = inventory()
    data['records'][0].update(runtime_measurement_status='VERIFIED', measurement_evidence=dict(
        status='VERIFIED', task_id='same', raw_artifact=str(path), raw_sha256=sha(path)))
    assert validate_retained_measurements(data) == 1
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'):
        validate_retained_measurements(data)
    del data['records'][0]['measurement_evidence']
    with pytest.raises(ValueError, match='Missing'):
        validate_retained_measurements(data)


def test_old_failures_not_erased_by_next_family():
    data = inventory()
    data['verification_errors'] = [dict(reason='old failure')]
    result = join(data, '/tmp/a', dict(records=[], errors=[]))
    assert result['verification_errors'] == data['verification_errors']


def test_audited_measurement_binding_restores_family_and_carrier():
    result=join(inventory(),'/tmp/a',dict(records=[dict(task_id='same',status='VERIFIED')],errors=[]),
        {'same':dict(carrier='weight',reference_family='GROUPED_CAUSAL_SOFTMAX')})
    row=result['records'][0]
    assert row['carrier']=='weight'
    assert row['reference_candidates']==[dict(
        family='GROUPED_CAUSAL_SOFTMAX',source='AUDITED_BOUND_MEASUREMENT_PLAN')]


def test_missing_or_conflicting_audited_binding_rejected():
    report=dict(records=[dict(task_id='same',status='VERIFIED')],errors=[])
    with pytest.raises(ValueError,match='lacks audited'):
        join(inventory(),'/tmp/a',report,{})
    data=inventory()
    data['records'][0]['carrier']='other'
    with pytest.raises(ValueError,match='carrier differs'):
        join(data,'/tmp/a',report,{'same':dict(carrier='weight',reference_family='X')})
