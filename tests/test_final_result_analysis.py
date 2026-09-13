"""Completion tracks verified work, never a desired positive result."""
from scripts import build_final_result_analysis as report


def prepare(monkeypatch, training_status='VERIFIED'):
    monkeypatch.setattr(report, 'verify_measurements', lambda: {
        'status': 'RECORDS_RECOMPUTED', 'counts': {}})
    monkeypatch.setattr(report, 'verify_training', lambda _: {
        'status': training_status, 'historical_on_reproduced': True,
        'primary': {'decision': 'NOT_CONFIRMED'}})
    records = {
        'baseline_verification.json': {
            'status': 'VERIFIED_RECORDS', 'errors': [],
            'unique_verified_captures': 0, 'local_results': {},
            'maximum_write_rms_among_allclose_passed': None, 'policies': []},
        'bounded_validation_actual_readback.json': {'status': 'PASS'},
        'exceedance_validation_replay.json': {'status': 'PASS'},
        'test_run.json': {'exit_code': 0},
        'final_additional_test_run.json': {'exit_code': 0},
        'compensation_control_scalar_followup/result.json': {
            'tensor_off_matches_eager_default_all_steps': True},
    }
    monkeypatch.setattr(report, 'read', records.__getitem__)
    # Source hashing is independent of this decision-table test.
    monkeypatch.setattr(report.Path, 'read_bytes', lambda _: b'test fixture')


def test_negative_result_can_complete_analysis(monkeypatch):
    prepare(monkeypatch)
    result = report.build()
    assert result['status'] == 'COMPLETED_ANALYSIS_UNDER_DECLARED_SCOPE'
    assert result['conclusions']['compensation_same_path_training'] == 'NOT_CONFIRMED'
    assert result['conclusions']['universal_mainline_complete'] is False


def test_missing_training_prevents_completion(monkeypatch):
    prepare(monkeypatch, 'INCOMPLETE')
    assert report.build()['status'] == 'ANALYSIS_NOT_FINISHED'


def test_invalid_training_prevents_completion(monkeypatch):
    prepare(monkeypatch, 'INVALID')
    assert report.build()['status'] == 'ANALYSIS_NOT_FINISHED'


def test_unreproduced_historical_control_needs_investigation(monkeypatch):
    prepare(monkeypatch)
    monkeypatch.setattr(report, 'verify_training', lambda _: {
        'status': 'VERIFIED', 'historical_on_reproduced': False})
    assert report.build()['status'] == 'ANALYSIS_NOT_FINISHED'
