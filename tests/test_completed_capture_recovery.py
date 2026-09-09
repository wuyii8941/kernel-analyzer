import hashlib
import json
import pytest
from scripts.recover_completed_numerical_capture import recover
from scripts.run_numerical_coverage import ROOT, sha
from kernel_analyzer.numerical_campaign import digest


def setup_capture(root):
    case = {'case_id': 'test', 'task_id': 'backward:1:out',
            'carrier': 'weight', 'reference_method': 'AOT_REPLAY'}
    coverage = {'rows': [{'status': 'READY_FOR_CAPTURE', 'task_id': case['task_id'], 'case': case}]}
    protocol = {'schema': 'test', 'primary_stage': 'PARAMETER_WRITE',
                'claim_scope': 'FIXED_SUITE_UPDATE', 'fixed_suite_margins': {'full_update_rms': .01},
                'coverage_sha256': digest(coverage),
                'source_sha256': {str(ROOT / 'src/kernel_analyzer/training_numerical_analysis.py'):
                                  sha(ROOT / 'src/kernel_analyzer/training_numerical_analysis.py')}}
    (root / 'coverage.json').write_text(json.dumps(coverage))
    (root / 'protocol.json').write_text(json.dumps(protocol))
    run = root / 'runs' / hashlib.sha256(case['task_id'].encode()).hexdigest()[:20]
    (run / 'raw').mkdir(parents=True)
    rows = [{'effect_energy': 0., 'repair_energy': 1., 'effect_repair_inner_product': 0.,
             'nonzero_effect_coordinates': 0} for _ in range(32)]
    payload = {'case_id': 'test', 'carrier': 'weight', 'status': 'COMPLETE',
               'contrast_id': 'LOCAL_IMPLEMENTATION_SUBSTITUTION',
               'runtime_boundary': {'task_id': case['task_id']},
               'determinism': {'all_exact': True},
               'parameter_write_protocol': {
                   'version': 'adamw-readback-v2',
                   'measurement': 'parameter_after_step_minus_parameter_before_step',
               },
               'state_ids': list(range(32)), 'calibration_state_ids': list(range(16)),
               'confirmation_state_ids': list(range(16, 32)),
               'original_coordinate_statistics': {stage: rows for stage in
                                                  ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE')}}
    raw_path = run / 'raw/test.json'
    raw_path.write_text(json.dumps(payload))
    return run, raw_path


def test_recovery_is_reanalysis_not_fabricated_process_success(tmp_path):
    run, raw = setup_capture(tmp_path)
    before = raw.read_bytes()
    assert recover(tmp_path)[0]['status'] == 'VALID'
    status = json.loads((run / 'status.json').read_text())
    assert status['returncode'] is None
    assert status['analysis_data_use'] == 'CURRENT_REANALYSIS'
    assert len(status['dependencies_not_frozen_in_original_protocol']) == 3
    assert raw.read_bytes() == before
    assert recover(tmp_path) == []


@pytest.mark.parametrize('field,value', [('case_id', 'other'), ('state_ids', list(range(31))),
                                       ('status', 'PARTIAL'), ('determinism', {'all_exact': False})])
def test_incomplete_capture_cannot_be_recovered(tmp_path, field, value):
    run, raw = setup_capture(tmp_path)
    payload = json.loads(raw.read_text()); payload[field] = value
    raw.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        recover(tmp_path)
    assert not (run / 'status.json').exists()
