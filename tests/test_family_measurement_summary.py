import json
import pytest
from scripts.summarize_family_measurements import summarize


def record(rms=0.0):
    return dict(status='VERIFIED',case_id='case',raw_artifact='raw.json',raw_sha256='hash',
        analysis=dict(claim_scope='FIXED_SUITE_UPDATE',bias_analysis=dict(
            fixed_suite_total_rms=rms,fixed_suite_aligned_ratio_of_sums=0.,confirmation_state_ids=['16','17'])))


def test_missing_positions_remain_and_magnitude_does_not_promote_training():
    report=dict(records=[record(0.2),dict(status='NOT_CAPTURED',case_id='pending')])
    result=summarize(report)
    assert len(result['audit']['records'])==2
    assert result['verified_positions']==1
    assert result['maximum_confirmation_update_rms']==0.2
    assert result['strong_training_result_established'] is False
    assert result['automatic_long_training_selection'].startswith('NOT_PERFORMED')


@pytest.mark.parametrize('rms',[float('nan'),float('inf'),-0.01])
def test_invalid_magnitudes_rejected(rms):
    with pytest.raises(ValueError): summarize(dict(records=[record(rms)]))


def test_no_observations_is_unknown_not_zero():
    result=summarize(dict(records=[dict(status='NOT_CAPTURED')]))
    assert result['maximum_confirmation_update_rms'] is None
    assert result['verified_positions']==0


def test_verified_capture_without_margin_keeps_descriptive_magnitude(tmp_path):
    raw=tmp_path/'raw.json'
    rows=[dict(effect_energy=x,repair_energy=1.,effect_repair_inner_product=0.)
          for x in (1.,1.,4.,4.)]
    raw.write_text(json.dumps(dict(
        state_ids=['0','1','2','3'],confirmation_state_ids=['2','3'],carrier='weight',
        original_coordinate_statistics={stage:rows for stage in (
            'LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')})))
    report=dict(records=[dict(status='VERIFIED',case_id='case',analysis=None,
        raw_artifact=str(raw),raw_sha256='recorded')])
    result=summarize(report)
    assert result['verified_positions']==1
    assert result['maximum_confirmation_update_rms']==2.
