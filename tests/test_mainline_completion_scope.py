import json
import hashlib
import pytest
from scripts import summarize_training_numerical_v2 as audit


@pytest.mark.parametrize('direction_present', [False, True])
def test_execution_completion_does_not_require_positive_result(tmp_path, monkeypatch, direction_present):
    monkeypatch.setattr(audit, 'BASE', tmp_path)
    monkeypatch.setattr(audit, 'CASES', {})
    monkeypatch.setattr(audit, 'RECOVERY_CASES', {})

    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    raw_hash = save('raw.json', {})
    report_hash = save('recomputed.json', {'measurement_status': 'VALID', 'case_id': 'x'})
    save('report_recomputation_verification.json', {'status': 'VERIFIED_RECOMPUTATION',
         'reports': [{'raw': 'raw.json', 'report': 'recomputed.json', 'raw_sha256': raw_hash, 'report_sha256': report_hash}]})
    save('synthetic_validation.json', {'status': 'PASS_FIXED_SUITE_PATH'})
    save('language_training_confirmation_iid/summary.json', {'status': 'COMPLETE_FROZEN_INITIALIZATION_CONFIRMATION', 'primary_prediction_confirmed': False})
    save('language_training_confirmation_iid/execution_verification.json', {'status': 'VERIFIED_RECORDED_EXECUTION'})
    save('language_checkpoint_direct_probe/summary.json', {'status': 'COMPLETE_SAME_TRAINING_CHECKPOINT_MECHANISM_FOLLOWUP'})
    save('language_accumulation_identity_retry/result.json', {'status': 'IDENTITY_VERIFIED'})
    save('granite_expert_order_confirmation_v2/recomputed.json', {'measurement_status': 'VALID', 'case_id': 'new',
         'provenance': {'data_use': 'NEW_MODEL_AND_MOE_COMBINATION_IMPLEMENTATION_FIXED_SUITE_CONFIRMATION'}})
    d = {'all_later_windows_same_halfspace': direction_present, 'each_window_split_half_same_halfspace': direction_present}
    save('language_temporal_extension/summary.json', {'status': 'COMPLETE_VERIFIED_TEMPORAL_EXTENSION',
         'rows': [{'pair': 0, 'trajectory_diagnostics': {'candidate': d, 'reference': d},
                   'loss_gaps': [{'candidate_minus_reference_loss': 1.}]}]})
    result = audit.build_progress()
    assert result['status'] == 'INCOMPLETE_FULL_RESEARCH_PLAN'
    assert result['selected_experiment_execution_complete'] is True
    assert result['observed_window_direction_and_loss_chain_supported'] is direction_present
    assert result['training_confirmation_primary_prediction_confirmed'] is False
    assert result['optimizer_quantization_mechanism_confirmed'] is False
    assert result['optimizer_training_material_effect_confirmed'] is False
    assert result['optimizer_training_modification_confirmed'] is False
    assert result['optimizer_component_prediction_confirmed'] is False
    assert result['optimizer_component_primary_write_reduced'] is None
    assert result['optimizer_hybrid_training_execution_verified'] is False
    assert result['optimizer_hybrid_training_decision'] is None
