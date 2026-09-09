import json
import hashlib
import pytest
from scripts.finalize_selected_nll_capture import verify
from kernel_analyzer.explicit_output_capture_audit import CONTRACTS


def test_failed_verification_does_not_leave_global_contract(tmp_path):
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/family_execution_protocol.json').write_text(json.dumps(dict(
        schema='selected-nll-capture-v1', source_sha256={})))
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
    assert 'selected-nll-capture-v1' not in CONTRACTS


def test_complete_record_verified_without_inventing_equivalence(tmp_path):
    raw_dir = tmp_path/'raw'
    raw_dir.mkdir()
    case = dict(case_id='nll', task_id='backward:1:in_out_ptr0', carrier='weight')
    plan, bank = tmp_path/'plan.json', tmp_path/'bank.json'
    plan.write_text(json.dumps(dict(cases=[case])))
    bank.write_text(json.dumps(dict(states=[dict(sequence_id='a'), dict(sequence_id='b')])))
    protocol = dict(schema='selected-nll-capture-v1', claim_scope='FIXED_SUITE_UPDATE',
                    capture_arguments=['--case-plan', str(plan), '--input-bank', str(bank), '--states', '2'],
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (plan, bank)})
    (raw_dir/'family_execution_protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'source_snapshot.json').write_text('{}')
    raw = dict(case_id='nll', carrier='weight', runtime_boundary=dict(task_id=case['task_id']),
               status='COMPLETE', state_ids=['a', 'b'], determinism=dict(all_exact=True),
               parameter_write_protocol=dict(version='adamw-readback-v2'),
               reference_comparison_scope=dict(comparison='INTERNAL_NLL_BACKWARD_OUTPUT_REPLACEMENT',
                   reference_variant='SAVED_INPUT_NLL_FP64_BF16_WRITE', same_local_operands=True,
                   includes_possible_upstream_differences=False, reference_is_absolute_truth=False),
               original_coordinate_statistics={s:[dict(effect_energy=0., repair_energy=1.,
                   effect_repair_inner_product=0.) for _ in range(2)]
                   for s in ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE')})
    (raw_dir/'nll.json').write_text(json.dumps(raw))
    report = verify(tmp_path)
    assert report['recorded_measurement_complete']
    assert report['records'][0]['equivalence_decision'] == 'NOT_ASSESSED'
    assert report['records'][0]['descriptive_effects']['stages']['PARAMETER_WRITE'][
        'complete_suite']['total_rms_ratio'] == 0.0
    raw['state_ids'] = ['a', 'a']
    (raw_dir/'nll.json').write_text(json.dumps(raw))
    assert not verify(tmp_path)['recorded_measurement_complete']


def test_softcapped_nll_uses_its_distinct_reference_contract(tmp_path):
    raw_dir = tmp_path/'raw'
    raw_dir.mkdir()
    case = dict(case_id='softcap', task_id='backward:2:in_out_ptr0', carrier='weight')
    plan, bank = tmp_path/'plan.json', tmp_path/'bank.json'
    plan.write_text(json.dumps(dict(cases=[case])))
    bank.write_text(json.dumps(dict(states=[dict(state_id='a'), dict(state_id='b')])))
    protocol = dict(schema='softcapped-nll-capture-v1', claim_scope='FIXED_SUITE_UPDATE',
                    capture_arguments=['--case-plan', str(plan), '--input-bank', str(bank), '--states', '2'],
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (plan, bank)})
    (raw_dir/'family_execution_protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'source_snapshot.json').write_text('{}')
    record = dict(case_id='softcap', carrier='weight', runtime_boundary=dict(task_id=case['task_id']),
                  status='COMPLETE', state_ids=['a', 'b'], determinism=dict(all_exact=True),
                  parameter_write_protocol=dict(version='adamw-readback-v2'),
                  reference_comparison_scope=dict(
                      comparison='INTERNAL_SOFTCAPPED_NLL_OUTPUT_REPLACEMENT',
                      reference_variant='SOFTCAPPED_NLL_FP64_BF16_WRITE',
                      same_local_operands=True, includes_possible_upstream_differences=False,
                      reference_is_absolute_truth=False),
                  original_coordinate_statistics={stage:[dict(
                      effect_energy=0., repair_energy=1., effect_repair_inner_product=0.)
                      for _ in range(2)] for stage in (
                          'LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE')})
    (raw_dir/'softcap.json').write_text(json.dumps(record))
    report = verify(tmp_path)
    assert report['recorded_measurement_complete']
    assert report['records'][0]['equivalence_decision'] == 'NOT_ASSESSED'
