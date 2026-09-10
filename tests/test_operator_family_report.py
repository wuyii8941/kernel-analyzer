import pytest
from scripts.summarize_operator_families import summarize
from scripts.summarize_operator_families import ALIASES, FAMILIES
from kernel_analyzer.source_reference_registry import REFERENCES


def row(task, labels, **extra):
    value = dict(release='release', task_id=task,
                 reference_candidates=[dict(family=x) for x in labels],
                 runtime_measurement_status='NOT_CAPTURED')
    value.update(extra)
    return value


def test_merge_does_not_promote_measurement_or_mechanism():
    result = summarize([row('a', ['DECAYED_RECURRENCE']), row('b', ['CONTINUED_RECURRENCE'])], [])
    assert result['families_in_supplied_reference_inventory'] == 1
    recurrence = next(r for r in result['families'] if r['family_id'] == 'RECURRENCE')
    assert recurrence['classified_positions'] == 2
    assert recurrence['recorded_runtime_status_counts'] == {'NOT_CAPTURED': 2}
    assert recurrence['support_stage_counts'] == {'REFERENCE_AVAILABLE_TRAINING_BINDING_REQUIRED': 2}
    assert result['confirmed_root_cause_count'] is None


def test_unknown_ambiguous_and_duplicates():
    result = summarize([row('a', ['NEW']), row('b', ['ROW_SUM', 'RMS_BACKWARD'])], [])
    assert result['unclassified_positions'] == 2
    assert result['unknown_labels'] == {'NEW': 1}
    assert len(result['ambiguous_positions']) == 1
    with pytest.raises(ValueError): summarize([row('a', []), row('a', [])], [])


def test_historical_convolution_does_not_become_new_verified_coverage():
    history=dict(family='CONVOLUTION',sha256='digest',path='historical.json',
                 import_status='HISTORICAL_RECORD_READ_NOT_RERUN')
    result=summarize([],[],[history])
    family=next(r for r in result['families'] if r['family_id']=='CONVOLUTION')
    assert family['classified_positions']==0
    assert family['additional_historical_artifacts']==[history]
    assert not family['historical_artifacts_are_current_protocol_verification']
    assert result['catalogue_family_count']==17
    with pytest.raises(ValueError):summarize([],[],[history,history])


def test_support_stages_do_not_confuse_binding_with_measurement():
    result = summarize([
        row('identified', [], eligibility='NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT'),
        row('referenced', ['ROW_SUM']),
        row('bound', ['ROW_SUM'], carrier='weight'),
        row('measured', ['ROW_SUM'], carrier='weight', runtime_measurement_status='VERIFIED'),
        row('ordinary', [], eligibility='OTHER_IMPLEMENTATION_REQUIRES_REFERENCE_AUDIT'),
    ], [])
    assert result['support_stage_counts'] == {
        'IDENTIFIED_REFERENCE_ADAPTER_REQUIRED': 1,
        'REFERENCE_AVAILABLE_TRAINING_BINDING_REQUIRED': 1,
        'REFERENCE_AND_TRAINING_BINDING_READY_NOT_VALIDLY_MEASURED': 1,
        'VALID_MEASUREMENT_COMPLETED': 1,
        'ORDINARY_IMPLEMENTATION_REFERENCE_AUDIT_REQUIRED': 1,
    }
    assert result['support_counts_are_positions_not_distinct_operator_families'] is True


def test_valid_backend_counts_only_measured_positions():
    result = summarize([
        row('triton', ['ROW_SUM'], runtime_measurement_status='VERIFIED',
            implementation_kind='TRITON'),
        row('external', ['ROW_SUM'], runtime_measurement_status='RECORDED_MEASUREMENT_CHECKED',
            implementation_kind='EXTERN'),
        row('pending', ['ROW_SUM'], implementation_kind='TRITON'),
    ], [])
    family = next(r for r in result['families'] if r['family_id'] == 'REDUCTION')
    assert family['valid_measurement_implementation_kind_counts'] == {
        'TRITON': 1, 'EXTERN': 1,
    }


def test_additional_evidence_does_not_invent_inventory_positions():
    extra=dict(family='SELECTION',artifact_sha256='digest',evidence_kind='FIXED_SUITE',
               measurement_status='VALID_FIXED_SUITE_ENGINEERING_MEASUREMENT')
    result=summarize([],[],additional_evidence=[extra])
    family=next(r for r in result['families'] if r['family_id']=='SELECTION')
    assert family['classified_positions']==0
    assert family['additional_measurement_evidence']==[extra]
    assert result['catalogue_family_count']==17
    with pytest.raises(ValueError,match='Repeated additional evidence'):
        summarize([],[],additional_evidence=[extra,extra])


def test_every_automatic_reference_has_one_reporting_family():
    assert set(REFERENCES) <= set(ALIASES)
    assert all(ALIASES[name] in FAMILIES for name in REFERENCES)
