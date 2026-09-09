from pathlib import Path
import pytest
from scripts import join_recurrence_bindings as module


def inputs(monkeypatch):
    monkeypatch.setattr(module,'sha',lambda p:'digest')
    task=dict(task_id='t',symbol='k',exact_aot_endpoint_id='e',
              status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT')
    monkeypatch.setattr(module,'read',lambda p:dict(rows=[task]))
    case=dict(task_id='t',case_id='c',carrier='w',expected_symbol='k',exact_aot_endpoint_id='e',
              reference_output_pointer='out_ptr0',reference_output_index=0,
              reference_method='CONTINUED_RECURRENCE_COMMON_INPUT')
    plan=dict(schema='continued-recurrence-bound-plan-v1',cases=[case],tasks_sha256='digest',
              source_sha256={'source':'digest'},contract=dict(symbol='k',
              output_pointers=['out_ptr0'],function_ast_sha256='function'))
    row=dict(release='/data1/tzh/release',task_id='t',symbol='k',formal_pointer='out_ptr0',
             implementation_kind='TRITON',carrier=None,reference_candidates=[],
             eligibility='NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT',runtime_measurement_status='NOT_CAPTURED')
    return dict(records=[row]),plan


def test_static_binding_does_not_promote_measurement(monkeypatch):
    inventory,plan=inputs(monkeypatch)
    result=module.join(inventory,Path('/data1/tzh/release'),plan,Path('/data1/tzh/plan.json'))
    row=result['records'][0]
    assert row['eligibility']=='REFERENCE_AND_STATIC_PARAMETER_BINDING_PRESENT'
    assert row['runtime_measurement_status']=='NOT_CAPTURED'
    assert inventory['records'][0]['carrier'] is None


def test_other_release_rejected(monkeypatch):
    inventory,plan=inputs(monkeypatch)
    plan['tasks_sha256']='other'
    with pytest.raises(ValueError):
        module.join(inventory,Path('/data1/tzh/release'),plan,Path('/data1/tzh/plan.json'))


def test_conflicting_inventory_carrier_rejected(monkeypatch):
    inventory,plan=inputs(monkeypatch)
    inventory['records'][0]['carrier']='different'
    with pytest.raises(ValueError):
        module.join(inventory,Path('/data1/tzh/release'),plan,Path('/data1/tzh/plan.json'))


def test_forward_binding_keeps_existing_record_evidence(monkeypatch):
    import scripts.run_residual_rms_forward_capture as capture
    inventory, plan = inputs(monkeypatch)
    plan['schema'] = 'residual-rms-forward-bound-plan-v1'
    plan['contracts'] = {'k': plan.pop('contract')}
    plan['cases'][0]['reference_method'] = 'RESIDUAL_RMS_FORWARD_COMMON_INPUT'
    selected = []
    monkeypatch.setattr(capture, 'select', lambda p,c: selected.append(c))
    inventory['records'][0]['runtime_measurement_status'] = 'RECORDED_MEASUREMENT_CHECKED'
    inventory['records'][0]['measurement_evidence'] = {'raw_sha256':'retained'}
    result = module.join(inventory, Path('/data1/tzh/release'), plan, Path('/data1/tzh/plan.json'))
    row = result['records'][0]
    assert selected
    assert row['reference_candidates'][0]['family'] == 'RESIDUAL_RMS_FORWARD'
    assert row['measurement_evidence'] == {'raw_sha256':'retained'}
    assert row['runtime_measurement_status'] == 'RECORDED_MEASUREMENT_CHECKED'
