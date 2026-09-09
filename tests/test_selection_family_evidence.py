import json

from scripts.build_selection_family_evidence import build


def test_selection_evidence_is_bounded(monkeypatch,tmp_path):
    recomputation=tmp_path/'recomputation.json'
    row=dict(state_id=0,status='MEASURED_ENGINEERING_PROBE',
             original_coordinate_statistics={stage:dict(effect_energy=0.0,
                 repair_energy=1.0,effect_repair_inner_product=0.0,
                 nonzero_effect_coordinates=0,accumulation_dtype='float64')
                 for stage in ('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')})
    payload=dict(schema='selection-capture-recomputation-v1',records=[row],
                 all_comparisons_valid=True,claim_scope='FIXED',
                 equivalence_decision='NOT_ASSESSED',
                 execution_identity_independently_proven=False,training_quality_claim=False)
    recomputation.write_text(json.dumps(payload))
    monkeypatch.setattr('scripts.build_selection_family_evidence.verify',lambda root: payload)
    result=build(tmp_path,recomputation)['records'][0]
    assert result['family']=='SELECTION'
    assert result['position_inventory_count']==0
    assert result['exact_zero_effect_by_stage']['PARAMETER_WRITE'] is True
    assert result['equivalence_decision']=='NOT_ASSESSED'
