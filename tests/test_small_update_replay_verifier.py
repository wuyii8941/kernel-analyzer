import copy
import pytest
import torch
from kernel_analyzer.update_write import adamw_parameter_write
from kernel_analyzer.update_write_diagnostics import SmallUpdateRecorder
from scripts.verify_small_update_replay import verify


def example():
    recorder=SmallUpdateRecorder(adamw_parameter_write)
    base=torch.ones(2)
    candidate=recorder(base,torch.tensor([1e-7,1.]))
    reference=recorder(base,torch.tensor([-1e-7,1.]))
    raw={'case_id':'test','carrier':'weight','state_ids':['state'], 'status':'COMPLETE',
         'calibration_state_ids':[], 'confirmation_state_ids':['state'],
         'optimizer':{}, 'runtime_boundary':{}, 'reference_comparison_scope':{},
         'parameter_write_protocol':{'version':'adamw-readback-v2'},
         'original_coordinate_statistics':{'PARAMETER_WRITE':[{
             'effect_energy':float((candidate-reference).double().square().sum()),
             'nonzero_effect_coordinates':1}]}}
    return raw,copy.deepcopy(raw),recorder.finish(raw)


def test_vector_replay_verifies_sign_change_without_persistence_claim():
    result=verify(*example())
    assert result['status']=='VERIFIED_REPLAY'
    assert result['changed_coordinates'][0]['gradient_sign_changed']
    assert 'no persistence' in result['scope']


@pytest.mark.parametrize('error',['write','input','state','statistics','coordinate','missing_identity','incomplete'])
def test_tampered_evidence_rejected(error):
    original,repeated,diagnostics=example()
    row=diagnostics['rows'][0]
    if error=='write': row['candidate']['actual_write'][0]=1.
    elif error=='input': row['candidate']['gradient'][0]=-1.
    elif error=='state': row['state_id']='other'
    elif error=='statistics': repeated['original_coordinate_statistics']['PARAMETER_WRITE'][0]['effect_energy']=1.
    elif error=='coordinate': row['nonzero_update_coordinates']=[]
    elif error=='missing_identity': original.pop('runtime_boundary'); repeated.pop('runtime_boundary')
    else: original['status']='PARTIAL'; repeated['status']='PARTIAL'
    with pytest.raises(ValueError): verify(original,repeated,diagnostics)
