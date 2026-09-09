import pytest
import torch
from kernel_analyzer.update_write import adamw_parameter_write
from kernel_analyzer.update_write_diagnostics import SmallUpdateRecorder


def test_recording_preserves_original_write_and_reproduces_energy():
    recorder = SmallUpdateRecorder(adamw_parameter_write)
    base = torch.ones(3)
    cgrad, rgrad = torch.tensor([1.,-1e-7,2.]), torch.tensor([1.,1e-7,2.])
    c = recorder(base,cgrad); r = recorder(base,rgrad)
    assert torch.equal(c,adamw_parameter_write(base,cgrad))
    assert torch.equal(r,adamw_parameter_write(base,rgrad))
    raw = {'case_id':'test','carrier':'weight','state_ids':['s'],
           'original_coordinate_statistics': {'PARAMETER_WRITE': [
               {'effect_energy':float((c-r).double().square().sum()), 'nonzero_effect_coordinates':1}]}}
    result = recorder.finish(raw)
    assert result['rows'][0]['nonzero_update_coordinates']==[1]
    assert not result['primary_analysis_changed']
    raw['original_coordinate_statistics']['PARAMETER_WRITE'][0]['effect_energy'] = 1.
    with pytest.raises(ValueError,match='energy'):
        recorder.finish(raw)


def test_large_parameter_refused_before_execution():
    recorder = SmallUpdateRecorder(lambda *a,**k: pytest.fail('must not execute'),max_elements=2)
    with pytest.raises(ValueError,match='too large'):
        recorder(torch.ones(3),torch.ones(3))


def test_missing_calls_are_not_associated_with_states():
    with pytest.raises(ValueError,match='calls'):
        SmallUpdateRecorder(adamw_parameter_write).finish({'state_ids':['s']})
