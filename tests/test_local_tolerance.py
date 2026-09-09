import pytest
import torch
from kernel_analyzer.local_tolerance import compare_outputs


def test_tolerance_uses_actual_elements_not_residual_rms():
    r=torch.tensor([100.,0.]); c=torch.tensor([100.,1e-4])
    result=compare_outputs(c,r,rtol=1e-5,atol=1e-8)
    assert result['allclose'] is False
    assert result['violating_coordinates']==1


def test_boundary_and_identity():
    result=compare_outputs(torch.tensor([.125]),torch.zeros(1),rtol=0.,atol=.125)
    assert result['allclose'] is True
    assert compare_outputs(torch.zeros(2),torch.zeros(2),rtol=0.,atol=0.)['allclose'] is True


def test_nonfinite_is_not_a_numeric_negative_or_safe_identity():
    result=compare_outputs(torch.tensor([float('inf')]),torch.tensor([float('inf')]),rtol=1e-5,atol=1e-8)
    assert result['allclose'] is None
    assert result['status']=='NONFINITE_OUTPUT'


@pytest.mark.parametrize('rtol,atol',[(-1,0),(0,float('nan')),(float('inf'),0)])
def test_invalid_threshold_rejected(rtol,atol):
    with pytest.raises(ValueError): compare_outputs(torch.ones(1),torch.ones(1),rtol=rtol,atol=atol)
