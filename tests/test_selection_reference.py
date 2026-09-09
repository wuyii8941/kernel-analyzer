import pytest
import torch
from kernel_analyzer.selection_reference import stable_selection,validate_selection


def test_legal_boundary_ties_not_errors():
    x=torch.tensor([[4.,3.,3.,1.]])
    result=validate_selection(x,torch.tensor([[4.,3.]]),torch.tensor([[0,2]]),2)
    assert result['boundary_ties']
    assert not result['training_bias_confirmed']


def test_unsorted_and_smallest():
    x=torch.tensor([[4.,2.,3.,1.]])
    validate_selection(x,torch.tensor([[2.,1.]]),torch.tensor([[1,3]]),2,
                       largest=False,sorted_output=False)
    with pytest.raises(ValueError,match='order'):
        validate_selection(x,torch.tensor([[2.,1.]]),torch.tensor([[1,3]]),2,largest=False)


def test_invalid_outputs_and_nonfinite():
    x=torch.tensor([[4.,3.,2.]])
    for values,ids in [([4.,4.],[0,0]),([4.,2.],[0,2]),([4.,3.],[0,9]),([4.,9.],[0,1])]:
        with pytest.raises(ValueError):
            validate_selection(x,torch.tensor([values]),torch.tensor([ids]),2)
    with pytest.raises(ValueError,match='Nonfinite'):
        stable_selection(torch.tensor([float('nan')]),1)


def test_reference_preserves_gradient_and_noncontiguous_inputs():
    base=torch.tensor([[1.,4.],[3.,2.]],requires_grad=True)
    x=base.T
    values,indices,_=stable_selection(x,1)
    validate_selection(x,values,indices,1)
    values.sum().backward()
    assert torch.equal(base.grad,torch.tensor([[0.,1.],[1.,0.]]))
