import pytest
import torch
from kernel_analyzer.indexed_row_accumulation_reference import evaluate


def test_duplicate_indices_match_embedding_gradient_without_mutation():
    weights=torch.zeros(5,3,requires_grad=True)
    indices=torch.tensor([2,1,2,4,1],dtype=torch.int64)
    values=torch.arange(15,dtype=torch.float32).reshape(5,3)
    loss=(torch.nn.functional.embedding(indices,weights)*values).sum()
    expected=torch.autograd.grad(loss,weights)[0]
    initial=torch.zeros_like(weights)
    actual=evaluate(initial,indices,values)
    assert torch.equal(actual,expected)
    assert torch.count_nonzero(initial)==0


def test_initial_content_negative_indices_and_empty_input():
    initial=torch.ones(3,2)
    actual=evaluate(initial,torch.tensor([-1,2]),torch.tensor([[2.,3.],[4.,5.]]))
    assert torch.equal(actual,torch.tensor([[1.,1.],[1.,1.],[7.,9.]]))
    assert torch.equal(evaluate(initial,torch.empty(0,dtype=torch.int64),torch.empty(0,2)),initial)


def test_order_is_declared_not_claimed_exact_or_unbiased():
    initial=torch.zeros(1,1)
    indices=torch.zeros(3,dtype=torch.int64)
    values=torch.tensor([[2.**25],[1.],[-2.**25]])
    assert evaluate(initial,indices,values).item()==0
    assert evaluate(initial,indices,values[[0,2,1]]).item()==1


@pytest.mark.parametrize('indices',[torch.tensor([3]),torch.tensor([-4]),torch.tensor([0.])])
def test_invalid_index_contract(indices):
    with pytest.raises(ValueError): evaluate(torch.zeros(3,2),indices,torch.ones(1,2))
