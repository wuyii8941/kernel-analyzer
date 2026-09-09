import pytest
import torch
from kernel_analyzer.gelu_product_reference import evaluate, decode_inputs


def test_tanh_gelu_derivative_matches_independent_autograd():
    x=torch.linspace(-8,8,101,dtype=torch.float64,requires_grad=True)
    grad=torch.linspace(-2,3,101,dtype=torch.float64)
    multiplier=torch.linspace(1,-1,101,dtype=torch.float64)
    y=torch.nn.functional.gelu(x,approximate='tanh')
    expected=torch.autograd.grad(y,x,grad*multiplier)[0]
    actual=evaluate(grad,multiplier,x.detach(),accumulation_dtype=torch.float64)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-14)


def test_no_implicit_broadcast_or_nonfinite_acceptance():
    x=torch.ones(4)
    with pytest.raises(ValueError):evaluate(x,x[:1],x)
    with pytest.raises(ValueError):evaluate(x,x,torch.full_like(x,float('nan')))
    assert evaluate(x,x,torch.zeros_like(x)).equal(torch.full_like(x,.5))


def test_declared_bf16_write():
    x=torch.linspace(-3,3,31,dtype=torch.bfloat16)
    result=evaluate(x,x,x,output_dtype=torch.bfloat16)
    expected=evaluate(x.float(),x.float(),x.float()).to(torch.bfloat16)
    assert torch.equal(result,expected)


def test_packed_selection_nonzero_storage_offset_and_snapshot():
    packed=torch.arange(30,dtype=torch.bfloat16)[3:27].reshape(3,8)
    g=torch.arange(6,dtype=torch.bfloat16)
    x=torch.ones(6,dtype=torch.bfloat16)
    pointers=dict(in_ptr0=g,in_ptr1=packed,in_ptr2=x)
    gradient,multiplier,saved=decode_inputs(pointers,elements=6,width=2,stride=8,offset=5)
    assert torch.equal(multiplier,packed[:,5:7].reshape(-1))
    expected=multiplier.clone()
    packed.zero_();g.zero_();x.zero_()
    assert torch.equal(multiplier,expected)
    assert gradient.sum()==15 and saved.sum()==6


def test_reject_noncontiguous_or_incomplete_packed_input():
    pointers=dict(in_ptr0=torch.ones(6,dtype=torch.bfloat16),
                  in_ptr1=torch.ones(8,3,dtype=torch.bfloat16).T,
                  in_ptr2=torch.ones(6,dtype=torch.bfloat16))
    with pytest.raises(ValueError):decode_inputs(pointers,elements=6,width=2,stride=8,offset=5)
    pointers['in_ptr1']=torch.ones(20,dtype=torch.bfloat16)
    with pytest.raises(ValueError):decode_inputs(pointers,elements=6,width=2,stride=8,offset=5)
