import textwrap
import pytest
import torch
from kernel_analyzer.scaled_masked_softmax_reference import BODY,check_source,evaluate,reference


def source():
    fn='def softmax(in_out_ptr0,in_ptr0,in_ptr1,in_ptr2,in_ptr3,xnumel,r0_numel,XBLOCK):\n'
    fn+=textwrap.indent(BODY.format(rows=8,width=4,scale=.125).strip(),'    ')
    return f'softmax=compile.triton("softmax",{fn!r})'


def test_formula_against_autograd_with_causal_mask():
    scores=torch.randn(8,4,dtype=torch.float64,requires_grad=True)
    mask=torch.triu(torch.full((4,4),-torch.inf,dtype=torch.float64),diagonal=1)
    mask=mask.repeat(2,1); scaled=scores*.125+mask
    maximum=scaled.detach().amax(-1,keepdim=True)
    denominator=(scaled.detach()-maximum).exp().sum(-1,keepdim=True)
    gradient=torch.randn_like(scores)
    expected,=torch.autograd.grad(torch.softmax(scaled,-1),scores,gradient)
    actual=evaluate(gradient,scores.detach(),mask,maximum,denominator,.125)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('old,new',[('xindex % 4','xindex % 3'),('tmp2 * tmp3','tmp2 + tmp3'),
    ('tmp4 + tmp5','tmp4 - tmp5'),('tmp20 * tmp3','tmp20'),
    ('in_ptr3 + x3','in_ptr3 + x0'),('tmp18, tmp17, tmp13','tmp12, tmp17, tmp13')])
def test_changed_semantics_rejected(old,new):
    with pytest.raises(ValueError): check_source(source().replace(old,new),'softmax')


def test_runtime_repeated_mask_and_saved_inputs():
    contract=check_source(source(),'softmax')
    pointers=dict(in_ptr0=torch.arange(32).reshape(8,4).float(),in_out_ptr0=torch.zeros(8,4),
                  in_ptr1=torch.zeros(4,4),in_ptr2=torch.zeros(8),in_ptr3=torch.full((8,),4.))
    actual=reference(dict(runtime_pointers=pointers,input_output_storage_aliases=[]),torch.empty(8,4),contract)
    g=pointers['in_ptr0']; expected=.125*(g*.25-.25*(g*.25).sum(-1,keepdim=True))
    torch.testing.assert_close(actual,expected)
    pointers['in_ptr3'][0]=0
    with pytest.raises(ValueError,match='normalizers'):
        reference(dict(runtime_pointers=pointers,input_output_storage_aliases=[]),torch.empty(8,4),contract)
