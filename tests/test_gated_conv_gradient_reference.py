import pytest
import torch
from kernel_analyzer.gated_conv_gradient_reference import evaluate


def source_fixture(channels=1536,steps=64,padding=3):
    import textwrap
    from kernel_analyzer.gated_conv_gradient_reference import BODY
    names=['in_out_ptr0','in_ptr0','in_ptr1','in_ptr2','in_ptr3','in_ptr4','out_ptr0']
    signature={n:'*fp32' if n in ('in_ptr1','in_ptr3') else '*bf16' for n in names}
    signature.update(xnumel='i32',r0_numel='i32')
    inner=(f'@heuristic(triton_meta={{"signature":{signature!r}}})\n'
           'def kernel('+','.join(names+['xnumel','r0_numel','XBLOCK','R0_BLOCK'])+'):\n'
           +textwrap.indent(BODY.format(channels=channels,steps=steps,padded=steps+padding),'    '))
    return f'kernel=async_compile.triton("kernel",{inner!r})'


@pytest.mark.parametrize('steps',[64,128,256])
def test_source_dimensions_and_two_explicit_outputs(steps):
    from kernel_analyzer.gated_conv_gradient_reference import check_source
    for output in ('out_ptr0','in_out_ptr0'):
        c=check_source(source_fixture(steps=steps),'kernel',output)
        assert (c['steps'],c['padding'],c['output_pointer'])==(steps,3,output)


@pytest.mark.parametrize('old,new',[
    ('tmp28 = tmp7 - tmp26','tmp28 = tmp7 + tmp26'),
    ('tmp38 = tl.sum(_tmp38, 1)','tmp38 = tl.sum(_tmp38.to(tl.bfloat16), 1)'),
    ('67*x0','66*x0'),('*bf16','*fp32')])
def test_source_semantic_changes_rejected(old,new):
    from kernel_analyzer.gated_conv_gradient_reference import check_source
    source=source_fixture()
    assert old in source
    with pytest.raises(ValueError): check_source(source.replace(old,new),'kernel')


@pytest.mark.parametrize('channels,steps,padding', [(2,7,3), (3,4,0), (1,1,3)])
def test_against_independent_forward_autograd(channels,steps,padding):
    gen=torch.Generator().manual_seed(851)
    def random(*shape): return torch.randn(*shape,generator=gen,dtype=torch.float64)
    x=random(channels,steps+padding).requires_grad_()
    bias=random(channels).requires_grad_()
    gate=random(steps,channels)
    upstream,recurrent,skip=[random(channels,steps) for _ in range(3)]
    scale=random(channels)
    h=torch.nn.functional.silu(x[:,:steps]+bias[:,None])
    loss=(h*torch.nn.functional.silu(gate.T)*scale[:,None]*upstream+h*recurrent+h*skip).sum()
    expected=torch.autograd.grad(loss,(x,bias))
    preactivation=torch.cat((x[:,:steps]+bias[:,None],x[:,steps:]),dim=1)
    actual=evaluate(preactivation,gate,upstream,scale,recurrent,skip)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_bias_reduction_must_precede_bfloat16_write():
    x=torch.zeros(1,67)
    skip=torch.full((1,64),1.003)
    gradient,bias=evaluate(x,torch.zeros(64,1),torch.zeros(1,64),torch.ones(1),torch.zeros(1,64),skip)
    assert torch.equal(bias,gradient.sum(1))
    assert torch.count_nonzero(gradient[:,64:])==0
    # Storage conversion is intentionally outside the mathematical reference.
    assert gradient.dtype==torch.float32
    assert not torch.equal(bias,gradient.bfloat16().float().sum(1))


def test_incompatible_layouts_rejected():
    with pytest.raises(ValueError):
        evaluate(torch.zeros(1,2),torch.zeros(3,1),torch.zeros(1,3),
                 torch.ones(1),torch.zeros(1,3),torch.zeros(1,3))


@pytest.mark.parametrize('output',['in_out_ptr0','out_ptr0'])
def test_runtime_uses_precall_values_and_explicit_output(output):
    from kernel_analyzer.gated_conv_gradient_reference import reference
    pointers={name:torch.ones(shape,dtype=dtype) for name,shape,dtype in [
        ('in_out_ptr0',(2,7),torch.bfloat16),('in_ptr0',(2,4),torch.bfloat16),
        ('in_ptr1',(4,2),torch.float32),('in_ptr2',(2,),torch.bfloat16),
        ('in_ptr3',(2,4),torch.float32),('in_ptr4',(2,4),torch.bfloat16)]}
    metadata=dict(runtime_pointers=pointers,formal_pointer=output,input_output_storage_aliases=[])
    contract=dict(channels=2,steps=4,padding=3,output_pointer=output)
    candidate=torch.full((2,7) if output=='in_out_ptr0' else (2,),999,dtype=torch.bfloat16)
    actual=reference(metadata,candidate,contract)
    expected=evaluate(*(pointers[n].float() for n in (
        'in_out_ptr0','in_ptr1','in_ptr0','in_ptr2','in_ptr3','in_ptr4')))
    assert torch.equal(actual,expected[0 if output=='in_out_ptr0' else 1].bfloat16())
    with pytest.raises(ValueError):
        reference(dict(metadata,formal_pointer='wrong'),candidate,contract)
