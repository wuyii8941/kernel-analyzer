import pytest
import torch
from kernel_analyzer.rotary_grouped_backward_reference import evaluate


def source_fixture(steps=64):
    import textwrap
    from kernel_analyzer.rotary_grouped_backward_reference import expected_body
    signature=dict(in_ptr0='*bf16',in_ptr1='*bf16',in_ptr2='*bf16',out_ptr0='*bf16',ynumel='i32',xnumel='i32')
    inner=(f'@heuristic(triton_meta={{"signature": {signature!r}}})\n'
           'def rotary(in_ptr0,in_ptr1,in_ptr2,out_ptr0,ynumel,xnumel,YBLOCK,XBLOCK):\n'
           +textwrap.indent(expected_body(768,steps),'    '))
    return f'rotary = async_compile.triton("rotary", {inner!r})'


@pytest.mark.parametrize('steps',[64,128])
def test_complete_source_contract(steps):
    from kernel_analyzer.rotary_grouped_backward_reference import check_source
    contract=check_source(source_fixture(steps),'rotary')
    assert (contract['groups'],contract['steps'])==(8,steps)


@pytest.mark.parametrize('old,new',[
    ('tmp10 = -tmp9','tmp10 = tmp9'),
    ('tmp0 >= tmp1','tmp0 > tmp1'),
    ('96*x2','95*x2'),
    ('*bf16','*fp32'),
])
def test_source_mutations_rejected(old,new):
    from kernel_analyzer.rotary_grouped_backward_reference import check_source
    source=source_fixture()
    assert old in source
    with pytest.raises(ValueError):
        check_source(source.replace(old,new),'rotary')


def test_runtime_reference_and_invalid_inputs():
    from kernel_analyzer.rotary_grouped_backward_reference import reference
    g=torch.randn(1,3,128,2).bfloat16()
    sine=torch.randn(2,96).bfloat16()
    cosine=torch.randn(2,96).bfloat16()
    output=torch.empty(1,96,2,dtype=torch.bfloat16)
    metadata={'input_output_storage_aliases':[],
              'runtime_pointers':dict(in_ptr0=g,in_ptr1=sine,in_ptr2=cosine)}
    contract=dict(groups=1,steps=2)
    assert torch.equal(reference(metadata,output,contract),
                       evaluate(g.float(),sine.float(),cosine.float()).bfloat16())
    metadata['input_output_storage_aliases']=['in_ptr0']
    with pytest.raises(ValueError): reference(metadata,output,contract)
    metadata['input_output_storage_aliases']=[]
    sine[0,0]=float('nan')
    with pytest.raises(ValueError): reference(metadata,output,contract)


@pytest.mark.parametrize('groups,repeats,width,rotary,steps',[(2,3,128,96,4),(1,1,8,6,7),(3,2,4,4,2)])
def test_matches_independent_forward_autograd(groups,repeats,width,rotary,steps):
    generator=torch.Generator().manual_seed(913)
    x=torch.randn(groups,rotary,steps,generator=generator,dtype=torch.float64,requires_grad=True)
    sine=torch.randn(steps,rotary,generator=generator,dtype=torch.float64)
    cosine=torch.randn(steps,rotary,generator=generator,dtype=torch.float64)
    upstream=torch.randn(groups,repeats,width,steps,generator=generator,dtype=torch.float64)
    half=rotary//2
    rotated=torch.cat((-x[:,half:,:],x[:,:half,:]),dim=1)
    forward=(x*cosine.T+rotated*sine.T)[:,None].expand(-1,repeats,-1,-1)
    expected=torch.autograd.grad((forward*upstream[:,:,:rotary,:]).sum(),x)[0]
    assert torch.allclose(evaluate(upstream,sine,cosine),expected,rtol=1e-12,atol=1e-12)


def test_unrotated_coordinates_are_not_part_of_this_output():
    g=torch.zeros(1,3,128,2,dtype=torch.float64)
    g[:,:,96:,:]=1000
    factors=torch.ones(2,96,dtype=torch.float64)
    assert torch.count_nonzero(evaluate(g,factors,factors))==0


def test_invalid_shapes_are_not_silently_broadcast():
    with pytest.raises(ValueError):
        evaluate(torch.zeros(1,3,128,2),torch.ones(2,95),torch.ones(2,95))


def test_generated_transposed_output_preserves_logical_coordinates():
    from kernel_analyzer.rotary_grouped_backward_reference import reference
    groups,steps=2,4
    g=torch.randn(groups*3,128,steps).bfloat16()
    sine=torch.randn(1,steps,96).bfloat16()
    cosine=torch.randn_like(sine)
    output=torch.empty_strided((1,groups,steps,96),(groups*96*steps,96*steps,1,steps),dtype=torch.bfloat16)
    metadata=dict(input_output_storage_aliases=[],runtime_pointers=dict(in_ptr0=g,in_ptr1=sine,in_ptr2=cosine))
    result=reference(metadata,output,dict(groups=groups,steps=steps))
    expected=evaluate(g.float().reshape(groups,3,128,steps),sine.float()[0],cosine.float()[0]).bfloat16()
    assert torch.equal(result,expected.transpose(-1,-2).unsqueeze(0))
    assert result.stride()==output.stride()
    assert not torch.equal(result,expected.reshape(output.shape))


def test_unknown_noncontiguous_output_still_rejected():
    from kernel_analyzer.rotary_grouped_backward_reference import reference
    output=torch.empty_strided((1,2,4,96),(1536,768,1,8),dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='Output layout'):
        reference(dict(input_output_storage_aliases=[]),output,dict(groups=2,steps=4))
