import pytest
import torch
from kernel_analyzer.decayed_recurrence_reference import evaluate


@pytest.mark.parametrize('steps',[1,3,8])
def test_independent_expanded_sum(steps):
    gen=torch.Generator().manual_seed(52)
    initial=torch.randn(2,3,generator=gen,dtype=torch.float64)
    rate=torch.randn(2,3,generator=gen,dtype=torch.float64)
    times=torch.randn(steps,2,generator=gen,dtype=torch.float64)
    bias=torch.randn(2,generator=gen,dtype=torch.float64)
    additions=torch.randn(steps,2,3,generator=gen,dtype=torch.float64)
    # Independent expansion of every prefix, not another recurrence loop.
    dt=torch.logaddexp(torch.zeros_like(times),times+bias)
    factors=torch.exp(-torch.exp(rate)[None]*dt[...,None])
    expected=[]
    for t in range(steps):
        value=initial*factors[:t+1].prod(0)
        for k in range(t+1):
            value=value+additions[k]*factors[k+1:t+1].prod(0)
        expected.append(value)
    assert torch.allclose(evaluate(initial,rate,times,bias,additions),torch.stack(expected),atol=1e-12,rtol=1e-12)


def test_large_positive_softplus_input_stays_finite():
    result=evaluate(torch.ones(1,1),torch.zeros(1,1),torch.tensor([[1000.]]),torch.zeros(1),torch.ones(1,1,1))
    assert torch.equal(result,torch.ones(1,1,1))


def test_broadcasting_not_silently_allowed():
    with pytest.raises(ValueError):
        evaluate(torch.ones(2,3),torch.zeros(1,3),torch.zeros(4,2),torch.zeros(2),torch.zeros(4,2,3))


def test_pointer_time_order_and_every_output():
    from kernel_analyzer.decayed_recurrence_reference import from_runtime_pointers
    pointers={f'in_ptr{i}':torch.tensor([float(i+1)]).bfloat16() for i in range(11)}
    pointers['in_ptr2']=torch.zeros(1)
    pointers['in_ptr3']=torch.tensor([100.,1.,2.,3.]).bfloat16()
    pointers['in_ptr4']=torch.zeros(1).bfloat16()
    state=torch.tensor(2.)
    for j,time in enumerate((3.,2.,1.)):
        state=state*torch.exp(-torch.nn.functional.softplus(torch.tensor(time)))+(6+2*j)*(7+2*j)
        got=from_runtime_pointers(pointers,channels=1,width=1,outputs=3,output_index=j)
        assert torch.allclose(got.squeeze(),state)
    with pytest.raises(ValueError):
        from_runtime_pointers(pointers,channels=1,width=1,outputs=3,output_index=3)
    pointers['in_ptr0']=pointers['in_ptr0'].float()
    with pytest.raises(ValueError,match='in_ptr0'):
        from_runtime_pointers(pointers,channels=1,width=1,outputs=3,output_index=0)


def test_runtime_boundary_and_dependency_rejected():
    import hashlib
    from pathlib import Path
    from kernel_analyzer import decayed_recurrence_source
    from kernel_analyzer.decayed_recurrence_reference import reference
    contract=dict(symbol='recurrence',output_pointer='out_ptr0',output_pointers=['out_ptr0'],
        checker_dependency_sha256='incorrect')
    with pytest.raises(ValueError,match='checker changed'):
        reference({},torch.zeros(1),contract)
    contract['checker_dependency_sha256']=hashlib.sha256(Path(decayed_recurrence_source.__file__).read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='identity'):
        reference(dict(symbol='recurrence',formal_pointer='out_ptr1',input_output_storage_aliases=[]),torch.zeros(1),contract)


def test_singleton_strides_and_storage_offset_match_flat_pointer_values():
    from kernel_analyzer.decayed_recurrence_reference import from_runtime_pointers
    channels,width,outputs=3,2,1
    sizes=[channels,width,channels*width,(outputs+1)*channels,channels,channels,width]
    pointers={}
    for i,size in enumerate(sizes):
        dtype=torch.float32 if i==2 else torch.bfloat16
        storage=torch.arange(size+7,dtype=dtype)/16
        # Source inputs include singleton dimensions with strides exceeding
        # the storage size. Only non-singleton dimensions address data.
        pointers[f'in_ptr{i}']=storage.as_strided((1,1,size),(5120,1,1),storage_offset=7)
        assert pointers[f'in_ptr{i}'].is_contiguous()
    flat={k:v.flatten().clone() for k,v in pointers.items()}
    args=dict(channels=channels,width=width,outputs=outputs,output_index=0)
    assert torch.equal(from_runtime_pointers(pointers,**args),from_runtime_pointers(flat,**args))
    pointers['in_ptr1']=torch.ones(4,dtype=torch.bfloat16)[::2]
    with pytest.raises(ValueError,match='in_ptr1'):
        from_runtime_pointers(pointers,**args)
