import pytest
import torch
from kernel_analyzer.row_sum_reference import BODY, check_source, reference


def source(rows=3,width=5):
    body = BODY.format(rows=rows,width=width,block=1 << (width-1).bit_length())
    signature = dict(in_ptr0='*fp32',out_ptr0='*bf16',xnumel='i32',r0_numel='i32')
    code = '@check(triton_meta=' + repr(dict(signature=signature)) + ')\n'
    code += 'def k(in_ptr0,out_ptr0,xnumel,r0_numel,XBLOCK:tl.constexpr):\n'
    code += '\n'.join('    '+line for line in body.strip().splitlines())
    return 'k = async_compile.triton("k", '+repr(code)+')'


@pytest.mark.parametrize('width',[1,5,8,32])
def test_reference_and_masked_padding(width):
    c = check_source(source(width=width),'k')
    x = torch.arange(3*width,dtype=torch.float32).reshape(3,width)-3
    y = reference(dict(runtime_pointers=dict(in_ptr0=x),input_output_storage_aliases=[]),
                  torch.empty(3,dtype=torch.bfloat16),c)
    assert torch.equal(y,x.double().sum(-1).to(torch.bfloat16))


@pytest.mark.parametrize('old,new',[('tl.sum(tmp3, 1)','tl.max(tmp3, 1)'),
                                   ('5*x0','4*x0'),('*fp32','*bf16'),
                                   ('r0_mask & xmask','xmask')])
def test_reject_changed_semantics(old,new):
    with pytest.raises(ValueError):
        check_source(source().replace(old,new),'k')


def test_no_unverified_alias_or_nonfinite_input():
    c = check_source(source(),'k')
    x = torch.full((3,5),float('inf'))
    for meta in [dict(runtime_pointers=dict(in_ptr0=x)),
                 dict(runtime_pointers=dict(in_ptr0=x),input_output_storage_aliases=[])]:
        with pytest.raises(ValueError):
            reference(meta,torch.empty(3,dtype=torch.bfloat16),c)
