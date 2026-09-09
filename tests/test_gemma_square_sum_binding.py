import pytest
import torch
import textwrap
from scripts.recover_gemma_square_sum_boundary import check, reference, EXPECTED, SYMBOL


def test_original_source_proves_square_sum_and_mutations_fail():
    code=f'def {SYMBOL}(in_ptr0, out_ptr0, XBLOCK, R0_BLOCK):\n'+textwrap.indent(EXPECTED.strip(),'    ')
    source=f'{SYMBOL} = async_compile.triton({SYMBOL!r}, {code!r})'
    assert len(check(source))==64
    with pytest.raises(ValueError):check(source.replace('tmp2 = tmp1 * tmp1','tmp2 = tmp1 + tmp1'))
    with pytest.raises(ValueError):check(source.replace('tl.store(out_ptr0 + (x0), tmp4, xmask)','tl.store(out_ptr0 + (x0), tmp4 + 1, xmask)'))


def test_bound_reference_and_wrong_layout():
    x=torch.arange(128*1536).reshape(128,1536).to(torch.bfloat16)
    candidate=torch.empty(1,128,1)
    r=reference({'runtime_pointers':{'in_ptr0':x}},candidate)
    assert torch.equal(r.flatten(),x.float().square().sum(-1))
    with pytest.raises(RuntimeError):reference({'runtime_pointers':{'in_ptr0':x.T}},candidate)
    with pytest.raises(RuntimeError):reference({'runtime_pointers':{'in_ptr0':x.float()}},candidate)
