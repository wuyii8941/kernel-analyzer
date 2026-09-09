import pytest
import torch
from kernel_analyzer.softplus_bias_backward_reference import BODY, EXTRA_STORE, FINAL_STORE, check_source, evaluate, reference


def source(channels=7, steps=8, extra=False):
    pointers = ['in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'out_ptr0'] + (['out_ptr1'] if extra else [])
    types = dict(zip(pointers, ['*bf16', '*bf16', '*fp32', '*fp32', '*bf16'] + (['*bf16'] if extra else [])))
    types.update(xnumel='i32', r0_numel='i32')
    body = (BODY+(EXTRA_STORE if extra else '')+FINAL_STORE).format(channels=channels, steps=steps)
    code = '@check(triton_meta='+repr({'signature':types})+')\ndef kernel('+','.join(pointers+['xnumel','r0_numel','XBLOCK: tl.constexpr'])+'):\n'
    code += '\n'.join('    '+line for line in body.strip().splitlines())
    return 'kernel = async_compile.triton("kernel", '+repr(code)+')'


@pytest.mark.parametrize('extra', [False, True])
def test_source_and_independent_autograd(extra):
    contract = check_source(source(extra=extra), 'kernel')
    assert contract['extra_output_present'] is extra
    torch.manual_seed(37)
    x = torch.randn(8,7, dtype=torch.float64)*15
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    g1, g2 = torch.randn(2,7,8,dtype=torch.float64)
    y = torch.nn.functional.softplus(x+bias, beta=1, threshold=20)
    expected, = torch.autograd.grad(y, bias, (g1+g2).T)
    torch.testing.assert_close(evaluate(x,bias,g1,g2), expected)


@pytest.mark.parametrize('old,new', [('tmp13 * tmp14','tmp13 + tmp14'),
    ('1536','1537'), ('20.0','19.0'), ('*bf16','*fp16'),
    ('tmp3 * tmp4','tmp3 + tmp4'), ('x0 + 1536 * r0_1','r0_1 + 1536 * x0')])
def test_changed_formula_or_layout_rejected(old,new):
    text = source(channels=1536)
    # Change only an address for the literal-dimension mutation.
    if old == '1536':
        text = text.replace('1536 * r0_1', '1537 * r0_1')
    else:
        text = text.replace(old,new)
    with pytest.raises(ValueError): check_source(text,'kernel')


def test_duplicate_and_unmasked_padding_rejected():
    with pytest.raises(ValueError): check_source(source()+'\n'+source(),'kernel')
    with pytest.raises(ValueError): check_source(source(steps=7),'kernel')


def test_runtime_reference_and_invalid_inputs():
    contract = check_source(source(),'kernel')
    values = dict(in_ptr0=torch.randn(8,7).bfloat16(), in_ptr1=torch.randn(7).bfloat16(),
                  in_ptr2=torch.randn(7,8), in_ptr3=torch.randn(7,8))
    meta = dict(input_output_storage_aliases=[], runtime_pointers=values)
    candidate = torch.zeros(7, dtype=torch.bfloat16)
    expected = evaluate(values['in_ptr0'].float(),values['in_ptr1'].float(),values['in_ptr2'],values['in_ptr3']).bfloat16()
    assert torch.equal(reference(meta,candidate,contract),expected)
    with pytest.raises(ValueError): reference({**meta,'input_output_storage_aliases':None},candidate,contract)
    values['in_ptr0'] = values['in_ptr0'].T
    with pytest.raises(ValueError): reference(meta,candidate,contract)


def test_extremes_and_threshold_are_finite():
    x = torch.tensor([[-1000.,0.,20.,21.,1000.]],dtype=torch.float64)
    result = evaluate(x,torch.zeros(5,dtype=torch.float64),torch.ones(5,1,dtype=torch.float64),torch.zeros(5,1,dtype=torch.float64))
    assert torch.isfinite(result).all()
    assert result[0] == 0 and result[-1] == 1 and result[1] == .5
