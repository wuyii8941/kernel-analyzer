import textwrap
import pytest
import torch
from kernel_analyzer.silu_backward_reference import BODY, check_source, evaluate, reference


def source(elements=24):
    code = 'def silu(in_out_ptr0,in_ptr0,in_ptr1,out_ptr0,xnumel,XBLOCK):\n' + textwrap.indent(
        BODY.format(elements=elements).strip(), '    ')
    return f'silu = async_compile.triton("silu", {code!r})'


def test_derivative_against_independent_autograd():
    gen = torch.Generator().manual_seed(745)
    gate = torch.randn(24, dtype=torch.float64, generator=gen).requires_grad_()
    up = torch.randn(24, dtype=torch.float64, generator=gen)
    gradient = torch.randn(24, dtype=torch.float64, generator=gen)
    expected, = torch.autograd.grad(up * torch.nn.functional.silu(gate), gate, gradient)
    torch.testing.assert_close(evaluate(gradient, gate.detach(), up), expected, rtol=1e-12, atol=1e-12)


def test_source_and_same_inputs():
    c = check_source(source(), 'silu')
    pointers = {'in_ptr0': torch.ones(24), 'in_ptr1': torch.zeros(24), 'in_out_ptr0': torch.full((24,), 2.)}
    result = reference(dict(runtime_pointers=pointers, input_output_storage_aliases=[]), torch.empty(24), c)
    assert torch.equal(result, torch.ones(24))
    assert c['unchanged_output_pointer'] == 'out_ptr0'
    pointers['in_out_ptr0'].fill_(4.)
    assert torch.equal(reference(dict(runtime_pointers=pointers, input_output_storage_aliases=[]), torch.empty(24), c), torch.full((24,),2.))


@pytest.mark.parametrize('old,new', [('tmp19 = tmp15 * tmp18','tmp19 = tmp15 + tmp18'),
    ('tmp3 = -tmp2','tmp3 = tmp2'), ('in_ptr1 + (x0)','in_ptr1 + (x0+1)'),
    ('in_out_ptr0,in_ptr0,in_ptr1','in_ptr0,in_out_ptr0,in_ptr1')])
def test_source_changes_rejected(old,new):
    with pytest.raises(ValueError): check_source(source().replace(old,new), 'silu')


@pytest.mark.parametrize('bad', ['alias', 'missing', 'shape', 'dtype'])
def test_invalid_runtime_rejected(bad):
    pointers = {k:torch.ones(24) for k in ('in_ptr0','in_ptr1','in_out_ptr0')}
    meta = dict(runtime_pointers=pointers, input_output_storage_aliases=[])
    if bad == 'alias': meta['input_output_storage_aliases'] = ['in_ptr0']
    if bad == 'missing': pointers.pop('in_ptr1')
    if bad == 'shape': pointers['in_ptr0'] = torch.ones(23)
    if bad == 'dtype': pointers['in_ptr0'] = torch.ones(24,dtype=torch.int64)
    with pytest.raises(ValueError): reference(meta,torch.empty(24),check_source(source(),'silu'))
