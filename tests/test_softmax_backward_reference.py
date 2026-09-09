import textwrap
import pytest
import torch
from kernel_analyzer.softmax_backward_reference import BODY, check_source, evaluate, reference


def source(rows=3, width=8, scale=.125):
    code = 'def softmax(in_out_ptr0,in_ptr0,in_ptr1,in_ptr2,XBLOCK):\n' + textwrap.indent(
        BODY.format(rows=rows, width=width, scale=scale).strip(), '    ')
    return f'softmax = async_compile.triton("softmax", {code!r})'


def test_formula_matches_independent_autograd():
    generator = torch.Generator().manual_seed(84)
    scores = torch.randn(3, 8, dtype=torch.float64, generator=generator).requires_grad_()
    gradient = torch.randn(3, 8, dtype=torch.float64, generator=generator)
    expected, = torch.autograd.grad(scores.softmax(-1), scores, gradient)
    maximum = scores.detach().amax(-1, keepdim=True)
    denominator = (scores.detach() - maximum).exp().sum(-1, keepdim=True)
    observed = evaluate(gradient, scores.detach(), maximum, denominator, .125)
    torch.testing.assert_close(observed, expected * .125, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('rows,width,scale', [(3,8,.125),(1024,64,.08838834764831845),(4096,256,.125)])
def test_source_dimensions_and_declared_scale(rows, width, scale):
    contract = check_source(source(rows, width, scale), 'softmax')
    assert (contract['rows'], contract['width'], contract['scale']) == (rows,width,scale)


@pytest.mark.parametrize('old,new', [('tmp14 = -tmp8','tmp14 = tmp8'),
                                   ('8*x0','9*x0'), ('tmp6 / tmp7','tmp6 * tmp7'),
                                   ('tl.sum(tmp12, 1)','tl.sum(tmp12, 0)')])
def test_source_mutations_rejected(old, new):
    with pytest.raises(ValueError):
        check_source(source().replace(old,new), 'softmax')


def inputs():
    return {'in_ptr0': torch.arange(24).reshape(3,8).float(), 'in_out_ptr0': torch.zeros(3,8),
            'in_ptr1': torch.zeros(3), 'in_ptr2': torch.full((3,),8.)}


def test_reference_uses_saved_values_not_a_new_forward():
    pointers = inputs()
    contract = check_source(source(), 'softmax')
    candidate = torch.empty(3,8)
    meta = {'runtime_pointers': pointers, 'input_output_storage_aliases': []}
    expected = (pointers['in_ptr0'] - pointers['in_ptr0'].mean(-1,keepdim=True)) / 8 * .125
    torch.testing.assert_close(reference(meta,candidate,contract), expected)
    pointers['in_ptr2'].fill_(16.)
    assert not torch.equal(reference(meta,candidate,contract), expected)


@pytest.mark.parametrize('bad', ['missing', 'dtype', 'zero', 'alias'])
def test_invalid_saved_state_or_alias_rejected(bad):
    pointers = inputs()
    meta = {'runtime_pointers': pointers, 'input_output_storage_aliases': []}
    if bad == 'missing': pointers.pop('in_ptr1')
    elif bad == 'dtype': pointers['in_ptr1'] = pointers['in_ptr1'].bfloat16()
    elif bad == 'zero': pointers['in_ptr2'].zero_()
    else: meta['input_output_storage_aliases'] = ['in_ptr0']
    with pytest.raises(ValueError):
        reference(meta,torch.empty(3,8),check_source(source(),'softmax'))
