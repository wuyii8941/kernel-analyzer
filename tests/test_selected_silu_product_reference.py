import pytest
import torch

from kernel_analyzer.selected_silu_product_reference import BODY, check_source, reference


def source(body=None):
    body = body or BODY.format(elements=4, length=8, step=3, offset=12)
    fn = 'def selected(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK):\n'
    fn += '\n'.join('    '+line for line in body.strip().splitlines())
    return 'selected = compile.triton("selected", '+repr(fn)+')'


def test_formula_matches_independent_autograd():
    contract = check_source(source(), 'selected')
    activation = torch.randn(8, 4)
    gradient = torch.randn(4, 8)
    candidate = torch.empty(4)
    result = reference(dict(input_output_storage_aliases=[], runtime_pointers=dict(
        in_ptr0=gradient, in_ptr1=activation)), candidate, contract)
    multiplier = torch.ones(4, requires_grad=True)
    loss = (multiplier * torch.nn.functional.silu(activation[3]) * gradient[:, 3]).sum()
    expected, = torch.autograd.grad(loss, multiplier)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('before,after', [('tmp2 * tmp9','tmp2 + tmp9'),
    ('xindex < xnumel','xindex <= xnumel'), ('12 + x0','13 + x0'),
    ('3 + 8 * x0','8 + 8 * x0'), ('out_ptr0 + x0','out_ptr0 + 2 * x0'),
    ('tmp3 / tmp7','tmp3 * tmp7')])
def test_semantic_changes_rejected(before, after):
    text = source().replace(before, after)
    with pytest.raises(ValueError):
        check_source(text, 'selected')


def test_runtime_requires_layout_and_alias_evidence():
    contract = check_source(source(), 'selected')
    with pytest.raises(ValueError, match='Nonaliasing'):
        reference({}, torch.empty(4), contract)
    with pytest.raises(ValueError, match='layout'):
        reference(dict(input_output_storage_aliases=[], runtime_pointers=dict(
            in_ptr0=torch.empty(4,16)[:,::2], in_ptr1=torch.empty(8,4))), torch.empty(4), contract)


def test_simplified_zero_step_address_and_mismatch():
    body = BODY.format(elements=4, length=8, step=0, offset=0)
    body = body.replace('(0 + 8 * x0)', '8 * x0').replace('(0 + x0)', 'x0')
    assert check_source(source(body), 'selected')['selected_step'] == 0
    with pytest.raises(ValueError):
        check_source(source(body.replace('in_ptr1 + x0', 'in_ptr1 + 4 + x0')), 'selected')


def test_saved_activation_uses_physical_not_logical_flattening():
    contract=check_source(source(),'selected')
    physical=torch.arange(32,dtype=torch.float32).reshape(8,4)-10
    saved=physical.T.unsqueeze(0)
    gradient=torch.arange(32,dtype=torch.float32).reshape(4,8)
    result=reference(dict(input_output_storage_aliases=[],runtime_pointers=dict(
        in_ptr0=gradient,in_ptr1=saved)),torch.empty(4),contract)
    torch.testing.assert_close(result,gradient[:,3]*torch.nn.functional.silu(physical[3]))
