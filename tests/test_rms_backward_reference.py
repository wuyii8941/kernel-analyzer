import textwrap
import pytest
import torch
from kernel_analyzer.rms_backward_reference import BODY, check_source, evaluate, reference


def source(rows=3, width=8):
    code = 'def rms(in_out_ptr0,in_ptr0,in_ptr1,in_ptr2,in_ptr3,in_ptr4,XBLOCK,R0_BLOCK):\n' + textwrap.indent(
        BODY.format(rows=rows, width=width, inverse_width=1./width).strip(), '    ')
    return f'rms = async_compile.triton("rms", {code!r})'


def test_formula_matches_independent_autograd_rmsnorm():
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(3, 8, dtype=torch.float64, generator=generator).requires_grad_()
    w = torch.randn(8, dtype=torch.float64, generator=generator)
    p0 = torch.randn(3, 8, dtype=torch.float64, generator=generator)
    p1 = torch.randn(3, 8, dtype=torch.float64, generator=generator)
    a = torch.randn(3, 8, dtype=torch.float64, generator=generator)
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    gradient, = torch.autograd.grad(x * r * w, x, p0 + p1)
    torch.testing.assert_close(evaluate(a, p0, p1, w, x.detach(), r.detach()), a + gradient,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('rows,width', [(3, 8), (64, 4096), (256, 4096)])
def test_checked_dimensions(rows, width):
    contract = check_source(source(rows, width), 'rms')
    assert (contract['rows'], contract['width']) == (rows, width)


@pytest.mark.parametrize('old,new', [('tmp35 = tmp12 + tmp34', 'tmp35 = tmp12 - tmp34'),
                                   ('0.125', '0.25'), ('other=0.0', 'other=1.0'),
                                   ('8*x0', '9*x0')])
def test_wrong_formula_or_address_is_rejected(old, new):
    with pytest.raises(ValueError): check_source(source().replace(old, new), 'rms')


def test_reference_requires_saved_accumulator_and_layout():
    contract = check_source(source(), 'rms')
    pointers = {name: torch.ones(3, 8) for name in ('in_out_ptr0', 'in_ptr0', 'in_ptr1', 'in_ptr3')}
    pointers.update(in_ptr2=torch.ones(8), in_ptr4=torch.ones(3))
    result = reference({'runtime_pointers': pointers, 'input_output_storage_aliases': []}, torch.empty(3, 8), contract)
    assert torch.equal(result, torch.ones(3, 8))
    del pointers['in_out_ptr0']
    with pytest.raises(ValueError): reference({'runtime_pointers': pointers, 'input_output_storage_aliases': []}, torch.empty(3, 8), contract)


@pytest.mark.parametrize('aliases', [None, ['in_ptr0'], ['in_ptr4']])
def test_shared_or_unknown_read_input_storage_is_rejected(aliases):
    # Rejection happens before reading tensors: cloned tensors cannot establish
    # whether the original call used overlapping input/output storage.
    metadata = {'runtime_pointers': {}}
    if aliases is not None:
        metadata['input_output_storage_aliases'] = aliases
    with pytest.raises(ValueError, match='Nonaliasing'):
        reference(metadata, torch.empty(3, 8), check_source(source(), 'rms'))


def test_saved_inverse_rms_dtype_is_part_of_reference_contract():
    pointers = {name: torch.ones(3, 8) for name in ('in_out_ptr0', 'in_ptr0', 'in_ptr1', 'in_ptr3')}
    pointers.update(in_ptr2=torch.ones(8), in_ptr4=torch.ones(3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='Saved inverse RMS must be FP32'):
        reference({'runtime_pointers': pointers, 'input_output_storage_aliases': []},
                  torch.empty(3, 8), check_source(source(), 'rms'))
