import pytest
import torch

from kernel_analyzer.silu_backward_reference import VARIANTS, check_source, evaluate, reference


def test_factorial_variants_change_only_declared_evaluation_choices():
    gradient = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float32)
    gate = torch.tensor([-2.0, 0.25, 3.0], dtype=torch.float32)
    up = torch.tensor([1.2, -0.4, 0.8], dtype=torch.float32)

    values = {name: evaluate(gradient, gate, up, variant=name) for name in VARIANTS}

    assert set(values) == set(VARIANTS)
    for value in values.values():
        assert value.shape == gate.shape
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()


def test_reference_preserves_candidate_storage_representation():
    gradient = torch.tensor([0.3, -0.7], dtype=torch.bfloat16)
    gate = torch.tensor([-2.0, 0.25], dtype=torch.bfloat16)
    up = torch.tensor([1.2, -0.4], dtype=torch.bfloat16)
    candidate = torch.empty(2, dtype=torch.bfloat16)
    metadata = {
        'input_output_storage_aliases': [],
        'runtime_pointers': {
            'in_ptr0': gradient,
            'in_ptr1': gate,
            'in_out_ptr0': up,
        },
    }
    contract = {'elements': 2}

    output = reference(
        metadata,
        candidate,
        contract,
        variant='EXPLICIT_EXP_SOURCE_ORDER',
    )

    assert output.shape == candidate.shape
    assert output.dtype == candidate.dtype


def test_unknown_variant_is_rejected():
    value = torch.ones(2)
    with pytest.raises(ValueError, match='Unsupported SiLU backward variant'):
        evaluate(value, value, value, variant='UNKNOWN')


def test_source_identity_ignores_device_specific_decorator_metadata():
    import textwrap

    body = """{prefix} = factory('name', '''
@pointwise(device={device})
def {prefix}(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
{body}
''')
"""
    from kernel_analyzer.silu_backward_reference import BODY

    symbol = 'generated_silu'
    function_body = textwrap.indent(BODY.format(elements=8).strip(), '    ')
    source0 = body.format(prefix=symbol, device=0, body=function_body)
    source1 = body.format(prefix=symbol, device=1, body=function_body)
    assert check_source(source0, symbol)['function_ast_sha256'] == check_source(
        source1, symbol
    )['function_ast_sha256']
