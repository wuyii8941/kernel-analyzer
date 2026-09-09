import pytest
import torch
import ast
from pathlib import Path

from kernel_analyzer.source_reference_registry import REFERENCES, get_reference
from kernel_analyzer.row_reduction_reference import VARIANTS
from kernel_analyzer.selected_silu_product_reference import reference
from kernel_analyzer.residual_rms_forward_source import expected_body as residual_rms_body


def test_legacy_variant_policy_preserved():
    assert get_reference('ROW_SQUARE_SUM').variants == VARIANTS
    assert get_reference('EXPONENTIAL_WEIGHTED_REDUCTION').variants == (
        'FP32_NATIVE', 'FP32_REVERSE_COMPONENT_ORDER', 'FP64_EVALUATION')
    for family in REFERENCES:
        if family not in ('ROW_SQUARE_SUM', 'EXPONENTIAL_WEIGHTED_REDUCTION'):
            assert get_reference(family).variants == ('FP32_NATIVE',)


def test_unknown_family_and_variant_never_fall_back():
    with pytest.raises(ValueError,match='Unsupported'):
        get_reference('UNKNOWN_KERNEL')
    with pytest.raises(ValueError,match='Variant'):
        get_reference('SELECTED_SILU_PRODUCT').evaluate({},None,{},variant='FP64_EVALUATION')


def test_softcapped_nll_is_registered_as_one_reviewed_family():
    reference = get_reference('SOFTCAPPED_NLL_BACKWARD')
    assert reference.module == 'softcapped_nll_registered_reference'
    assert reference.variants == ('FP32_NATIVE',)
    assert reference.case_suffix == '-softcapped-nll-common-input'


def test_selected_nll_is_registered_as_one_reviewed_family():
    reference = get_reference('SELECTED_NLL_BACKWARD')
    assert reference.module == 'selected_nll_registered_reference'
    assert reference.variants == ('FP32_NATIVE',)
    assert reference.case_suffix == '-selected-nll-common-input'


@pytest.mark.parametrize(('family', 'module', 'suffix'), [
    ('GROUPED_CAUSAL_SOFTMAX_PROBABILITY',
     'grouped_causal_softmax_registered_reference',
     '-grouped-causal-softmax-probability-common-input'),
    ('RESIDUAL_RMS_FORWARD_NORMALIZED',
     'residual_rms_forward_registered_reference',
     '-residual-rms-normalized-common-input'),
    ('RMS_FORWARD_NORMALIZED',
     'rms_forward_registered_reference',
     '-rms-forward-normalized-common-input'),
    ('RMS_SIMPLE_BACKWARD',
     'rms_simple_backward_registered_reference',
     '-rms-simple-backward-common-input'),
    ('GELU_PRODUCT_BACKWARD',
     'gelu_product_registered_reference',
     '-gelu-product-common-input'),
    ('GELU_FORWARD_PRODUCT',
     'gelu_forward_product_registered_reference',
     '-gelu-forward-product-common-input'),
    ('CHANNEL_BIAS_ADD',
     'channel_bias_registered_reference',
     '-channel-bias-common-input'),
    ('ATTENTION_POSITION_SCALING',
     'attention_position_scaling_registered_reference',
     '-attention-position-scaling-common-input'),
    ('FORWARD_STATE_RECURRENCE_FINAL',
     'forward_state_recurrence_registered_reference',
     '-forward-state-recurrence-final-common-input'),
])
def test_multi_output_families_register_one_explicit_output(family, module, suffix):
    specification = get_reference(family)
    assert specification.module == module
    assert specification.variants == ('FP32_NATIVE',)
    assert specification.case_suffix == suffix


def test_registry_reuses_selected_reference_without_metric_changes():
    metadata=dict(input_output_storage_aliases=[],runtime_pointers=dict(
        in_ptr0=torch.arange(12).reshape(3,4).float(),
        in_ptr1=torch.arange(12).reshape(4,3).float().T.unsqueeze(0)))
    contract=dict(elements=3,sequence_length=4,selected_step=2)
    candidate=torch.empty(3)
    expected=reference(metadata,candidate,contract)
    actual=get_reference('SELECTED_SILU_PRODUCT').evaluate(
        metadata,candidate,contract,variant='FP32_NATIVE')
    assert torch.equal(actual,expected)


def test_grouped_causal_softmax_registry_binds_only_probability_output():
    source = (Path(__file__).resolve().parents[1]
              / 'results/coverage/runtime_releases/qwen_seq64_r1/trace/'
              / 'model__0_forward_segment0_executed/output_code.py').read_text()
    symbol = ('triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_'
              'index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_7')
    specification = get_reference('GROUPED_CAUSAL_SOFTMAX_PROBABILITY')
    contract = specification.check_source(source, symbol)
    assert contract['output_pointer'] == 'out_ptr2'
    rows, width = contract['rows'], contract['width']
    pointers = {
        'in_out_ptr0': torch.zeros(rows * width, dtype=torch.bfloat16),
        'in_ptr0': torch.zeros(width, dtype=torch.int64),
        'out_ptr0': torch.empty(rows),
        'out_ptr1': torch.empty(rows),
        'out_ptr2': torch.empty(rows * width, dtype=torch.bfloat16),
    }
    candidate = torch.empty(rows * width, dtype=torch.bfloat16)
    metadata = dict(symbol=symbol, formal_pointer='out_ptr2',
                    input_output_storage_aliases=[], runtime_pointers=pointers)
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    probability = result.reshape(rows, width)
    assert probability.shape == (rows, width)
    assert torch.equal(probability[0], torch.nn.functional.one_hot(
        torch.tensor(0), width).to(torch.bfloat16))
    torch.testing.assert_close(probability.float().sum(-1), torch.ones(rows),
                               atol=2e-3, rtol=2e-3)
    with pytest.raises(ValueError, match='probability boundary'):
        specification.evaluate(dict(metadata, formal_pointer='out_ptr0'), candidate,
                               contract, variant='FP32_NATIVE')


def _residual_rms_literal_source():
    signature = dict(in_out_ptr0='*bf16', in_out_ptr1='*fp32', in_ptr0='*bf16',
                     in_ptr1='*bf16', out_ptr0='*bf16', xnumel='i32', r0_numel='i32')
    inner = (f'@wrap(triton_meta={{"signature": {signature!r}}})\n'
             'def kernel(' + ','.join([*signature, 'XBLOCK', 'R0_BLOCK']) + '):\n')
    inner += '\n'.join('    ' + line for statement in residual_rms_body(2, 4, 1e-6)
                        for line in ast.unparse(statement).splitlines())
    return f'kernel=async_compile.triton("kernel",{inner!r})'


def test_residual_rms_registry_binds_only_normalized_output():
    specification = get_reference('RESIDUAL_RMS_FORWARD_NORMALIZED')
    contract = specification.check_source(_residual_rms_literal_source(), 'kernel')
    assert contract['output_pointer'] == 'out_ptr0'
    pointers = {
        'in_out_ptr0': torch.full((8,), 1 / 256, dtype=torch.bfloat16),
        'in_out_ptr1': torch.empty(2),
        'in_ptr0': torch.ones(8, dtype=torch.bfloat16),
        'in_ptr1': torch.ones(4, dtype=torch.bfloat16),
        'out_ptr0': torch.empty(8, dtype=torch.bfloat16),
    }
    candidate = torch.empty(8, dtype=torch.bfloat16)
    metadata = dict(symbol='kernel', formal_pointer='out_ptr0',
                    input_output_storage_aliases=[], runtime_pointers=pointers)
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    assert result.shape == candidate.shape
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match='normalized-output boundary'):
        specification.evaluate(dict(metadata, formal_pointer='in_out_ptr1'), candidate,
                               contract, variant='FP32_NATIVE')


def test_rms_forward_registry_binds_normalized_output_only():
    source_path = (Path(__file__).resolve().parents[1]
                   / 'results/property/declared_persistent_4096/gemma4_norm_v3_long/'
                   / 'runtime_release/trace/model__0_forward_segment0_executed/output_code.py')
    symbol = 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_view_21'
    specification = get_reference('RMS_FORWARD_NORMALIZED')
    contract = specification.check_source(source_path.read_text(), symbol)
    assert (contract['rows'], contract['width'], contract['epsilon']) == (128, 512, 1e-6)
    elements = contract['rows'] * contract['width']
    pointers = {
        'in_out_ptr0': torch.empty(contract['rows'], dtype=torch.float32),
        'in_ptr0': torch.ones(elements, dtype=torch.bfloat16),
        'out_ptr0': torch.empty(elements, dtype=torch.bfloat16),
    }
    candidate = torch.empty(elements, dtype=torch.bfloat16)
    metadata = dict(symbol=symbol, formal_pointer='out_ptr0',
                    input_output_storage_aliases=[], runtime_pointers=pointers)
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    expected = torch.full_like(result, 1 / (1 + contract['epsilon']) ** 0.5)
    assert torch.equal(result, expected)
    with pytest.raises(ValueError, match='normalized-output'):
        specification.evaluate(dict(metadata, formal_pointer='in_out_ptr0'), candidate,
                               contract, variant='FP32_NATIVE')


def test_simple_rms_backward_registry_uses_pre_call_gradient():
    source_path = (Path(__file__).resolve().parents[1]
                   / 'results/property/declared_persistent_4096/gemma4_norm_v3_long/'
                   / 'runtime_release/trace/model__1_backward_segment0_executed/output_code.py')
    symbol = ('triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_'
              'squeeze_sum_transpose_view_58')
    specification = get_reference('RMS_SIMPLE_BACKWARD')
    contract = specification.check_source(source_path.read_text(), symbol)
    assert (contract['rows'], contract['width']) == (128, 512)
    elements = contract['rows'] * contract['width']
    pointers = {
        'in_out_ptr0': torch.ones(elements, dtype=torch.bfloat16),
        'in_ptr0': torch.zeros(elements, dtype=torch.bfloat16),
        'in_ptr1': torch.ones(contract['rows'], dtype=torch.float32),
    }
    candidate = torch.empty(elements, dtype=torch.bfloat16)
    metadata = dict(symbol=symbol, formal_pointer='in_out_ptr0',
                    input_output_storage_aliases=[], runtime_pointers=pointers)
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    assert torch.equal(result, torch.ones_like(result))


def test_gelu_product_registry_infers_layout_and_reuses_common_snapshot():
    source_path = (Path(__file__).resolve().parents[1]
                   / 'results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/'
                   / 'runtime_release_rebound_current/trace/'
                   / 'model__1_backward_segment0_executed/output_code.py')
    symbol = 'triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_select_view_4'
    specification = get_reference('GELU_PRODUCT_BACKWARD')
    contract = specification.check_source(source_path.read_text(), symbol)
    assert (contract['elements'], contract['width'], contract['stride'], contract['offset']) == (
        32768, 256, 8960, 8704)
    rows = contract['elements'] // contract['width']
    pointers = {
        'in_ptr0': torch.ones(contract['elements'], dtype=torch.bfloat16),
        'in_ptr1': torch.ones(rows * contract['stride'], dtype=torch.bfloat16),
        'in_ptr2': torch.zeros(contract['elements'], dtype=torch.bfloat16),
        'out_ptr0': torch.empty(contract['elements'], dtype=torch.bfloat16),
    }
    candidate = torch.empty(contract['elements'], dtype=torch.bfloat16)
    metadata = dict(symbol=symbol, formal_pointer='out_ptr0',
                    input_output_storage_aliases=[], runtime_pointers=pointers)
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    assert torch.equal(result, torch.full_like(result, 0.5))


@pytest.mark.parametrize(('symbol', 'layout', 'offset'), [
    ('triton_poi_fused__unsafe_view_gelu_mul_select_16', 'PACKED_SELECTION', 256),
    ('triton_poi_fused__unsafe_view_gelu_mul_select_13', 'PACKED_SELECTION', 0),
    ('triton_poi_fused__unsafe_view_gelu_mul_11', 'CONTIGUOUS', None),
])
def test_forward_gelu_product_registry_accepts_reviewed_layouts(symbol, layout, offset):
    source_path = (Path(__file__).resolve().parents[1]
                   / 'results/property/declared_persistent_4096/gemma4_norm_v3_long/'
                   / 'runtime_release/trace/model__0_forward_segment0_executed/output_code.py')
    specification = get_reference('GELU_FORWARD_PRODUCT')
    contract = specification.check_source(source_path.read_text(), symbol)
    assert contract['layout'] == layout
    assert contract['offset'] == offset
    elements = contract['elements']
    second_size = (elements if layout == 'CONTIGUOUS'
                   else elements // contract['width'] * contract['stride'])
    pointers = {
        'in_ptr0': torch.ones(elements, dtype=torch.bfloat16),
        'in_ptr1': torch.full((second_size,), 2, dtype=torch.bfloat16),
        'out_ptr0': torch.empty(elements, dtype=torch.bfloat16),
    }
    candidate = torch.empty(elements, dtype=torch.bfloat16)
    metadata = dict(symbol=symbol, formal_pointer='out_ptr0',
                    input_output_storage_aliases=[], runtime_pointers=pointers)
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    expected = (torch.nn.functional.gelu(torch.ones(elements), approximate='tanh') * 2
                ).to(torch.bfloat16)
    assert torch.equal(result, expected)


def test_forward_gelu_product_registry_rejects_inplace_and_arithmetic_mutation():
    source_path = (Path(__file__).resolve().parents[1]
                   / 'results/property/declared_persistent_4096/gemma4_norm_v3_long/'
                   / 'runtime_release/trace/model__0_forward_segment0_executed/output_code.py')
    source = source_path.read_text()
    specification = get_reference('GELU_FORWARD_PRODUCT')
    symbol = 'triton_poi_fused__unsafe_view_gelu_mul_select_16'
    with pytest.raises(ValueError, match='arithmetic'):
        specification.check_source(source.replace(
            'tmp6 = tl.full([1], 0.044715, tl.float32)',
            'tmp6 = tl.full([1], 0.04, tl.float32)'), symbol)
    inplace_source = (Path(__file__).resolve().parents[1]
                      / 'results/property/declared_persistent_4096/operator_scan_targets/'
                      / 'gemma4_scan_00_pilot_gpu2b/runtime_release/trace/'
                      / 'model__0_forward_segment0_executed/output_code.py').read_text()
    inplace = 'triton_poi_fused__unsafe_view_gelu_mul_11'
    with pytest.raises(ValueError, match='signature'):
        specification.check_source(inplace_source, inplace)


def test_channel_bias_registry_reuses_pre_call_inplace_value():
    source_path = (Path(__file__).resolve().parents[1]
                   / 'results/coverage/runtime_releases/mamba_seq64_r1/trace/'
                   / 'model__0_forward_segment0_executed/output_code.py')
    symbol = 'triton_poi_fused__unsafe_view_convolution_split_transpose_2'
    specification = get_reference('CHANNEL_BIAS_ADD')
    contract = specification.check_source(source_path.read_text(), symbol)
    assert (contract['channels'], contract['length'], contract['output_pointer']) == (
        1536, 67, 'in_out_ptr0')
    base = torch.ones(contract['channels'] * contract['length'], dtype=torch.bfloat16)
    bias = torch.arange(contract['channels'], dtype=torch.bfloat16)
    candidate = torch.empty_like(base)
    metadata = dict(symbol=symbol, formal_pointer='in_out_ptr0',
                    input_output_storage_aliases=[],
                    runtime_pointers={'in_out_ptr0': base, 'in_ptr0': bias})
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    expected = (base.reshape(contract['channels'], contract['length']).float()
                + bias.float()[:, None]).to(torch.bfloat16).reshape(-1)
    assert torch.equal(result, expected)


def test_forward_recurrence_registry_binds_only_final_state():
    source_path = (Path(__file__).resolve().parents[1]
                   / 'results/coverage/runtime_releases/mamba_seq64_r1/trace/'
                   / 'model__0_forward_segment0_executed/output_code.py')
    symbol = ('triton_poi_fused__to_copy__unsafe_view_add_exp_mul_neg_select_'
              'softplus_split_with_sizes_transpose_unsqueeze_zeros_4')
    specification = get_reference('FORWARD_STATE_RECURRENCE_FINAL')
    source_contract = specification.check_source(source_path.read_text(), symbol)
    assert source_contract['steps'] == 64
    assert source_contract['output_pointer'] == 'out_ptr63'

    contract = dict(symbol='kernel', steps=3, channels=2, state_width=2,
                    packed_width=6, state_offset=2, output_pointer='out_ptr2')
    pointers = dict(
        in_ptr0=torch.zeros(4),
        in_ptr1=torch.zeros(6, dtype=torch.bfloat16),
        in_ptr2=torch.zeros(2, dtype=torch.bfloat16),
        in_ptr3=torch.ones(18, dtype=torch.bfloat16),
        in_ptr4=torch.ones(6, dtype=torch.bfloat16),
        out_ptr0=torch.empty(4), out_ptr1=torch.empty(4),
        out_ptr2=torch.empty(4, dtype=torch.bfloat16),
    )
    candidate = torch.empty(4, dtype=torch.bfloat16)
    metadata = dict(symbol='kernel', formal_pointer='out_ptr2',
                    input_output_storage_aliases=[], runtime_pointers=pointers)
    result = specification.evaluate(metadata, candidate, contract, variant='FP32_NATIVE')
    assert result.shape == candidate.shape
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match='final-output boundary'):
        specification.evaluate(dict(metadata, formal_pointer='out_ptr1'), candidate,
                               contract, variant='FP32_NATIVE')
