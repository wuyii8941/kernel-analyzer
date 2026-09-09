from pathlib import Path
import pytest
from kernel_analyzer.channel_bias_source import check_source

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = 'triton_poi_fused__unsafe_view_convolution_split_transpose_2'


@pytest.mark.parametrize('length', [64, 128, 256])
def test_actual_bias_source(length):
    path = ROOT/f'results/coverage/runtime_releases/mamba_seq{length}_r1/trace/model__0_forward_segment0_executed/output_code.py'
    result = check_source(path.read_text(), SYMBOL, length=length+3)
    assert not result['runtime_binding_complete']


def test_reject_changed_arithmetic():
    path = ROOT/'results/coverage/runtime_releases/mamba_seq64_r1/trace/model__0_forward_segment0_executed/output_code.py'
    with pytest.raises(ValueError):
        check_source(path.read_text().replace('tmp2 = tmp0 + tmp1', 'tmp2 = tmp0 * tmp1'), SYMBOL)


def test_accept_inductor_signature_with_explicit_xblock():
    path = ROOT/'results/coverage/runtime_releases/mamba_seq64_r1/trace/model__0_forward_segment0_executed/output_code.py'
    source = path.read_text().replace(
        "'in_ptr0': '*bf16', 'xnumel': 'i32'},",
        "'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'},",
        1,
    )
    result = check_source(source, SYMBOL)
    assert result['channels'] == 1536
    assert result['length'] == 67
    original = check_source(path.read_text(), SYMBOL)
    assert (result['function_semantic_ast_sha256']
            == original['function_semantic_ast_sha256'])
    assert result['function_ast_sha256'] != original['function_ast_sha256']


@pytest.mark.parametrize(
    'replacement',
    [
        "'in_ptr0': '*fp32', 'xnumel': 'i32'},",
        "'in_ptr0': '*bf16', 'xnumel': 'i32', 'YBLOCK': 'constexpr'},",
        "'in_ptr0': '*bf16', 'xnumel': 'i64'},",
    ],
)
def test_reject_changed_or_unknown_signature_entries(replacement):
    path = ROOT/'results/coverage/runtime_releases/mamba_seq64_r1/trace/model__0_forward_segment0_executed/output_code.py'
    source = path.read_text().replace(
        "'in_ptr0': '*bf16', 'xnumel': 'i32'},",
        replacement,
        1,
    )
    with pytest.raises(ValueError, match='Bias storage precision differs'):
        check_source(source, SYMBOL)
