from pathlib import Path
import pytest
from kernel_analyzer.grouped_causal_softmax_source import check_source

SYMBOL = 'triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_7'
ROOT = Path(__file__).resolve().parents[1] / 'results/coverage/runtime_releases'


@pytest.mark.parametrize('release', ['qwen_seq64_r1', 'qwen_seq128_r1', 'qwen_seq256_r1',
                                   'deepseek8b_seq64_r1', 'deepseek8b_seq128_r1', 'deepseek8b_seq256_r1'])
def test_real_sources(release):
    source = (ROOT / release / 'trace/model__0_forward_segment0_executed/output_code.py').read_text()
    result = check_source(source, SYMBOL)
    assert result['width'] == int(release.split('seq')[1].split('_')[0])
    assert len(result['output_pointers']) == 4
    assert not result['runtime_binding_complete']


@pytest.mark.parametrize('old,new', [('tmp3 <= tmp4', 'tmp3 < tmp4'),
                                   ('tmp8 == tmp9', 'tmp8 != tmp9'),
                                   ('tmp30 / tmp28', 'tmp30 * tmp28'),
                                   ("'out_ptr1': '*fp32'", "'out_ptr1': '*bf16'")])
def test_changed_semantics_rejected(old, new):
    source = (ROOT / 'qwen_seq128_r1/trace/model__0_forward_segment0_executed/output_code.py').read_text()
    assert old in source
    with pytest.raises(ValueError):
        check_source(source.replace(old, new), SYMBOL)
