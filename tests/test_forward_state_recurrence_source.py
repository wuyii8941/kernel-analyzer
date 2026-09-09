from pathlib import Path
import pytest
from kernel_analyzer.forward_state_recurrence_source import check_source


SYMBOL = 'triton_poi_fused__to_copy__unsafe_view_add_exp_mul_neg_select_softplus_split_with_sizes_transpose_unsqueeze_zeros_4'
SOURCE = Path(__file__).resolve().parents[1] / 'results/coverage/runtime_releases/mamba_seq64_r1/trace/model__0_forward_segment0_executed/output_code.py'


def test_observed_full_source():
    result = check_source(SOURCE.read_text(), SYMBOL)
    assert result['steps'] == 64
    assert not result['runtime_binding_complete']
    assert not result['population_bias_proved']


@pytest.mark.parametrize('before,after', [
    ('tmp2 = -tmp1', 'tmp2 = tmp1'),
    ('tmp23 = tmp20 * tmp22', 'tmp23 = tmp20 + tmp22'),
    ('48 + x0', '49 + x0'),
    ('63 + 64*x1', '62 + 64*x1'),
    ("'out_ptr63': '*bf16'", "'out_ptr63': '*fp32'"),
    ('tmp1222, None)', 'tmp1202, None)'),
])
def test_reject_modified_source(before, after):
    source = SOURCE.read_text()
    assert before in source
    with pytest.raises(ValueError):
        check_source(source.replace(before, after), SYMBOL)
