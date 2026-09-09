from pathlib import Path
import pytest
from kernel_analyzer.gelu_product_source import check_source

SOURCE=Path('results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/runtime_release_rebound_current/trace/model__1_backward_segment0_executed/output_code.py')
SYMBOL='triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_select_view_4'


def check(text):
    return check_source(text,SYMBOL,elements=32768,width=256,stride=8960,offset=8704)


def test_actual_definition_matches_but_is_not_runtime_proof():
    contract=check(SOURCE.read_text())
    assert contract['output_pointer']=='out_ptr0'
    assert not contract['runtime_binding_complete']


@pytest.mark.parametrize('old,new',[
    ('tmp13 = libdevice.tanh(tmp12)','tmp13 = libdevice.erf(tmp12)'),
    ('8704 + x0 + 8960*x1','8703 + x0 + 8960*x1'),
    ('tmp28 = tmp3 * tmp27','tmp28 = tmp3 + tmp27')])
def test_reject_semantic_or_address_change(old,new):
    text=SOURCE.read_text()
    assert old in text
    with pytest.raises(ValueError):check(text.replace(old,new))
