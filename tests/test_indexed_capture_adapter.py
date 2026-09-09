import pytest
from scripts.run_indexed_accumulation_capture import translate_plan


def test_translation_preserves_scientific_reference_and_original_plan():
    case=dict(case_id='test',reference_method='INDEXED_INPUT_ORDER_FP32',
              implementation_kind='DIRECT_ATEN',expected_symbol='index_put_',
              exact_aot_endpoint_id='backward:index_put',carrier='embedding.weight')
    result=translate_plan(dict(cases=[case]))['cases'][0]
    assert result['reference_method']=='PARTIAL_REDUCTION_FROM_BOUND_INPUT'
    assert result['declared_reference_method']==case['reference_method']
    assert case['reference_method']=='INDEXED_INPUT_ORDER_FP32'
    with pytest.raises(ValueError): translate_plan(dict(cases=[dict(case,exact_aot_endpoint_id=None)]))
    with pytest.raises(ValueError): translate_plan(dict(cases=[]))
