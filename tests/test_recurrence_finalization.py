import pytest
from scripts.finalize_decayed_recurrence import verify_translation


def plans():
    case=dict(case_id='one',task_id='task',carrier='weight',expected_symbol='kernel',
              reference_output_pointer='out_ptr0',reference_output_index=0,
              reference_method='DECAYED_RECURRENCE_COMMON_INPUT')
    original=dict(contract=dict(symbol='kernel',output_pointers=['out_ptr0']),cases=[case])
    translated=dict(cases=[dict(case,reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                               declared_reference_method=case['reference_method'])])
    return original,translated


def test_only_dispatch_translation_is_allowed():
    original,translated=plans()
    assert verify_translation(original,translated)==original['cases']
    assert translated['cases'][0]['reference_method']=='PARTIAL_REDUCTION_FROM_BOUND_INPUT'


@pytest.mark.parametrize('field,value',[
    ('carrier','other'),('task_id','other'),('expected_symbol','other'),
    ('reference_output_pointer','out_ptr1'),('reference_output_index',1),
    ('declared_reference_method','OTHER'),('reference_method','OTHER')])
def test_translation_must_not_change_scientific_identity(field,value):
    original,translated=plans(); translated['cases'][0][field]=value
    with pytest.raises(ValueError): verify_translation(original,translated)


def test_segmented_plan_uses_declared_method_without_changing_case():
    from scripts.run_decayed_recurrence_capture import plan_method
    original,translated=plans()
    original['schema']='segmented-recurrence-first-bound-plan-v1'
    method='SEGMENTED_RECURRENCE_FIRST_COMMON_INPUT'
    original['cases'][0]['reference_method']=method
    translated['cases'][0]['declared_reference_method']=method
    assert plan_method(original)==(method,'SEGMENTED_RECURRENCE_FIRST')
    assert verify_translation(original,translated)==original['cases']
    original['cases'][0]['reference_method']='DECAYED_RECURRENCE_COMMON_INPUT'
    with pytest.raises(ValueError,match='method differs'): plan_method(original)
