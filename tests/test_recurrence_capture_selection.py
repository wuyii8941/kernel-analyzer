import pytest
from scripts.run_decayed_recurrence_capture import select_contracts
from scripts.run_decayed_recurrence_capture import plan_method


def plan():
    return dict(contract=dict(symbol='kernel',output_pointers=['out_ptr0','out_ptr1']),cases=[
        dict(case_id=str(i),expected_symbol='kernel',reference_output_pointer=f'out_ptr{i}',
             reference_output_index=i) for i in range(2)])


def test_two_outputs_keep_distinct_reference_selection():
    p=plan(); result=select_contracts(p,p['cases'])
    assert set(result)=={('kernel','out_ptr0'),('kernel','out_ptr1')}
    assert result[('kernel','out_ptr1')]['output_pointer']=='out_ptr1'


def test_changed_output_is_not_accepted_as_original_case():
    p=plan()
    with pytest.raises(ValueError):
        select_contracts(p,[dict(p['cases'][0],reference_output_pointer='out_ptr1')])


def test_empty_selection_rejected():
    with pytest.raises(ValueError): select_contracts(plan(),[])


def test_duplicate_selection_rejected():
    p=plan()
    with pytest.raises(ValueError,match='Duplicate selected'):
        select_contracts(p,[p['cases'][0],p['cases'][0]])


def test_duplicate_source_plan_rejected():
    p=plan(); p['cases'].append(p['cases'][0])
    with pytest.raises(ValueError,match='Duplicate case IDs'):
        select_contracts(p,p['cases'][:1])


@pytest.mark.parametrize('field,value', [('expected_symbol','other'),('reference_output_index',1)])
def test_invalid_binding_rejected_even_when_present_in_plan(field,value):
    p=plan(); p['cases'][0][field]=value
    with pytest.raises(ValueError): select_contracts(p,p['cases'])


def test_continued_plan_uses_shared_dispatch():
    p=plan()
    p['schema']='continued-recurrence-bound-plan-v1'
    for c in p['cases']:
        c['reference_method']='CONTINUED_RECURRENCE_COMMON_INPUT'
    assert plan_method(p)==('CONTINUED_RECURRENCE_COMMON_INPUT','CONTINUED_RECURRENCE')
    assert len(select_contracts(p,p['cases']))==2
    p['cases'][0]['reference_method']='SEGMENTED_RECURRENCE_FIRST_COMMON_INPUT'
    with pytest.raises(ValueError): plan_method(p)
