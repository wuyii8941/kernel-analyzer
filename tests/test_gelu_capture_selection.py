import pytest
from scripts.run_gelu_product_capture import select


def test_selection_cannot_change_parameter_or_repeat_task():
    case=dict(task_id='backward:1',expected_symbol='gelu',carrier='weight')
    plan=dict(schema='gelu-product-task-plan-v1',cases=[case],contracts={'gelu':{'elements':6}})
    assert select(plan,[case])==plan['contracts']
    with pytest.raises(ValueError):select(plan,[dict(case,carrier='other')])
    with pytest.raises(ValueError):select(plan,[case,case])
    with pytest.raises(ValueError):select(plan,[])
def test_state_count_rejects_short_or_repeated_bank():
    import pytest
    from scripts.run_gelu_product_capture import check_state_count
    bank={'states':[{'state_id':str(i)} for i in range(26)]}
    with pytest.raises(ValueError,match='Insufficient'):
        check_state_count(bank,['--states','32'])
    check_state_count(bank,['--states','26'])
    with pytest.raises(ValueError,match='Duplicate'):
        check_state_count({'states':[{'state_id':'same'}]*32},['--states','32'])
