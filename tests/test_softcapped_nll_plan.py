from copy import deepcopy
import pytest
from scripts.run_softcapped_nll_capture import validate_plan


def plan():
    return dict(schema='softcapped-nll-task-plan-v1', trainable_parameters=['weight'],
                contracts={'kernel': {}}, cases=[dict(task_id='backward:1:in_out_ptr0',
                carrier='weight', reference_method='SOFTCAPPED_NLL_COMMON_INPUT', expected_symbol='kernel')])


def test_valid_plan():
    validate_plan(plan())


@pytest.mark.parametrize('change', ['schema', 'empty', 'duplicate', 'parameter', 'reference', 'symbol', 'scope'])
def test_refuse_invalid_plan(change):
    p = deepcopy(plan())
    if change == 'schema': p['schema'] = 'selected-nll-task-plan-v1'
    elif change == 'empty': p['cases'] = []
    elif change == 'duplicate': p['cases'] *= 2
    elif change == 'scope': p['trainable_parameters'].append('another')
    else:
        field = {'parameter':'carrier', 'reference':'reference_method', 'symbol':'expected_symbol'}[change]
        p['cases'][0][field] = 'different'
    with pytest.raises(ValueError):
        validate_plan(p)
