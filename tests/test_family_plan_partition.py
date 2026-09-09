import pytest
from scripts.partition_family_plan import partitions


def test_keeps_every_case_in_order():
    cases = [dict(case_id=str(i), task_id=str(i), carrier='parameter') for i in range(7)]
    result = partitions(dict(cases=cases, unresolved=['unknown']), 3)
    assert [c for b in result for c in b['cases']] == cases
    assert [len(b['cases']) for b in result] == [3,3,1]
    assert all(b['unresolved_preserved_in_source_plan'] and not b['numerical_results_read'] for b in result)


@pytest.mark.parametrize('cases,size', [([],2), ([dict(case_id='a')],0), ([dict(case_id='a')]*2,1)])
def test_rejects_invalid_partition(cases,size):
    with pytest.raises(ValueError): partitions(dict(cases=cases),size)
