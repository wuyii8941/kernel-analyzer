import pytest
from scripts.report_unbound_kernel_groups import summarize


def row(release='a', task='0', **kw):
    return dict(release=release, task_id=task, implementation_kind='TRITON',
                phase='FORWARD', symbol='k', formal_pointer='out0',
                eligibility='NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT', **kw)


def test_release_qualified_and_outcome_independent():
    rows = [row(), row('b'), row(task='1')]
    before = summarize(rows)
    for r in rows:
        r['runtime_measurement_status'] = 'ANY_LABEL'
        r['bias'] = 1000
    assert summarize(rows) == before
    assert before[0]['positions'] == 3
    assert before[0]['releases'] == ['a', 'b']


def test_duplicate_rejected_and_bound_excluded():
    with pytest.raises(ValueError):
        summarize([row(), row()])
    r = row()
    r['eligibility'] = 'BOUND'
    assert summarize([r]) == []
