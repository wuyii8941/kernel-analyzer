import pytest
from scripts.bind_continued_recurrence import bind


def inputs():
    tasks = [dict(task_id='t', symbol='kernel', formal_pointer='out_ptr1',
                  exact_aot_endpoint_id='e', status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT')]
    mapping = [dict(task_id='t', case_id='c', expected_symbol='kernel',
                    exact_aot_endpoint_id='e', carrier='parameter')]
    contract = dict(symbol='kernel', output_pointers=['out_ptr0', 'out_ptr1'])
    return tasks, mapping, contract


def test_preserves_original_binding():
    tasks, mapping, contract = inputs()
    cases, unresolved = bind(tasks, mapping, contract)
    assert not unresolved
    assert cases[0]['carrier'] == 'parameter'
    assert cases[0]['reference_output_index'] == 1
    assert cases[0]['source_case_id'] == 'c'
    assert cases[0]['runtime_parameter_reach'] == 'NOT_YET_MEASURED'


def test_missing_mapping_retained():
    tasks, _, contract = inputs()
    cases, unresolved = bind(tasks, [], contract)
    assert not cases and len(unresolved) == 1


def test_unknown_output_retained():
    tasks, mapping, contract = inputs()
    tasks[0]['formal_pointer'] = 'out_ptr9'
    cases, unresolved = bind(tasks, mapping, contract)
    assert not cases and len(unresolved) == 1


def test_conflicting_mapping_rejected():
    tasks, mapping, contract = inputs()
    mapping[0]['expected_symbol'] = 'other'
    with pytest.raises(ValueError):
        bind(tasks, mapping, contract)


def test_duplicate_mapping_rejected():
    tasks, mapping, contract = inputs()
    with pytest.raises(ValueError):
        bind(tasks, mapping+mapping, contract)


def test_empty_observed_symbol_rejected():
    with pytest.raises(ValueError):
        bind([], [], dict(symbol='kernel', output_pointers=['out_ptr0']))
