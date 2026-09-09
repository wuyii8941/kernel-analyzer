import pytest

from scripts.select_registered_reference_cases import select


def case(task, symbol):
    return {'task_id': task, 'reference_contract_symbol': symbol}


def test_selection_prefers_distinct_source_symbols_and_is_order_invariant():
    rows = [case('forward:3:o', 'b'), case('forward:2:o', 'a'),
            case('forward:1:o', 'a')]
    first = select({'cases': rows, 'unresolved': [{}]}, 2)
    second = select({'cases': list(reversed(rows)), 'unresolved': [{}]}, 2)
    assert first == second
    assert [row['task_id'] for row in first['cases']] == ['forward:1:o', 'forward:3:o']
    assert first['numerical_results_read'] is False
    assert first['source_case_count'] == 3


def test_selection_fills_from_repeated_symbol_and_rejects_duplicates():
    rows = [case('forward:1:o', 'a'), case('forward:2:o', 'a')]
    assert len(select({'cases': rows}, 2)['cases']) == 2
    with pytest.raises(ValueError, match='Duplicate'):
        select({'cases': [rows[0], rows[0]]}, 1)
