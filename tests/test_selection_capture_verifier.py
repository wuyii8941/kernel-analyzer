import copy
import hashlib
import json
import pytest
import torch
from scripts.verify_selection_capture import check_outcomes, same_recomputed_value, verify


def records():
    row = dict(scores=torch.ones(1, 3), values=torch.ones(1, 2),
               indices=torch.tensor([[0, 1]]), gradient=torch.ones(2),
               write=torch.ones(2), loss=torch.tensor(1.))
    return [copy.deepcopy(row) for _ in range(4)]


def test_recomputation_comparison_allows_only_last_bit_numeric_drift():
    assert same_recomputed_value({'x': 1.0}, {'x': 1.0 + 5e-15})
    assert not same_recomputed_value({'x': 1.0}, {'x': 1.0 + 2e-14})
    assert not same_recomputed_value({'x': 1.0}, {'y': 1.0})


def test_permutation_is_not_changed_expert_set():
    rows = records()
    for i in (1, 3): rows[i]['indices'] = torch.tensor([[1, 0]])
    result = check_outcomes(rows)
    assert result['changed_index_coordinates'] == 2
    assert result['changed_expert_sets'] == 0
    assert result['repeat_determinism']


def test_legal_tie_can_change_expert_set():
    rows = records()
    for i in (1, 3): rows[i]['indices'] = torch.tensor([[1, 2]])
    assert check_outcomes(rows)['changed_expert_sets'] == 1


def test_repeat_failure_and_invalid_selection():
    rows = records()
    rows[2]['gradient'][0] = 2
    assert check_outcomes(rows)['status'] == 'INVALID_COMPARISON'
    rows[0]['indices'][0, 1] = 0
    with pytest.raises(ValueError, match='repeat'):
        check_outcomes(rows)


def test_saved_capture_recomputed_and_tampering_rejected(tmp_path):
    rows = records()
    raw = tmp_path/'state_000.pt'
    torch.save(rows, raw)
    record = check_outcomes(rows)
    record.pop('changed_expert_sets')
    record.update(state_id='fixed-0', raw_path=str(raw),
                  raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    protocol = dict(schema='granite-selection-probe-v1', states=1,
                    state_scope='FIXED_CHECKPOINT_TEXT_SUITE_COLD_OPTIMIZER')
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'state_000.json').write_text(json.dumps(record))
    (tmp_path/'summary.json').write_text(json.dumps(dict(protocol=protocol, records=[record])))
    assert verify(tmp_path)['all_comparisons_valid']
    rows[0]['write'][0] = 7
    torch.save(rows, raw)
    with pytest.raises(ValueError, match='hash'):
        verify(tmp_path)
