import pytest
from scripts.summarize_forward_measurements import summarize


def data():
    rows = [{'effect_energy': 0, 'repair_energy': 1, 'effect_repair_inner_product': 0},
            {'effect_energy': 1, 'repair_energy': 1, 'effect_repair_inner_product': 1},
            {'effect_energy': 0, 'repair_energy': 100, 'effect_repair_inner_product': 0}]
    return dict(state_ids=['cal', 'a', 'b'], confirmation_state_ids=['a', 'b'],
                original_coordinate_statistics={k: rows for k in
                    ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE')})


def test_ratio_of_sums_and_no_verdict():
    raw = data()
    raw['optimizer'] = {'moments': 'ZERO_AT_EVERY_INPUT_STATE'}
    raw['carrier'] = 'one.weight'
    result = summarize(raw)
    assert result['stages']['PARAMETER_WRITE']['aligned_ratio_of_sums'] == 1/101
    assert result['equivalence_decision'] == 'NOT_ASSESSED'
    assert result['population_guarantee'] is False
    assert result['optimizer'] == raw['optimizer']
    assert result['parameter_scope'] == 'one.weight'


def test_unknown_confirmation_rejected():
    raw = data()
    raw['confirmation_state_ids'] = ['missing']
    with pytest.raises(ValueError):
        summarize(raw)


def test_invalid_energy_rejected():
    raw = data()
    raw['original_coordinate_statistics']['LOCAL'][1]['effect_energy'] = -1
    with pytest.raises(ValueError):
        summarize(raw)


def test_negative_alignment_does_not_imply_smaller_update_norm():
    raw = data()
    # r=(1,0), c=(0,1): same norm, u=(-1,1), Q=2, beta=-1.
    for rows in raw['original_coordinate_statistics'].values():
        for row in rows:
            row.update(effect_energy=2, repair_energy=1, effect_repair_inner_product=-1)
    stage = summarize(raw)['stages']['PARAMETER_WRITE']
    assert stage['aligned_ratio_of_sums'] == -1
    assert stage['candidate_to_repair_energy_ratio'] == 1


def test_queue_uses_same_record_summary(monkeypatch, tmp_path):
    from scripts import summarize_forward_measurements as module
    from scripts import join_forward_measurements as joins
    root = tmp_path/'job'
    monkeypatch.setattr(joins, 'queue_roots', lambda q: ([root], [{'status':'TERMINAL_SUCCESS_REQUIRES_FRESH_AUDIT'}]))
    visited = []
    def checked_report(path):
        visited.append(path)
        return {'records':[{'summary':summarize(data())}]}
    monkeypatch.setattr(module, 'report', checked_report)
    monkeypatch.setattr(module, 'sha', lambda p:'hash')
    result = module.queue_report(tmp_path)
    assert visited == [root]
    assert result['captures'][0]['report']['records'][0]['summary']['equivalence_decision'] == 'NOT_ASSESSED'
