import json

import pytest

from scripts import audit_numerical_execution_manifest as module


def test_pending_queue_keeps_full_denominator(tmp_path, monkeypatch):
    queue = tmp_path / 'not_started'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'families': [dict(
        name='example', plan=str(tmp_path / 'plan.json'), queues=[str(queue)])]}))
    def fake_audit(plan, roots, queues):
        assert queues == []
        return dict(eligible_positions=28, counts={'VERIFIED': 4, 'NOT_SCHEDULED': 24},
                    eligible_measurement_complete=False)
    monkeypatch.setattr(module, 'audit', fake_audit)
    result = module.build(manifest)
    assert result['eligible_positions'] == 28
    assert result['verified_positions'] == 4
    assert result['families'][0]['pending_queue_metadata'] == [str(queue)]
    assert not result['selected_plans_complete']


def test_complete_selected_plans_not_whole_research(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'families': [dict(name='one', plan='plan.json')]}))
    monkeypatch.setattr(module, 'audit', lambda *a: dict(
        eligible_positions=1, counts={'VERIFIED': 1}, eligible_measurement_complete=True))
    result = module.build(manifest)
    assert result['selected_plans_complete']
    assert not result['whole_research_plan_complete']
    assert not result['all_kernel_support_established']


def test_duplicate_family_rejected(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'families': [dict(name='same'), dict(name='same')]}))
    with pytest.raises(ValueError, match='Duplicate'):
        module.build(manifest)
