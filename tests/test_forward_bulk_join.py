from pathlib import Path
import pytest
from scripts import join_forward_queue_once as bulk
from scripts.join_coverage_measurements import join


def test_all_captures_audited_and_raw_provenance_retained(monkeypatch):
    tasks = str(Path('/data1/tzh/release/same_dtype_tasks.json.gz').resolve())
    monkeypatch.setattr(bulk, 'sha', lambda path: 'digest')
    monkeypatch.setattr(bulk, 'read', lambda path: {'source_sha256': {tasks: 'digest'}})
    calls = []
    def audit(root):
        calls.append(root)
        return dict(records=[dict(task_id=root.name, raw_path=str(root/'raw.json'),
                                  raw_sha256='raw', status='RECORDED_MEASUREMENT_CHECKED')],
                    counts={'RECORDED_MEASUREMENT_CHECKED': 1})
    monkeypatch.setattr(bulk, 'finalize', audit)
    roots = [Path('/data1/tzh/job0'), Path('/data1/tzh/job1')]
    report, audits = bulk.collect(Path('/data1/tzh/release'), roots)
    assert calls == roots and len(audits) == 2
    assert report['records'][0]['raw_artifact'] == '/data1/tzh/job0/raw.json'
    assert report['records'][1]['status'] == 'RECORDED_MEASUREMENT_CHECKED'
    monkeypatch.setattr(bulk, 'read', lambda path: {'source_sha256': {tasks: 'changed'}})
    with pytest.raises(ValueError):
        bulk.collect(Path('/data1/tzh/release'), roots)


def test_single_merge_matches_repeated_merges_and_rejects_duplicates():
    release = Path('/data1/tzh/release')
    inventory = dict(records=[dict(release=str(release), task_id=str(i),
        runtime_measurement_status='NOT_ASSESSED_BY_THIS_INVENTORY',
        eligibility='UNCHANGED_BINDING') for i in range(3)])
    first = dict(task_id='0', status='RECORDED_MEASUREMENT_CHECKED',
                 raw_artifact='raw0', raw_sha256='hash0')
    second = dict(task_id='1', status='NOT_CAPTURED')
    sequential = join(join(inventory, release, dict(records=[first], errors=[])),
                      release, dict(records=[second], errors=[]))
    combined = join(inventory, release, dict(records=[first, second], errors=[]))
    assert combined == sequential
    assert all(r['eligibility'] == 'UNCHANGED_BINDING' for r in combined['records'])
    with pytest.raises(ValueError, match='duplicate'):
        join(inventory, release, dict(records=[first, first], errors=[]))
