from pathlib import Path
import pytest
from scripts import bind_discovered_recurrences as binding


def fixture(monkeypatch):
    monkeypatch.setattr(binding, 'sha', lambda p: 'digest')
    contract = dict(symbol='triton_x', source_sha256='digest',
                    segment_input_kind='FP32_PREVIOUS_STATE')
    scan = dict(schema='continued-recurrence-source-scan-v1', source_sha256={'checker':'digest'},
        sources=[dict(path='/data1/tzh/source.py', sha256='digest', rows=[
            dict(symbol='triton_x', status='SOURCE_CHECKED', contract=contract),
            dict(symbol='triton_y', status='REJECTED', reason='unsupported')])])
    return scan, Path('/data1/tzh/source.py')


def test_all_checked_and_rejected_preserved(monkeypatch):
    scan, path = fixture(monkeypatch)
    checked, rejected = binding.candidates(scan, path)
    assert len(checked)==1 and len(rejected)==1


def test_changed_source_rejected(monkeypatch):
    scan, path = fixture(monkeypatch)
    scan['sources'][0]['sha256']='changed'
    with pytest.raises(ValueError): binding.candidates(scan,path)


def test_changed_checker_rejected(monkeypatch):
    scan, path = fixture(monkeypatch)
    scan['source_sha256']['checker']='changed'
    with pytest.raises(ValueError): binding.candidates(scan,path)


def test_duplicate_discovery_rejected(monkeypatch):
    scan, path = fixture(monkeypatch)
    scan['sources'][0]['rows'].append(scan['sources'][0]['rows'][0])
    with pytest.raises(ValueError): binding.candidates(scan,path)


def test_contract_identity_must_match(monkeypatch):
    scan, path = fixture(monkeypatch)
    scan['sources'][0]['rows'][0]['contract']['symbol']='other'
    with pytest.raises(ValueError): binding.candidates(scan,path)
