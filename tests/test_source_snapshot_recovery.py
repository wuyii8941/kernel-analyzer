import hashlib
import json

import pytest

from scripts.recover_source_snapshot_by_hash import recover


def test_recovers_only_exact_preexisting_content(tmp_path):
    source_name = '/data1/tzh/kernel-analyzer/src/example.py'
    text = 'value = 1\n'
    digest = hashlib.sha256(text.encode()).hexdigest()
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'source_sha256': {source_name: digest}}))
    prior = tmp_path / 'prior/source_snapshot.json'
    prior.parent.mkdir()
    prior.write_text(json.dumps({'some/old/path.py': text}))
    output = tmp_path / 'new/source_snapshot.json'
    provenance = tmp_path / 'new/recovery.json'
    report = recover(protocol, tmp_path, output, provenance)
    assert report['status'] == 'RECOVERED_EXACT_PROTOCOL_PINNED_SOURCE_TEXT'
    assert json.loads(output.read_text()) == {source_name: text}
    assert json.loads(provenance.read_text())['sources'][0]['required_sha256'] == digest


def test_missing_content_digest_is_not_reconstructed(tmp_path):
    source_name = '/data1/tzh/kernel-analyzer/src/example.py'
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'source_sha256': {source_name: '0' * 64}}))
    with pytest.raises(ValueError, match='no current or prior snapshot'):
        recover(protocol, tmp_path, tmp_path / 'snapshot.json', tmp_path / 'recovery.json')


def test_uses_current_file_only_when_it_still_matches(tmp_path):
    source = tmp_path / 'current.py'
    source.write_text('still frozen\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'source_sha256': {str(source): digest}}))
    output = tmp_path / 'snapshot.json'
    provenance = tmp_path / 'recovery.json'
    report = recover(protocol, tmp_path, output, provenance)
    assert report['sources'][0]['recovery_method'] == 'CURRENT_FILE_STILL_MATCHES_PROTOCOL_DIGEST'
    source.write_text('changed\n')
    assert json.loads(output.read_text())[str(source)] == 'still frozen\n'
