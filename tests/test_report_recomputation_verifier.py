import json
import sys
import pytest
from scripts import verify_training_numerical_reports_v2 as verifier


@pytest.mark.parametrize('tampered', [False, True])
def test_recompute_rejects_cached_decision_changes(tmp_path, monkeypatch, tampered):
    monkeypatch.setattr(verifier, 'BASE', tmp_path)
    monkeypatch.setattr(verifier, 'hashes', lambda: {'fixed': 'hash'})
    raw = tmp_path / 'raw.json'
    raw.write_text('{}')
    (tmp_path / 'protocol.json').write_text(json.dumps({'source_sha256': {'fixed': 'hash'}}))
    result = dict(case_id='x', contrast_id='comparison', measurement_status='VALID',
                  claim_scope='FIXED_SUITE_UPDATE', bias_analysis={'q': 1.},
                  equivalence_decision='NON_EQUIVALENT', mandatory_endpoints=['Q'])
    monkeypatch.setattr(verifier, 'analyze_artifact', lambda raw, protocol: result)
    saved = dict(result, provenance={'raw_sha256': verifier.file_sha256(raw)})
    if tampered:
        saved['equivalence_decision'] = 'EQUIVALENT'
    (tmp_path / 'recomputed.json').write_text(json.dumps(saved))
    output = tmp_path / 'verification.json'
    monkeypatch.setattr(sys, 'argv', ['verify', '--output', str(output)])
    if tampered:
        with pytest.raises(RuntimeError, match='Recomputation differs'):
            verifier.main()
        assert not output.exists()
    else:
        verifier.main()
        assert json.loads(output.read_text())['report_count'] == 1
