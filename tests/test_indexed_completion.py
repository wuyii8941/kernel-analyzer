import json
import pytest
from scripts.finalize_indexed_accumulation import verify_runtime


def test_missing_runtime_evidence_is_not_completion(tmp_path):
    raw=tmp_path/'raw'; raw.mkdir()
    (raw/'family_execution_protocol.json').write_text(json.dumps({'contracts':{'task':{}}}))
    with pytest.raises(FileNotFoundError): verify_runtime(tmp_path)
    (raw/'indexed_runtime_bindings.json').write_text(json.dumps({'observations':[]}))
    with pytest.raises(ValueError,match='Missing'): verify_runtime(tmp_path)
    (raw/'indexed_runtime_bindings.json').write_text(json.dumps({'observations':[{}]}))
    with pytest.raises(ValueError,match='case set'): verify_runtime(tmp_path)
