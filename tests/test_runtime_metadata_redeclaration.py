import hashlib
import json
import pytest
from scripts.redeclare_runtime_metadata import checked_metadata


def setup(root):
    tokens = [1, 2, 3]
    token_hash = hashlib.sha256(b'[1,2,3]').hexdigest()
    aot = root / 'aot.json'; bank = root / 'bank.json'
    aot.write_text(json.dumps({'input': {'sequence_length': 3, 'token_ids_sha256': token_hash},
                               'capture': {'graphs': [{'phase': 'FORWARD'}, {'phase': 'BACKWARD'}]}}))
    bank.write_text(json.dumps({'states': [{'input_ids': tokens}]}))
    for index, phase in enumerate(('forward', 'backward')):
        path = root / 'trace' / f'model__{index}_{phase}_segment0_executed'
        path.mkdir(parents=True); (path / 'output_code.py').write_text('# recorded source')
    return aot, bank


def test_new_declaration_is_not_original_metadata_or_execution_proof(tmp_path):
    aot, bank = setup(tmp_path)
    result = checked_metadata(tmp_path, aot, bank)
    assert [m['phase'] for m in result['modules']] == ['FORWARD', 'BACKWARD']
    assert not result['original_metadata_recovered']
    assert not result['runtime_identity_verified']


def test_changed_bank_or_phase_order_rejected(tmp_path):
    aot, bank = setup(tmp_path)
    bank.write_text(json.dumps({'states': [{'input_ids': [3, 2, 1]}]}))
    with pytest.raises(ValueError): checked_metadata(tmp_path, aot, bank)
    bank.write_text(json.dumps({'states': [{'input_ids': [1, 2, 3]}]}))
    data = json.loads(aot.read_text()); data['capture']['graphs'].reverse()
    aot.write_text(json.dumps(data))
    with pytest.raises(ValueError): checked_metadata(tmp_path, aot, bank)
