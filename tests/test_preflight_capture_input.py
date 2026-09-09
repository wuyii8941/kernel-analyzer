import hashlib
import json
import pytest
from scripts.preflight_capture_input import check, check_command


def fixture(tmp_path, tokens=None):
    values = [[1, 2], [3, 4]] if tokens is None else tokens
    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'states': [{'input_ids': x} for x in values]}))
    capture = tmp_path / 'capture.json'
    capture.write_text(json.dumps({'input': {
        'input_bank_sha256': hashlib.sha256(bank.read_bytes()).hexdigest(),
        'token_ids_sha256': hashlib.sha256(b'[1,2]').hexdigest(),
        'sequence_length': 2}}))
    return bank, capture


def test_matching_bank(tmp_path):
    assert check(*fixture(tmp_path), 2)['states'] == 2


def test_wrong_bank(tmp_path):
    bank, capture = fixture(tmp_path)
    bank.write_text(bank.read_text() + ' ')
    with pytest.raises(ValueError, match='release-bound'):
        check(bank, capture, 2)


@pytest.mark.parametrize('values,message', [([[9, 2], [3, 4]], 'anchor'),
                                         ([[1, 2], [3]], 'shape')])
def test_wrong_tokens(tmp_path, values, message):
    with pytest.raises(ValueError, match=message):
        check(*fixture(tmp_path, values), 2)


def test_insufficient_states(tmp_path):
    with pytest.raises(ValueError, match='Insufficient'):
        check(*fixture(tmp_path), 3)


def test_command_and_duplicate_flags(tmp_path):
    bank, capture = fixture(tmp_path)
    cmd = ['python', 'scripts/run_residual_rms_forward_capture.py',
           '--input-bank=' + str(bank), '--release-dir', str(tmp_path), '--states', '2']
    assert check_command(cmd)['states'] == 2
    with pytest.raises(ValueError, match='Unique'):
        check_command(cmd + ['--states=3'])
    with pytest.raises(ValueError, match='Separate state bank'):
        check_command(cmd + ['--state-bank=' + str(bank)])
    assert check_command(['python', 'different_capture.py']) is None


def test_grouped_softmax_capture_uses_same_input_preflight(tmp_path):
    bank, _capture = fixture(tmp_path)
    cmd = ['python', 'scripts/run_grouped_causal_softmax_capture.py',
           '--input-bank', str(bank), '--release-dir', str(tmp_path), '--states=2']
    assert check_command(cmd)['states'] == 2
