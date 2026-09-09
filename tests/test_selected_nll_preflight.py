import pytest
from scripts.run_selected_nll_capture import preflight_state_bank


def test_legacy_sequence_ids_preserved_without_mutating_bank():
    bank = dict(states=[dict(sequence_id='a'), dict(sequence_id='b')])
    preflight_state_bank(bank, ['--states', '2'])
    assert 'state_id' not in bank['states'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        preflight_state_bank(dict(states=[dict(sequence_id='a'), dict(state_id='a')]), ['--states', '2'])
