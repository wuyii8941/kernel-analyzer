import hashlib
import pytest
from scripts.scan_selected_nll_pool import scan


def test_source_hash_and_duplicate_pool_are_checked(tmp_path):
    path = tmp_path/'source.py'
    path.write_text('unrelated = 1\n')
    row = dict(source=str(path), symbol='unrelated',
               source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = scan(dict(records=[row]))
    assert result['source_matches'] == 0
    assert result['records'][0]['status'] == 'NOT_SUPPORTED_BY_THIS_REFERENCE'
    with pytest.raises(ValueError, match='Duplicate'):
        scan(dict(records=[row, row]))
    path.write_text('unrelated = 2\n')
    with pytest.raises(ValueError, match='changed'):
        scan(dict(records=[row]))
