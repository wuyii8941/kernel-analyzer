import hashlib

import pytest

from scripts.scan_registered_reference_pool import checked_sources


def test_pool_preserves_all_sources_and_deduplicates(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('x = 1\n')
    row = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert checked_sources(dict(records=[row, row])) == [str(source)]
    source.write_text('x = 2\n')
    with pytest.raises(ValueError, match='changed'):
        checked_sources(dict(records=[row]))


def test_empty_or_conflicting_pool_rejected():
    with pytest.raises(ValueError, match='Empty'):
        checked_sources(dict(records=[]))
    with pytest.raises(ValueError, match='Conflicting'):
        checked_sources(dict(records=[dict(source='a', source_sha256='1'),
                                      dict(source='a', source_sha256='2')]))
