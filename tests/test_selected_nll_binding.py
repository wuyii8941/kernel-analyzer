import pytest
from scripts.bind_selected_nll_tasks import bind


def test_release_and_duplicate_mapping_rejected(tmp_path):
    mapping = dict(schema='release-parameter-mappings-v1', release_path=str(tmp_path), cases=[])
    empty = bind(dict(records=[]), mapping, dict(rows=[]), tmp_path)
    assert not empty['runtime_measurement_complete']
    with pytest.raises(ValueError, match='release'):
        bind(dict(records=[]), mapping, dict(rows=[]), tmp_path/'different')
    mapping['cases'] = [dict(task_id='one'), dict(task_id='one')]
    with pytest.raises(ValueError, match='Duplicate'):
        bind(dict(records=[]), mapping, dict(rows=[]), tmp_path)
