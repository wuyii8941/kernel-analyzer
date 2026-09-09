import json
import pytest
from scripts.summarize_training_quality_cost import summarize


def make_pair(root):
    (root / 'plan.json').write_text(json.dumps({'pairs': 1, 'steps': 32, 'scope': 'test'}))
    for condition, seconds in [('candidate', 10.), ('reference', 20.)]:
        path = root / 'pair0' / condition
        path.mkdir(parents=True)
        (path / 'status.json').write_text(json.dumps({
            'status': 'COMPLETE_DEVELOPMENT_PILOT', 'initial_parameters_sha256': 'same',
            'evaluations': [{'step': 32, 'shared_evaluation_loss': 1.}],
            'elapsed_seconds_including_evaluation': seconds, 'peak_allocated_bytes': 100,
        }))


def test_recorded_cost_is_not_throughput_claim(tmp_path):
    make_pair(tmp_path)
    result = summarize(tmp_path)
    assert result['median_recorded_time_ratio'] == .5
    assert not result['isolated_throughput_claim']
    assert not result['new_quality_test_performed']


@pytest.mark.parametrize('field,value', [('status', 'COMPLETE_WITH_FAILURE'),
                                       ('initial_parameters_sha256', 'different'),
                                       ('evaluations', []),
                                       ('elapsed_seconds_including_evaluation', 0)])
def test_invalid_pair_is_not_silently_omitted(tmp_path, field, value):
    make_pair(tmp_path)
    path = tmp_path / 'pair0/reference/status.json'
    data = json.loads(path.read_text()); data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize(tmp_path)
