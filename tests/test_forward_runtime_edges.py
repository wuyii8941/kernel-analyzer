from copy import deepcopy
import pytest
from kernel_analyzer.forward_runtime_edges import observed_edges


def run(source='saved', graph=0):
    return dict(backward_phase=dict(graph_index=graph), backward_inputs=[dict(
        placeholder='input', global_forward_matches=[dict(phase_graph_index=graph,
        source_node=source, identity_mode='EXACT_PYTHON_OBJECT')])])


def test_repeated_identity_and_graph_namespaces():
    result = observed_edges([run(), run(), run(graph=1)])
    assert len(result['edges']) == 2
    assert result['edges'][0]['observations'] == 2
    assert not result['unresolved']


@pytest.mark.parametrize('change', ['missing', 'ambiguous', 'different', 'weak'])
def test_no_selection_from_inconsistent_observations(change):
    second = run()
    matches = second['backward_inputs'][0]['global_forward_matches']
    if change == 'missing': second['backward_inputs'] = []
    if change == 'ambiguous': matches.append(dict(matches[0], source_node='other'))
    if change == 'different': matches[0]['source_node'] = 'other'
    if change == 'weak': matches[0]['identity_mode'] = 'SAME_SHAPE'
    result = observed_edges([run(), second])
    assert not result['edges']
    assert len(result['unresolved']) == 1


def test_duplicate_placeholder_rejected():
    observation = run()
    observation['backward_inputs'] *= 2
    with pytest.raises(ValueError): observed_edges([observation])


def test_input_not_mutated():
    data = [run()]
    before = deepcopy(data)
    observed_edges(data)
    assert data == before
