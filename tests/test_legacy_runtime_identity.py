from copy import deepcopy
import pytest
from kernel_analyzer.legacy_runtime_identity import normalize_capture
from kernel_analyzer.forward_runtime_edges import observed_edges


def fixture():
    return dict(cross_phase_runtime_bridge=dict(runs=[dict(
        forward_phase=dict(graph_index=0), backward_phase=dict(graph_index=0),
        forward_outputs=[dict(runtime_token='opaque-token', source_node='saved')],
        backward_inputs=[dict(placeholder='b', forward_output_matches=[dict(
            runtime_token='opaque-token', identity_mode='EXACT_PYTHON_OBJECT')])])]))


def test_record_lookup_without_name_inference_or_mutation():
    data = fixture()
    before = deepcopy(data)
    normalized = normalize_capture(data)
    assert data == before
    edges = observed_edges(normalized['cross_phase_runtime_bridge']['runs'])['edges']
    assert edges[0]['source_node'] == 'saved'


def test_missing_token_rejected():
    data = fixture()
    data['cross_phase_runtime_bridge']['runs'][0]['forward_outputs'] = []
    with pytest.raises(ValueError): normalize_capture(data)


def test_new_format_not_overridden():
    data = fixture()
    data['cross_phase_runtime_bridge']['runs'][0]['backward_inputs'][0]['global_forward_matches'] = []
    normalized = normalize_capture(data)
    assert not observed_edges(normalized['cross_phase_runtime_bridge']['runs'])['edges']
