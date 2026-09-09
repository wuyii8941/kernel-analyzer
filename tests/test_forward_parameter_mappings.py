import pytest
from scripts.build_forward_parameter_mappings import map_endpoints


def fixture():
    graphs = [dict(phase='FORWARD', graph_index=0, nodes=[
        dict(name='p', op='placeholder'),
        dict(name='saved', op='call_function', input_edges=[dict(source_node='p')])]),
        dict(phase='BACKWARD', graph_index=0, nodes=[
        dict(name='b', op='placeholder'),
        dict(name='grad', op='call_function', input_edges=[dict(source_node='b')]),
        dict(name='output', op='output', input_edges=[dict(source_node='grad', argument_path=[0])])])]
    runs = [dict(backward_phase=dict(graph_index=0), backward_inputs=[dict(
        placeholder='b', global_forward_matches=[dict(phase_graph_index=0,
        source_node='saved', identity_mode='EXACT_PYTHON_OBJECT')])])]
    binding = dict(name='weight', aliases=['weight'], shape=[1])
    return dict(graphs=graphs, cross_phase_runtime_bridge=dict(runs=runs)), {'p': binding}


def test_saved_path_reuses_backward_output_binding():
    capture, bindings = fixture()
    result = map_endpoints(capture, bindings, {'forward:graph0:saved'})
    parameter = result['rows'][0]['parameters'][0]
    assert parameter['name'] == 'weight'
    assert parameter['aot_distance'] == 2
    assert parameter['backward_entry'] == 'backward:graph0:b'
    assert not result['runtime_measurement_complete']


def test_missing_identity_not_replaced_by_name_guess():
    capture, bindings = fixture()
    capture['cross_phase_runtime_bridge']['runs'] = []
    assert not map_endpoints(capture, bindings, {'forward:graph0:saved'})['rows'][0]['parameters']


def test_unbound_parameter_preserved_as_unresolved():
    capture, _ = fixture()
    assert map_endpoints(capture, {}, {'forward:graph0:saved'})['rows'][0]['status'] == 'UNRESOLVED_PARAMETER_PATH'


def test_multiple_segments_not_silently_paired():
    capture, bindings = fixture()
    capture['graphs'][1]['graph_index'] = 1
    with pytest.raises(ValueError): map_endpoints(capture, bindings, {'forward:graph0:saved'})
