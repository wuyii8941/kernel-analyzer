from copy import deepcopy
import pytest
from kernel_analyzer.aot_structure_identity import compare


def fixture():
    return dict(graphs=[dict(phase='FORWARD', graph_index=0, nodes=[dict(
        name='x', op='call_function', target='aten.add', arguments={'x': 1},
        input_edges=[], tensor_meta=['fp32'])])])


def test_identical_structure_not_runtime_proof():
    result = compare(fixture(), fixture())
    assert result['structure_identical']
    assert not result['runtime_values_or_execution_identity_proved']


@pytest.mark.parametrize('field,value', [('target','aten.mul'), ('arguments',{}),
    ('input_edges',[dict(source_node='other')]), ('tensor_meta',['bf16'])])
def test_same_names_not_enough(field, value):
    changed = deepcopy(fixture())
    changed['graphs'][0]['nodes'][0][field] = value
    assert not compare(fixture(), changed)['structure_identical']


def test_missing_records_not_identity():
    data = fixture()
    del data['graphs'][0]['nodes'][0]['input_edges']
    assert not compare(data, data)['structure_identical']
