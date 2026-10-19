"""Public-call correspondence checks independent of the real Triton candidates."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

DIRECTORY = Path(__file__).resolve().parents[1] / 'corpus/external_batch'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('external_conv', DIRECTORY / 'run_conv.py')
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)
CASES = json.loads((DIRECTORY / 'selection_conv_20261019_v1.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_public_forward_all_gradients_match_exact_reference(case):
    operands, upstream, options = conv.inputs(case, [1, 4, 6], 20261230, 'cpu')
    info, expected = conv.mathematical_reference(operands, upstream, options)
    assert [r['outcome'] for r in info['results']] == ['value', 'value']
    actual, capture = conv.execute(conv.functional, operands, upstream, options)
    comparison = conv.matrix.compare(actual, actual, expected)
    assert set(actual) == {'out', 'g0', 'g1', 'g2'}
    assert comparison['absent_gradients'] == ([] if case.get('bias', True) else ['g2'])
    # Small dyadic values have exactly representable products and sums here.
    for result in comparison['standard_math'].values():
        assert result['exact_equal'] == result['coordinates']
    assert capture['triton_scope'] == 'not_established'


def test_same_padding_uses_floor_left_and_crops_gradient_not_output():
    x = torch.tensor([[[1., 2., 3.]]])
    w = torch.tensor([[[1., 2.]]])
    g = torch.tensor([[[1., 2., 3.]]])
    info, ref = conv.mathematical_reference([x, w, None], g, {'kind': 'conv1d', 'padding': 'same'})
    assert info['mapping']['explicit_zero_padding'] == [(0, 1)]
    assert info['queries'][0]['args'][0]['v'] == [[[1., 2., 3., 0.]]]
    assert all(conv.batch.reference_comparison(torch.tensor(v), ref[k])['outside_mathematical_enclosure'] == 0
               for k, v in {'out': [[[5., 8., 3.]]], 'g0': [[[1., 4., 7.]]], 'g1': [[[14., 8.]]]}.items())
    assert ref['g2'] is None


def test_missing_gradient_and_wrong_upstream_refused():
    operands, upstream, options = conv.inputs(CASES[0], [1, 4, 6], 0, 'cpu')
    _, expected = conv.mathematical_reference(operands, upstream, options)
    actual, _ = conv.execute(conv.functional, operands, upstream, options)
    with pytest.raises(ValueError, match='missing complete output'):
        conv.matrix.compare({k: v for k, v in actual.items() if k != 'g1'}, actual, expected)
    info, bad = conv.mathematical_reference(operands, upstream[..., :-1], options)
    assert bad is None and info['results'][1]['outcome'] == 'invalid_input'
    with pytest.raises(ValueError, match='same padding requires'):
        conv.mathematical_reference(operands, upstream, {'kind': 'conv1d', 'padding': 'same', 'stride': 2})
