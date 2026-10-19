"""The external attention runner must bind the complete public training task."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


DIRECTORY = Path(__file__).resolve().parents[1] / 'corpus/external_batch'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('external_attention', DIRECTORY / 'run_attention.py')
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)
SELECTION = json.loads((DIRECTORY / 'selection_attention_20261019_v1.json').read_text())


@pytest.mark.parametrize('case', SELECTION['development']['cases'], ids=lambda value: value['id'])
def test_development_calls_have_complete_reference_and_match_public_math_backend(case):
    q, k, v, mask, upstream, options = attention.inputs(case, 20261310, 'cpu')
    info, mathematical = attention.mathematical_reference(q, k, v, mask, upstream, options)
    assert info['outcome'] == 'value'
    actual, capture = attention.execute(attention.standard, q, k, v, mask, upstream, options)
    comparison = attention.compare(actual, actual, mathematical)
    assert set(actual) == {'out', 'dq', 'dk', 'dv'}
    assert comparison['compatible']
    assert all(value['coordinates'] == actual[name].numel() and value['nonfinite'] == 0
               for name, value in comparison['standard_math'].items())
    assert capture['triton_scope'] == 'not_established'


def test_candidate_result_cannot_drop_or_add_a_gradient():
    case = SELECTION['development']['cases'][0]
    q, k, v, mask, upstream, options = attention.inputs(case, 1, 'cpu')
    _, mathematical = attention.mathematical_reference(q, k, v, mask, upstream, options)
    actual, _ = attention.execute(attention.standard, q, k, v, mask, upstream, options)
    with pytest.raises(ValueError, match='missing or extra'):
        attention.compare({name: value for name, value in actual.items() if name != 'dk'}, actual, mathematical)
    with pytest.raises(ValueError, match='missing or extra'):
        attention.compare(dict(actual, lse=torch.zeros(1)), actual, mathematical)


def test_float16_public_call_keeps_its_physical_dtype_in_exact_comparison():
    case = {'shape': [1, 1, 1, 3, 3, 16], 'dtype': 'f16', 'is_causal': True, 'scale': 0.5}
    q, k, v, mask, upstream, options = attention.inputs(case, 2, 'cpu')
    _, mathematical = attention.mathematical_reference(q, k, v, mask, upstream, options)
    actual, _ = attention.execute(attention.standard, q, k, v, mask, upstream, options)
    comparison = attention.compare(actual, actual, mathematical, declared_atol=0.01)
    assert all(value.dtype == torch.float16 for value in actual.values())
    assert comparison['compatible']
    assert all(value['coordinates'] == actual[name].numel()
               for name, value in comparison['standard_math'].items())
    assert comparison['candidate_source_tolerance']['all_outputs']
    assert comparison['candidate_source_tolerance']['atol'] == 0.01


def test_tutorial_mapping_refuses_tasks_outside_its_interface():
    class Module:
        @staticmethod
        def attention(q, k, v, causal, scale, warp_specialize):
            return attention.standard(q, k, v, None, is_causal=causal, scale=scale, enable_gqa=False)
    candidate = attention.tutorial_function(Module)
    q = torch.zeros(1, 1, 2, 16)
    with pytest.raises(ValueError, match='mask or GQA'):
        candidate(q, q, q, torch.ones(1, 1, 2, 2, dtype=torch.bool), is_causal=False, scale=None,
                  enable_gqa=False)
    with pytest.raises(ValueError, match='equal q/k/v'):
        candidate(q, q[:, :, :1], q[:, :, :1], None, is_causal=False, scale=None, enable_gqa=False)
