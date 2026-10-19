"""Real public normalization parameters must select the same complete mathematical task."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '2_tool/src'), str(ROOT / '5_specification/corpus/external_batch')]
import run_norm as runner
from kernel_spec import registry


class NormPublicCallMapping(unittest.TestCase):
    def test_rms_none_epsilon_is_input_dtype_default_without_mutating_call(self):
        operands = [torch.tensor([[2., 4.]]), None, None]
        options = {'kind': 'rms_norm', 'eps': None, 'normalized_shape': [2]}
        q = runner.mathematical_query(operands, torch.ones(1, 2), options)
        self.assertEqual(q['eps'], 2 ** -23)
        self.assertIsNone(options['eps'])
        self.assertNotIn('bias', q)
        self.assertIsNone(q['weight'])

    def test_complete_public_vjp_matches_exact_rms_witness(self):
        operands = [torch.tensor([[2., 4.]]), torch.tensor([1., 3.]), None]
        g = torch.tensor([[1., -1.]])
        options = {'kind': 'rms_norm', 'eps': 6., 'normalized_shape': [2]}
        result = registry.query('NORM', 'torch_autograd', runner.mathematical_query(operands, g, options))
        self.assertEqual(result.outcome, 'value', result.note)
        expected = runner.complete_reference(result.fields, operands)
        with patch.object(torch.cuda, 'synchronize'):
            actual, _, _ = runner.observation.execute(runner.functional, operands, g, options)
        comparison = runner.observation.compare(actual, actual, expected)
        for item in comparison['candidate_math'].values():
            self.assertEqual(item['coordinates'], item['exact_equal'])
        self.assertIsNone(expected['g2'])

    def test_missing_or_extra_gradient_is_refused(self):
        operands = [torch.ones(1, 2), None, torch.ones(2)]
        valid = {'y': [[1, 1]], 'grad': {'x': [[1, 1]], 'bias': [1, 1]}}
        self.assertIsNone(runner.complete_reference(valid, operands)['g1'])
        for grad in ({'x': [[1, 1]]}, {**valid['grad'], 'weight': [1, 1]}):
            with self.subTest(grad=grad), self.assertRaises(ValueError):
                runner.complete_reference({'y': valid['y'], 'grad': grad}, operands)

    def test_unsupported_physical_inputs_are_not_silently_reinterpreted(self):
        options = {'kind': 'rms_norm', 'eps': None, 'normalized_shape': [2]}
        for x in (torch.ones(1, 2, dtype=torch.float64), torch.empty(0, 2), torch.full((1, 2), float('nan'))):
            with self.subTest(shape=tuple(x.shape), dtype=x.dtype), self.assertRaises(ValueError):
                runner.mathematical_query([x, None, None], torch.ones_like(x), options)
