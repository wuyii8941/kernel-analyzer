"""Complete gradient observation and ignored input state in real matrix calls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '2_tool/src'), str(ROOT / '5_specification/corpus/external_batch')]
import run_matrix as runner


class MatrixObservation(unittest.TestCase):
    def test_absent_bias_keeps_explicit_null_gradient(self):
        with patch.object(torch.cuda, 'synchronize'):
            observed, _, _ = runner.execute(runner.functional,
                [torch.tensor([[2.]]), torch.tensor([[3.]]), None], torch.ones(1, 1), {'kind': 'linear'})
        reference = {'out': [[6]], 'g0': [[3]], 'g1': [[2]], 'g2': None}
        self.assertEqual(runner.compare(observed, observed, reference)['absent_gradients'], ['g2'])
        for bad in ({k: v for k, v in observed.items() if k != 'g2'},
                    {**observed, 'g2': torch.zeros(1)}, {**observed, 'g0': None}):
            with self.subTest(keys=list(bad)), self.assertRaises(ValueError):
                runner.compare(bad, observed, reference)

    def test_ignored_nan_is_not_mistaken_for_input_mutation(self):
        with patch.object(torch.cuda, 'synchronize'):
            observed, _, _ = runner.execute(runner.functional,
                [torch.full((1, 1), float('nan')), torch.tensor([[2.]]), torch.tensor([[3.]])],
                torch.ones(1, 1), {'kind': 'addmm', 'beta': 0.})
        self.assertEqual(observed['out'].item(), 6.)
        self.assertEqual(observed['g0'].item(), 0.)

    def test_ignored_addend_is_still_readonly(self):
        def bad(values, **options):
            with torch.no_grad():
                values[0].zero_()
            return runner.functional(values, **options)
        with patch.object(torch.cuda, 'synchronize'), self.assertRaisesRegex(ValueError, 'input values changed'):
            runner.execute(bad,
                [torch.full((1, 1), float('nan')), torch.tensor([[2.]]), torch.tensor([[3.]])],
                torch.ones(1, 1), {'kind': 'addmm', 'beta': 0.})
