"""A stateful public task must expose the whole updated weight to comparison."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '2_tool/src'), str(ROOT / '5_specification/corpus/external_batch')]
import run_embedding as runner
import run as batch
from run_embedding_state import state_relations


class EmbeddingStateObservation(unittest.TestCase):
    def test_default_call_keeps_the_existing_readonly_result(self):
        w = torch.tensor([[1., 2.], [3., 4.]])
        i = torch.tensor([1, 0])
        with patch.object(torch.cuda, 'synchronize'):
            result, _ = runner.execute(runner.functional, w, i, torch.ones(2, 2),
                                       dict(padding_idx=None, scale_grad_by_freq=False))
        self.assertEqual(set(result), {'out', 'dweight'})
        self.assertTrue(torch.equal(result['out'], w[i]))
        self.assertTrue(torch.equal(result['dweight'], torch.ones_like(w)))

    def test_full_state_is_returned_and_padding_lookup_is_not_zeroed(self):
        w = torch.tensor([[3., 4.], [6., 8.], [0., 2.]])
        i = torch.tensor([0, 0, 2])
        g = torch.ones(3, 2)
        options = dict(padding_idx=0, scale_grad_by_freq=False, max_norm=1., norm_type=2.)
        with patch.object(torch.cuda, 'synchronize'):
            result, _ = runner.execute(runner.functional, w, i, g, options)
        self.assertEqual(set(result), {'out', 'dweight', 'weight_after'})
        self.assertEqual(tuple(result['weight_after'].shape), tuple(w.shape))
        self.assertTrue(torch.equal(result['weight_after'][1], w[1]))
        self.assertTrue(torch.equal(result['out'], result['weight_after'][i]))
        self.assertFalse(torch.equal(result['weight_after'][0], w[0]))
        self.assertTrue(torch.equal(result['dweight'][0], torch.zeros(2)))
        self.assertTrue(torch.equal(result['dweight'][2], torch.ones(2)))
        self.assertTrue(torch.equal(w, torch.tensor([[3., 4.], [6., 8.], [0., 2.]])))

    def test_missing_or_truncated_state_cannot_pass_standard_comparison(self):
        expected = {'out': torch.ones(1, 2), 'dweight': torch.ones(3, 2), 'weight_after': torch.ones(3, 2)}
        for candidate in (
            {k: v for k, v in expected.items() if k != 'weight_after'},
            {**expected, 'weight_after': expected['weight_after'][:1]},
        ):
            with self.subTest(keys=list(candidate)), self.assertRaises(ValueError):
                batch.standard_comparison(candidate, expected)
            with self.assertRaises(ValueError):
                state_relations(candidate, torch.ones(3, 2), torch.tensor([0]))

    def test_no_max_norm_still_refuses_weight_mutation(self):
        def bad(input, weight, **kwargs):
            with torch.no_grad():
                weight.add_(1)
            return torch.nn.functional.embedding(input, weight)
        with patch.object(torch.cuda, 'synchronize'), self.assertRaisesRegex(ValueError, 'read-only'):
            runner.execute(bad, torch.ones(3, 2), torch.tensor([0]), torch.ones(1, 2),
                           dict(padding_idx=None, scale_grad_by_freq=False))

    def test_weight_mutation_permission_does_not_allow_index_changes(self):
        def bad(input, weight, **kwargs):
            input.fill_(0)
            return runner.functional(input, weight, **kwargs)
        with patch.object(torch.cuda, 'synchronize'), self.assertRaisesRegex(ValueError, 'read-only'):
            runner.execute(bad, torch.ones(3, 2), torch.tensor([1]), torch.ones(1, 2),
                           dict(padding_idx=None, scale_grad_by_freq=False, max_norm=1.))
