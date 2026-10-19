"""Reject incomplete pooling observations; retain limits of mathematical references."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '2_tool/src'), str(ROOT / '5_specification/corpus/external_batch')]
import run_pool as runner


class PoolPublicCallMapping(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[[[1., 9., 3.], [4., 5., 6.], [7., 8., 2.]]]])
        self.upstream = torch.tensor([[[[1., 2.], [3., 4.]]]])
        self.options = {'kind': 'max', 'kernel_size': 2, 'stride': 1}

    def test_full_indices_and_overlapping_gradient_hand_witness(self):
        actual, launches, _ = runner.execute(runner.functional, self.x, self.upstream, self.options)
        self.assertEqual(actual['out'].tolist(), [[[[9., 9.], [8., 8.]]]])
        self.assertEqual(actual['indices'].tolist(), [[[[1, 1], [7, 7]]]])
        self.assertEqual(actual['dx'].tolist(), [[[[0., 3., 0.], [0., 0., 0.], [0., 7., 0.]]]])
        self.assertEqual(launches, [])
        self.assertTrue(runner.compare(actual, actual, 'max')['compatible'])
        info, expected = runner.mathematical_reference(self.x, self.upstream, self.options)
        self.assertEqual(info['outcome'], 'value', info['note'])
        self.assertEqual(set(expected), {'out', 'indices', 'dx'})
        self.assertEqual(expected['indices'], [[[[1, 1], [7, 7]]]])
        self.assertEqual(runner.mathematical_comparison(actual['out'], expected['out'], 'out')['exact_equal'], 4)
        self.assertTrue(runner.mathematical_comparison(actual['indices'], expected['indices'], 'indices')['exact_equal'])
        self.assertEqual(runner.mathematical_comparison(actual['dx'], expected['dx'], 'dx')['exact_equal'], 9)

    def test_missing_outputs_and_wrong_index_dtype_are_rejected(self):
        actual, _, _ = runner.execute(runner.functional, self.x, self.upstream, self.options)
        for name in actual:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                runner.compare({k: v for k, v in actual.items() if k != name}, actual, 'max')
        with self.assertRaises(ValueError):
            runner.compare({**actual, 'indices': actual['indices'].float()}, actual, 'max')

    def test_indices_and_max_values_cannot_pass_by_float_tolerance(self):
        standard = {'out': torch.ones(1), 'dx': torch.ones(1), 'indices': torch.tensor([1000000])}
        changed = {**standard, 'indices': standard['indices'] + 1}
        self.assertFalse(runner.compare(changed, standard, 'max')['compatible'])
        changed = {**standard, 'out': torch.nextafter(standard['out'], torch.full((1,), 2.))}
        self.assertFalse(runner.compare(changed, standard, 'max')['compatible'])

    def test_unavailable_math_does_not_invent_complete_task_answer(self):
        for options in ({'kind': 'avg', 'kernel_size': 2}, {**self.options, 'padding': 1}):
            with self.subTest(options=options):
                info, expected = runner.mathematical_values(self.x, options)
                self.assertEqual(info['outcome'], 'unsupported')
                self.assertIsNone(expected)

        tied = torch.tensor([[[[5., 5.], [1., 0.]]]])
        info, expected = runner.mathematical_reference(tied, torch.ones(1, 1, 1, 1),
                                                       {'kind': 'max', 'kernel_size': 2})
        self.assertEqual(info['outcome'], 'unsupported')
        self.assertIsNone(expected)

    def test_average_divisor_override_is_part_of_forward_and_gradient(self):
        actual, _, _ = runner.execute(runner.functional, torch.ones(1, 1, 2, 2), torch.ones(1, 1, 1, 1),
                                     {'kind': 'avg', 'kernel_size': 2, 'divisor_override': 2})
        self.assertEqual(actual['out'].item(), 2.)
        self.assertTrue(torch.equal(actual['dx'], torch.full((1, 1, 2, 2), .5)))
        self.assertEqual(set(actual), {'out', 'dx'})

    def test_full_average_reference_binds_actual_upstream_and_public_defaults(self):
        x, g = torch.ones(1, 1, 2, 2), torch.full((1, 1, 1, 1), 2.)
        info, expected = runner.mathematical_reference(x, g, {'kind': 'avg', 'kernel_size': 2})
        self.assertEqual(info['outcome'], 'value', info['note'])
        self.assertEqual(info['query']['upstream'], runner.batch.record(g))
        self.assertTrue(info['task_choice'])
        self.assertEqual(set(expected), {'out', 'dx'})
        self.assertEqual(expected['out'], [[[[1]]]])
        self.assertEqual(expected['dx'], [[[[.5, .5], [.5, .5]]]])

    def test_public_call_preserves_noncontiguous_input_and_declared_rank(self):
        for unbatched in (False, True):
            case = {'kind': 'max', 'options': {'kernel_size': 2}, 'noncontiguous': True, 'unbatched': unbatched}
            x, upstream, options = runner.inputs(case, [2, 3, 6, 8], 91, device='cpu')
            actual, _, strides = runner.execute(runner.functional, x, upstream, options)
            self.assertEqual(strides, list(x.stride()))
            self.assertEqual(actual['dx'].shape, x.shape)
            self.assertEqual(x.ndim, 3 if unbatched else 4)
