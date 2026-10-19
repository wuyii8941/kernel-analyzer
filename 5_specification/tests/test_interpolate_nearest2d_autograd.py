"""Independent checks for explicit-size PyTorch 2.10 legacy nearest interpolation."""
from fractions import Fraction
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported

PROFILE = 'torch210_nearest2d_size_autograd'
CHOICE = ('Select PyTorch 2.10 rank-4 functional interpolate with explicit size and legacy nearest coordinates '
          'floor(output_index*input_size/output_size), plus the complete input VJP.')
DTYPES = {torch.float16: 'f16', torch.bfloat16: 'bf16', torch.float32: 'f32', torch.float64: 'f64'}


def record(tensor):
    return {'t': DTYPES[tensor.dtype], 'shape': list(tensor.shape), 'v': tensor.tolist()}


def query(x, upstream, size, **changes):
    result = {'op': 'torch.nn.functional.interpolate', 'x': record(x), 'upstream': record(upstream),
              'size': size, 'scale_factor': None, 'mode': 'nearest', 'align_corners': None,
              'recompute_scale_factor': None, 'antialias': False}
    result.update(changes)
    return result


def independent_reference(x, upstream, size):
    out_h, out_w = (size, size) if isinstance(size, int) else tuple(size)
    height, width = x.shape[-2:]
    out = np.empty(x.shape[:2] + (out_h, out_w), dtype=object)
    grad = np.zeros(x.shape, dtype=object)
    values, weights = x.tolist(), upstream.tolist()
    for n, c, oh, ow in np.ndindex(out.shape):
        ih, iw = oh * height // out_h, ow * width // out_w
        out[n, c, oh, ow] = Fraction(values[n][c][ih][iw])
        grad[n, c, ih, iw] += Fraction(weights[n][c][oh][ow])
    return {'out': out.tolist(), 'grad': grad.tolist()}


class InterpolateNearest2dAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load('INTERP').adapter
        self.x = torch.tensor([[[[1., 2.], [3., 4.]]]])
        self.g = torch.arange(1., 17.).reshape(1, 1, 4, 4)

    def test_hand_witness_accumulates_all_repeated_reads(self):
        fields = self.adapter.reference(PROFILE, query(self.x, self.g, [4, 4])).fields
        self.assertEqual(fields, {'out': [[[[1, 1, 2, 2], [1, 1, 2, 2],
                                           [3, 3, 4, 4], [3, 3, 4, 4]]]],
                                  'grad': [[[[14, 22], [46, 54]]]]})

    def test_independent_integer_coordinates_and_pytorch_210(self):
        for dtype, shape, size in itertools.product(
                DTYPES, [(1, 2, 5, 7), (2, 1, 4, 3)], [(8, 11), (3, 4), (5, 10), 2]):
            with self.subTest(dtype=dtype, shape=shape, size=size):
                x = ((torch.arange(int(np.prod(shape))).reshape(shape) - 11) / 8).to(dtype).requires_grad_()
                out = F.interpolate(x, size=size, mode='nearest')
                upstream = ((torch.arange(out.numel()).reshape(out.shape) % 7 - 3) / 4).to(dtype)
                dx, = torch.autograd.grad(out, x, upstream)
                expected = independent_reference(x.detach(), upstream, size)
                actual = self.adapter.reference(PROFILE, query(x.detach(), upstream, size)).fields
                self.assertEqual(actual, expected)
                self.assertEqual(out.detach().tolist(), np.vectorize(float)(np.array(expected['out'], dtype=object)).tolist())
                self.assertEqual(dx.tolist(), np.vectorize(float)(np.array(expected['grad'], dtype=object)).tolist())

    def test_registry_requires_versioned_coordinate_choice(self):
        q = query(self.x, self.g, [4, 4])
        self.assertEqual(registry.query('INTERP', PROFILE, q).outcome, 'not_established')
        admitted = registry.query('INTERP', PROFILE, q, task_bindings={PROFILE: CHOICE})
        self.assertEqual(admitted.outcome, 'value', admitted.note)
        self.assertEqual(set(admitted.fields), {'out', 'grad'})

    def test_other_tasks_and_malformed_inputs_are_refused(self):
        valid = query(self.x, self.x, [2, 2])
        cases = [
            ({**valid, 'mode': 'nearest-exact'}, Unsupported),
            ({**valid, 'size': None, 'scale_factor': 1}, Unsupported),
            ({**valid, 'align_corners': False}, Invalid),
            ({**valid, 'recompute_scale_factor': True}, Invalid),
            ({**valid, 'antialias': True}, Invalid),
            ({**valid, 'size': [2]}, Invalid),
            ({**valid, 'size': [2, 0]}, Invalid),
            ({**valid, 'upstream': record(torch.ones(1, 1, 1, 1))}, Invalid),
            ({k: value for k, value in valid.items() if k != 'mode'}, Invalid),
        ]
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_checker_requires_both_physical_tensors(self):
        q = query(self.x, self.g, [4, 4])
        valid = {'out': record(F.interpolate(self.x, size=(4, 4), mode='nearest')),
                 'grad': record(torch.tensor([[[[14., 22.], [46., 54.]]]]))}
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {'out': valid['out']}))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {**valid, 'grad': record(torch.zeros_like(self.x))}))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {**valid, 'out': {**valid['out'], 't': 'f64'}}))


if __name__ == '__main__':
    unittest.main()
