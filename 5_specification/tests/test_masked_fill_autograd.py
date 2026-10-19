"""Independent checks for the PyTorch 2.10 public scalar masked_fill profile."""
from fractions import Fraction
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported

PROFILE = 'torch210_masked_fill_scalar_autograd'
DTYPES = {torch.float16: 'f16', torch.bfloat16: 'bf16',
          torch.float32: 'f32', torch.float64: 'f64'}


def record(tensor):
    dtype = 'b8' if tensor.dtype == torch.bool else DTYPES[tensor.dtype]
    return {'t': dtype, 'shape': list(tensor.shape), 'v': tensor.tolist()}


def query(x, mask, value, upstream, **changes):
    result = {'op': 'torch.Tensor.masked_fill', 'input': record(x), 'mask': record(mask),
              'value': value, 'upstream': record(upstream)}
    result.update(changes)
    return result


def exact_nested(array):
    array = np.asarray(array, dtype=object)
    if array.ndim == 0:
        return Fraction(array.item())
    return [exact_nested(child) for child in array]


def independent_reference(x, mask, value, upstream):
    out_shape = torch.broadcast_shapes(x.shape, mask.shape)
    xb = np.broadcast_to(np.asarray(x.tolist(), dtype=object), out_shape)
    mb = np.broadcast_to(np.asarray(mask.tolist(), dtype=bool), out_shape)
    ub = np.asarray(upstream.tolist(), dtype=object)
    out = np.where(mb, Fraction(value), xb)
    grad = np.empty(x.shape, dtype=object)
    grad.fill(Fraction(0))
    leading = len(out_shape) - x.ndim
    for out_index in np.ndindex(out_shape):
        if mb[out_index]:
            continue
        input_index = tuple(0 if length == 1 else out_index[leading + axis]
                            for axis, length in enumerate(x.shape))
        grad[input_index] += Fraction(ub[out_index])
    return {'out': exact_nested(out), 'grad': exact_nested(grad)}


class MaskedFillAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load('ARITH').adapter

    def test_broadcast_input_hand_witness_sums_gradient_back(self):
        x = torch.tensor([[1., 2., 3.]])
        mask = torch.tensor([[True, False, True], [False, True, False]])
        upstream = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        fields = self.adapter.reference(PROFILE, query(x, mask, -2, upstream)).fields
        self.assertEqual(fields, {'out': [[-2, 2, -2], [1, -2, 3]],
                                  'grad': [[4, 2, 6]]})

    def test_independent_coordinate_formula_and_pytorch_210(self):
        shapes = [((1, 3), (2, 3)), ((2, 3), (2, 1)), ((3,), (1, 3)),
                  ((2, 1, 4), (1, 3, 4))]
        for dtype, (input_shape, mask_shape) in itertools.product(DTYPES, shapes):
            with self.subTest(dtype=dtype, input_shape=input_shape, mask_shape=mask_shape):
                count = int(np.prod(input_shape))
                x = ((torch.arange(count).reshape(input_shape) - 3) / 4).to(dtype).requires_grad_()
                mask_count = int(np.prod(mask_shape))
                mask = (torch.arange(mask_count).reshape(mask_shape) % 3 == 0)
                out = x.masked_fill(mask, -0.5)
                upstream = ((torch.arange(out.numel()).reshape(out.shape) % 5) - 2).to(dtype)
                dx, = torch.autograd.grad(out, x, upstream)
                expected = independent_reference(x.detach(), mask, Fraction(-1, 2), upstream)
                actual = self.adapter.reference(
                    PROFILE, query(x.detach(), mask, '-1/2', upstream)).fields
                self.assertEqual(actual, expected)
                self.assertEqual(out.detach().tolist(),
                                 np.vectorize(float)(np.asarray(expected['out'], dtype=object)).tolist())
                self.assertEqual(dx.tolist(),
                                 np.vectorize(float)(np.asarray(expected['grad'], dtype=object)).tolist())

    def test_public_task_and_complete_physical_inputs_are_required(self):
        x = torch.tensor([1., 2.])
        mask = torch.tensor([True, False])
        upstream = torch.ones(2)
        valid = query(x, mask, 0, upstream)
        cases = [
            ({**valid, 'op': 'torch.Tensor.masked_fill_'}, Unsupported),
            ({**valid, 'value': {'t': 'f32', 'shape': [], 'v': 0}}, Unsupported),
            ({**valid, 'mask': {'t': 'b8', 'v': [True, False]}}, Invalid),
            ({**valid, 'upstream': record(torch.ones(1))}, Invalid),
            ({**valid, 'upstream': record(torch.ones(2, dtype=torch.float64))}, Invalid),
            ({**valid, 'extra': 1}, Unsupported),
        ]
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_checker_requires_forward_and_gradient(self):
        x = torch.tensor([[1., 2., 3.]])
        mask = torch.tensor([[True, False, True], [False, True, False]])
        upstream = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        q = query(x, mask, -2, upstream)
        valid = {'out': record(x.masked_fill(mask, -2)),
                 'grad': record(torch.tensor([[4., 2., 6.]]))}
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {'out': valid['out']}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, 'grad': record(torch.zeros_like(x))}))
        self.assertFalse(self.adapter.check_output(
            PROFILE, q, {**valid, 'out': {**valid['out'], 't': 'f64'}}))


if __name__ == '__main__':
    unittest.main()
