"""Independent checks for the selected complete unique-max max_pool2d task."""
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
from kernel_spec.real import R

PROFILE = 'torch210_unique_max2d_autograd'
CHOICE = ('Select PyTorch 2.10 functional max_pool2d with return_indices=true on unique-max windows, '
          'using row-major spatial indices h*W+w and the complete input VJP.')
DTYPES = {torch.float16: 'f16', torch.bfloat16: 'bf16', torch.float32: 'f32', torch.float64: 'f64'}


def record(tensor):
    return {'t': DTYPES[tensor.dtype], 'shape': list(tensor.shape), 'v': tensor.tolist()}


def query(x, upstream, **options):
    return {'op': 'torch.nn.functional.max_pool2d', 'x': record(x), 'upstream': record(upstream),
            'kernel_size': 2, 'stride': None, 'padding': 0, 'dilation': 1,
            'ceil_mode': False, 'return_indices': True, **options}


def pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def output_length(length, kernel, stride, padding, dilation, ceil):
    numerator = length + 2 * padding - dilation * (kernel - 1) - 1
    result = ((numerator + stride - 1) // stride if ceil else numerator // stride) + 1
    if ceil and (result - 1) * stride >= length + padding:
        result -= 1
    return result


def independent_coordinate_reference(x, upstream, kernel, stride, padding, dilation, ceil):
    """Enumerate coordinates directly, independently of the production adapter."""
    kernel, padding, dilation = pair(kernel), pair(padding), pair(dilation)
    stride = kernel if stride is None else pair(stride)
    height, width = x.shape[-2:]
    out_shape = x.shape[:-2] + (output_length(height, kernel[0], stride[0], padding[0], dilation[0], ceil),
                                output_length(width, kernel[1], stride[1], padding[1], dilation[1], ceil))
    if tuple(upstream.shape) != tuple(out_shape):
        raise ValueError('independent test received the wrong upstream shape')
    out = np.empty(out_shape, dtype=object)
    indices = np.empty(out_shape, dtype=object)
    grad = np.zeros(x.shape, dtype=object)
    values = x.tolist()
    weights = upstream.tolist()
    for lead in np.ndindex(x.shape[:-2]):
        plane = values
        weight_plane = weights
        grad_plane = grad
        for index in lead:
            plane = plane[index]
            weight_plane = weight_plane[index]
            grad_plane = grad_plane[index]
        for oh, ow in itertools.product(range(out_shape[-2]), range(out_shape[-1])):
            samples = []
            for kh, kw in itertools.product(range(kernel[0]), range(kernel[1])):
                ih = oh * stride[0] - padding[0] + kh * dilation[0]
                iw = ow * stride[1] - padding[1] + kw * dilation[1]
                if 0 <= ih < height and 0 <= iw < width:
                    samples.append((Fraction(plane[ih][iw]), ih, iw))
            best = max(value for value, _, _ in samples)
            selected = [(ih, iw) for value, ih, iw in samples if value == best]
            if len(selected) != 1:
                raise ValueError('test data did not have a unique window maximum')
            ih, iw = selected[0]
            pos = lead + (oh, ow)
            out[pos], indices[pos] = best, ih * width + iw
            grad_plane[ih, iw] += Fraction(weight_plane[oh][ow])
    return {'out': out.tolist(), 'indices': indices.tolist(), 'grad': grad.tolist()}


def plain(value):
    # The registry loads adapters by file location; accept its R class even
    # when a test runner has imported this module under a second package name.
    if type(value).__name__ == 'R' and hasattr(value, 'v'):
        return value.v
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


class PoolMax2dAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load('POOL').adapter
        self.x = torch.tensor([[[[1., 9., 3.], [4., 5., 6.], [7., 8., 2.]]]])
        self.g = torch.tensor([[[[1., 2.], [3., 4.]]]])

    def test_hand_witness_has_values_indices_and_accumulated_vjp(self):
        fields = plain(self.adapter.reference(PROFILE, query(self.x, self.g, stride=1)).fields)
        self.assertEqual(fields, {'out': [[[[9, 9], [8, 8]]]],
                                  'indices': [[[[1, 1], [7, 7]]]],
                                  'grad': [[[[0, 3, 0], [0, 0, 0], [0, 7, 0]]]]})

    def test_registry_requires_the_versioned_index_task_choice(self):
        q = query(self.x, self.g, stride=1)
        self.assertEqual(registry.query('POOL', PROFILE, q).outcome, 'not_established')
        admitted = registry.query('POOL', PROFILE, q, task_bindings={PROFILE: CHOICE})
        self.assertEqual(admitted.outcome, 'value', admitted.note)
        self.assertEqual(set(admitted.fields), {'out', 'indices', 'grad'})

    def test_independent_coordinates_and_pytorch_210_cross_check(self):
        configurations = [
            {'kernel_size': 2, 'stride': None, 'padding': 0, 'dilation': 1, 'ceil_mode': False},
            {'kernel_size': 3, 'stride': 2, 'padding': 0, 'dilation': 1, 'ceil_mode': True},
            {'kernel_size': (2, 3), 'stride': (1, 2), 'padding': (1, 1), 'dilation': (2, 1), 'ceil_mode': False},
            {'kernel_size': (3, 2), 'stride': (2, 1), 'padding': (1, 0), 'dilation': 1, 'ceil_mode': True},
        ]
        for dtype, shape, options in itertools.product(
                DTYPES, [(1, 5, 6), (2, 2, 5, 6)], configurations):
            with self.subTest(dtype=dtype, shape=shape, options=options):
                count = int(np.prod(shape))
                permutation = torch.randperm(count, generator=torch.Generator().manual_seed(count + len(shape)))
                x = ((permutation.reshape(shape) - count // 2) / 8).to(dtype).requires_grad_()
                out, indices = F.max_pool2d(x, return_indices=True, **options)
                upstream = ((torch.arange(out.numel()).reshape(out.shape) % 7 - 3) / 4).to(dtype)
                dx, = torch.autograd.grad(out, x, upstream)
                expected = independent_coordinate_reference(x.detach(), upstream, options['kernel_size'],
                                                            options['stride'], options['padding'],
                                                            options['dilation'], options['ceil_mode'])
                fields = plain(self.adapter.reference(PROFILE, query(x.detach(), upstream, **options)).fields)
                self.assertEqual(fields, expected)
                self.assertEqual(out.detach().tolist(), np.vectorize(float)(np.array(expected['out'], dtype=object)).tolist())
                self.assertEqual(indices.tolist(), expected['indices'])
                self.assertEqual(dx.tolist(), np.vectorize(float)(np.array(expected['grad'], dtype=object)).tolist())

    def test_ties_and_incomplete_or_malformed_tasks_are_refused(self):
        tied = torch.tensor([[[[5., 5.], [1., 0.]]]])
        one = torch.ones(1, 1, 1, 1)
        with self.assertRaises(Unsupported):
            self.adapter.reference(PROFILE, query(tied, one))
        valid = query(self.x, self.g, stride=1)
        bad = [
            ({**valid, 'return_indices': False}, Unsupported),
            ({**valid, 'op': 'aten.max_pool2d_with_indices'}, Unsupported),
            ({**valid, 'padding': 2}, Invalid),
            ({**valid, 'upstream': record(one)}, Invalid),
            ({**valid, 'dilation': 0}, Invalid),
            ({**valid, 'ceil_mode': 1}, Invalid),
            ({k: v for k, v in valid.items() if k != 'upstream'}, Invalid),
        ]
        for malformed, error in bad:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                self.adapter.reference(PROFILE, malformed)

    def test_complete_output_comparison_checks_index_dtype_value_and_gradient(self):
        q = query(self.x, self.g, stride=1)
        valid = {'out': record(torch.tensor([[[[9., 9.], [8., 8.]]]])),
                 'indices': {'t': 'i64', 'shape': [1, 1, 2, 2], 'v': [[[[1, 1], [7, 7]]]]},
                 'grad': record(torch.tensor([[[[0., 3., 0.], [0., 0., 0.], [0., 7., 0.]]]]))}
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        for changed in (
            {k: v for k, v in valid.items() if k != 'indices'},
            {**valid, 'indices': {**valid['indices'], 't': 'f32'}},
            {**valid, 'indices': {**valid['indices'], 'v': [[[[1, 1], [7, 8]]]]}},
            {**valid, 'grad': record(torch.zeros_like(self.x))},
        ):
            self.assertFalse(self.adapter.check_output(PROFILE, q, changed))


if __name__ == '__main__':
    unittest.main()
