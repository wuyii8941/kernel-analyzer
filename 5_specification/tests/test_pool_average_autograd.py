"""Independent linear-map checks for the explicitly chosen 2D average-pooling task."""
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

PROFILE = 'finite_avg2d_autograd'
CHOICE = 'Use PyTorch 2.10 two-dimensional average pooling: count explicit padding but exclude ceil overhang; differentiate the same chosen linear map.'


def record(t):
    return {'t': {torch.float32: 'f32', torch.float64: 'f64', torch.float16: 'f16', torch.bfloat16: 'bf16'}[t.dtype],
            'shape': list(t.shape), 'v': t.tolist()}


def query(x, g, **options):
    return {'op': 'avg_pool2d', 'x': record(x), 'upstream': record(g), 'kernel_size': 3,
            'stride': 2, 'padding': 0, 'ceil_mode': True, 'count_include_pad': True,
            'divisor_override': None, **options}


def matrix_reference(x, g, k, stride, padding, ceil, include, override):
    # Build a rational linear map by testing every input coordinate against
    # independently enumerated window origins, then apply it and its transpose.
    h, w = x.shape[-2:]
    def origins(length, kernel, step, pad):
        return [i * step - pad for i in range(length + 2 * pad + 1)
                if i * step - pad + kernel <= length + pad + (step - 1 if ceil else 0)
                and (not ceil or i * step - pad < length)]
    oh, ow = origins(h, k[0], stride[0], padding[0]), origins(w, k[1], stride[1], padding[1])
    matrix = []
    for top, left in itertools.product(oh, ow):
        real_h = max(0, min(top + k[0], h) - max(top, 0))
        real_w = max(0, min(left + k[1], w) - max(left, 0))
        d = override if override is not None else (
            (min(top + k[0], h + padding[0]) - top) * (min(left + k[1], w + padding[1]) - left)
            if include else real_h * real_w)
        matrix.append([Fraction(1, d) if top <= r < top + k[0] and left <= c < left + k[1] else Fraction(0)
                       for r, c in itertools.product(range(h), range(w))])
    xv = [Fraction(v) for v in x.flatten().tolist()]
    gv = [Fraction(v) for v in g.flatten().tolist()]
    y = [sum((a * b for a, b in zip(row, xv)), Fraction(0)) for row in matrix]
    dx = [sum((matrix[o][i] * gv[o] for o in range(len(matrix))), Fraction(0)) for i in range(h * w)]
    return np.array(y, dtype=object).reshape(1, 1, len(oh), len(ow)).tolist(), np.array(dx, dtype=object).reshape(x.shape).tolist()


class PoolAverageAutograd(unittest.TestCase):
    def setUp(self):
        self.adapter = registry.load('POOL').adapter
        self.x = torch.ones(1, 1, 4, 4)
        self.g = torch.tensor([[[[0., 0.], [0., 1.]]]])

    def test_tail_forward_and_gradient_hand_witness(self):
        result = self.adapter.reference(PROFILE, query(self.x, self.g))
        self.assertEqual(result.fields['out'], [[[[1, 1], [1, 1]]]])
        self.assertEqual(result.fields['grad'], [[[[0, 0, 0, 0], [0, 0, 0, 0],
                                                   [0, 0, Fraction(1, 4), Fraction(1, 4)],
                                                   [0, 0, Fraction(1, 4), Fraction(1, 4)]]]])

    def test_registry_requires_explicit_divisor_task_choice(self):
        q = query(self.x, self.g)
        self.assertEqual(registry.query('POOL', PROFILE, q).outcome, 'not_established')
        result = registry.query('POOL', PROFILE, q, task_bindings={PROFILE: CHOICE})
        self.assertEqual(result.outcome, 'value', result.note)

    def test_linear_map_and_transpose_independent_parameter_cases(self):
        for k, stride, padding, ceil, include, override in itertools.product(
            [(2, 3), (3, 2)], [(1, 2), (2, 3)], [(0, 0), (1, 1)], [False, True], [False, True], [None, -5]):
            with self.subTest(k=k, stride=stride, padding=padding, ceil=ceil, include=include, override=override):
                x = torch.arange(-12., 12., dtype=torch.float64).reshape(1, 1, 4, 6).requires_grad_()
                y = F.avg_pool2d(x, k, stride, padding, ceil, include, override)
                g = (torch.arange(y.numel(), dtype=x.dtype).reshape(y.shape) - 2) / 4
                dx, = torch.autograd.grad(y, x, g)
                q = query(x.detach(), g, kernel_size=list(k), stride=list(stride), padding=list(padding),
                          ceil_mode=ceil, count_include_pad=include, divisor_override=override)
                ref = self.adapter.reference(PROFILE, q).fields
                wanted_y, wanted_dx = matrix_reference(x.detach(), g, k, stride, padding, ceil, include, override)
                self.assertEqual(ref, {'out': wanted_y, 'grad': wanted_dx})
                for actual, expected in ((y, wanted_y), (dx, wanted_dx)):
                    floats = np.vectorize(float)(np.array(expected, dtype=object))
                    np.testing.assert_allclose(actual.detach().numpy(), floats, rtol=1e-14, atol=1e-14)

    def test_all_physical_dtypes_rank3_and_single_parameter_form(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x, g = torch.ones(1, 2, 2, dtype=dtype), torch.ones(1, 1, 1, dtype=dtype)
                fields = self.adapter.reference(PROFILE, query(x, g, kernel_size=[2], stride=None, ceil_mode=False)).fields
                self.assertEqual(fields['out'], [[[1]]])
                self.assertEqual(fields['grad'], [[[Fraction(1, 4)] * 2] * 2])

    def test_structure_dtype_shape_and_parameters_cannot_be_silently_dropped(self):
        valid = query(self.x, self.g)
        bad = [({**valid, 'dilation': 1}, Invalid), ({**valid, 'stride': 3.9}, Invalid),
               ({**valid, 'divisor_override': 0}, Invalid), ({**valid, 'divisor_override': True}, Invalid),
               ({**valid, 'padding': 2}, Invalid), ({**valid, 'ceil_mode': 1}, Invalid),
               ({**valid, 'upstream': record(torch.ones(1, 1, 1, 1))}, Invalid),
               ({**valid, 'upstream': record(self.g.double())}, Invalid),
               ({**valid, 'x': record(torch.full_like(self.x, float('nan')))}, Unsupported)]
        for q, error in bad:
            with self.subTest(q=q), self.assertRaises(error):
                self.adapter.reference(PROFILE, q)

    def test_complete_gradient_output_and_tail_mutation_are_checked(self):
        q = query(self.x, self.g)
        out = record(torch.ones(1, 1, 2, 2))
        grad = torch.zeros_like(self.x); grad[..., 2:, 2:] = .25
        valid = {'out': out, 'grad': record(grad)}
        self.assertTrue(self.adapter.check_output(PROFILE, q, valid))
        self.assertFalse(self.adapter.check_output(PROFILE, q, {'out': out}))
        grad[..., 2:, 2:] = 1 / 9
        self.assertFalse(self.adapter.check_output(PROFILE, q, {'out': out, 'grad': record(grad)}))


if __name__ == '__main__':
    unittest.main()
