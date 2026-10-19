"""Official default ATen argument names must reach the reviewed matrix formulas."""
import unittest

from kernel_spec import registry
from kernel_spec.num import evaluate


def tensor(v):
    return {'t': 'f32', 'v': v}


class MatmulPublicKeywords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load('MATMUL').adapter

    def test_all_default_schema_names_match_positional_formulas(self):
        a, b = tensor([[1, 2]]), tensor([[3], [4]])
        aa, bb = tensor([[[1, 2]]]), tensor([[[3], [4]]])
        cases = [
            ('mm', [a, b], {'self': a, 'mat2': b}),
            ('matmul', [a, b], {'self': a, 'other': b}),
            ('bmm', [aa, bb], {'self': aa, 'mat2': bb}),
            ('addmm', [tensor([2]), a, b], {'self': tensor([2]), 'mat1': a, 'mat2': b}),
            ('baddbmm', [tensor([2]), aa, bb], {'self': tensor([2]), 'batch1': aa, 'batch2': bb}),
            ('linear', [a, tensor([[3, 4]]), tensor([2])],
             {'input': a, 'weight': tensor([[3, 4]]), 'bias': tensor([2])}),
        ]
        for op, args, kwargs in cases:
            with self.subTest(op=op):
                expected = self.adapter.reference('real', {'op': 'aten.' + op, 'args': args})
                observed = self.adapter.reference('real', {'op': 'aten.' + op, 'kwargs': kwargs})
                self.assertEqual(observed.fields, expected.fields)

    def test_beta_zero_named_addend_still_ignores_nan_and_infinity(self):
        for value in ('nan', 'inf', '-inf'):
            q = {'op': 'aten.addmm', 'kwargs': {'self': tensor([[value]]), 'mat1': tensor([[2]]),
                                               'mat2': tensor([[3]]), 'beta': 0, 'alpha': 2}}
            with self.subTest(value=value):
                self.assertEqual(self.adapter.reference('real', q).fields['out'], [[12]])
                self.assertTrue(self.adapter.check_output('real', q, {
                    'out': {'t': 'f32', 'shape': [1, 1], 'v': [[12]]}}))
                q['kwargs']['beta'] = 1
                self.assertIn(evaluate(self.adapter.reference, 'real', q).outcome,
                              ('invalid_input', 'unsupported'))

    def test_aliases_cannot_hide_duplicates_or_unsupported_overloads(self):
        a = tensor([[1]])
        calls = [
            ({'op': 'aten.mm', 'args': [a], 'kwargs': {'self': a, 'mat2': a}}, 'invalid_input'),
            ({'op': 'aten.mm', 'kwargs': {'a': a, 'self': a, 'mat2': a}}, 'invalid_input'),
            ({'op': 'aten.linear', 'kwargs': {'input': a, 'weight': a, 'w': a}}, 'invalid_input'),
            ({'op': 'aten.addmm', 'kwargs': {'self': a, 'mat1': a, 'mat2': a, 'out_dtype': 'f32'}}, 'unsupported'),
            ({'op': 'aten.mm', 'kwargs': {'self': a, 'mat2': a, 'out': a}}, 'unsupported'),
        ]
        for q, outcome in calls:
            with self.subTest(q=q):
                self.assertEqual(evaluate(self.adapter.reference, 'real', q).outcome, outcome)
