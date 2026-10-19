"""Independent Arb formulas, rather than another evaluation of the same ATen graph."""
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

import flint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid


def arb(value):
    value = Q(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def endpoint(value):
    mantissa, exponent = value.man_exp()
    return Q(int(mantissa)) * Q(2) ** int(exponent)


class SoftmaxIndependent(unittest.TestCase):
    def setUp(self):
        self.precision = flint.ctx.prec
        flint.ctx.prec = 512

    def tearDown(self):
        flint.ctx.prec = self.precision

    def encloses(self, result, expected):
        lo, hi = result.bounds()
        self.assertLessEqual(lo, endpoint(expected.lower()))
        self.assertGreaterEqual(hi, endpoint(expected.upper()))

    def test_binary_logistic_formula_and_both_online_outputs(self):
        for delta in (Q(-50), Q(-5, 3), Q(0), Q(7, 4), Q(40)):
            z = arb(delta).exp()
            probs = [z / (1 + z), 1 / (1 + z)]
            logs = [arb(delta) - (1 + z).log(), -(1 + z).log()]
            for shift in (Q(0), Q(10**80), Q(-10**80)):
                x = A.tensor([shift + delta, shift], "f64")
                for op, expected in ((A.softmax, probs), (A.safe_softmax, probs), (A.log_softmax, logs)):
                    result = op(x, 0)
                    self.assertEqual(result.shape, (2,))
                    for value, independent in zip(result.a, expected):
                        self.encloses(value, independent)
                maximum, mass = A.prepare_softmax_online(x, 0)
                self.assertEqual(maximum.shape, (1,))
                self.assertEqual(mass.shape, (1,))
                self.assertEqual(A.plain(maximum), [shift + max(delta, 0)])
                self.encloses(mass.a[0], 1 + arb(-abs(delta)).exp())

    def test_exact_translation_does_not_destroy_log_softmax_enclosures(self):
        base = A.log_softmax(A.tensor([Q(2, 3), Q(-5, 4)], "f64"), 0)
        for shift in (10**80, -10**80):
            shifted = A.log_softmax(A.tensor([shift + Q(2, 3), shift - Q(5, 4)], "f64"), 0)
            self.assertEqual([v.bounds() for v in shifted.a], [v.bounds() for v in base.a])

    def test_online_softmax_requires_one_integer_axis(self):
        x = A.tensor([[0, 1], [2, 3]], "f32")
        for dim in (None, [], [0], [0, 1], True, 1.0, 2, -3):
            with self.subTest(dim=dim), self.assertRaises(Invalid):
                A.prepare_softmax_online(x, dim)

    def test_log_softmax_vjp_against_independent_dense_jacobian(self):
        logits = [Q(-1), Q(2, 3), Q(3)]
        exps = [arb(x).exp() for x in logits]
        total = sum(exps)
        probabilities = [e / total for e in exps]
        grad = [Q(-3), Q(4, 5), Q(2)]
        output = A.log_softmax(A.tensor(logits, "f64"), 0)
        result = A.log_softmax_backward_data(A.tensor(grad, "f64"), output, 0, "f64")
        self.assertEqual(result.shape, (3,))
        for j, value in enumerate(result.a):
            independent = sum(arb(grad[i]) * ((1 if i == j else 0) - probabilities[j]) for i in range(3))
            self.encloses(value, independent)


if __name__ == "__main__":
    unittest.main()
