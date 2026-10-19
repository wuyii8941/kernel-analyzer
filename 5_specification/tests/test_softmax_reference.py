import sys
import unittest
from fractions import Fraction
from pathlib import Path

import flint
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from kernel_spec import real as R
from kernel_spec.num import Invalid, NotEstablished, Undefined
from softmax_reference import evaluate


class IndependentSoftmax(unittest.TestCase):
    def test_all_equal_logits_and_single_class(self):
        out = evaluate(np.full((2, 3), 2.), np.array([0, -100]))
        self.assertEqual(out["maxima"], [(Fraction(2), Fraction(2))] * 2)
        lo, hi = R.log(R.R(3)).bounds()
        self.assertLessEqual(max(lo, out["mean_loss"][0]), min(hi, out["mean_loss"][1]))
        self.assertEqual(out["distinct_exponentials"], 2)
        self.assertEqual(out["count"], 1)
        single = evaluate(np.array([[3.], [-2.]]), np.array([0, 0]))
        self.assertEqual(single["mean_loss"], (0, 0))
        self.assertEqual(single["log_mass"], [(0, 0), (0, 0)])

    def test_distinct_count_formula_agrees_with_unshifted_independent_backend(self):
        x = np.array([[1., 1., -2., 0.], [-1., 2., 2., 2.]])
        out = evaluate(x, np.array([2, 1]))
        losses = []
        for i, target in enumerate((2, 1)):
            log_sum = R.log(sum((R.exp(R.R(float(v))) for v in x[i]), R.R(0)))
            expected = log_sum - R.R(float(x[i, target]))
            losses.append(expected)
            lo, hi = (log_sum - R.R(float(max(x[i])))).bounds()
            a, b = out["log_mass"][i]
            self.assertLessEqual(max(a, lo), min(b, hi))
        lo, hi = ((losses[0] + losses[1]) / R.R(2)).bounds()
        a, b = out["mean_loss"]
        self.assertLessEqual(max(a, lo), min(b, hi))

    def test_invalid_domains_and_budget_failures_are_explicit(self):
        x = np.ones((2, 3))
        for labels in (np.array([0.9, 1.]), np.array([0]), np.array([3, 0])):
            with self.assertRaises(Invalid):
                evaluate(x, labels)
        with self.assertRaises(Undefined):
            evaluate(x, np.array([-100, -100]))
        old = flint.ctx.prec
        with self.assertRaises(NotEstablished):
            evaluate(x, np.array([0, 0]), max_unique=1, precision=128)
        self.assertEqual(flint.ctx.prec, old)
        with self.assertRaises(NotEstablished):
            evaluate(x, np.array([0, 0]), max_elements=5)
        with self.assertRaises(NotEstablished):
            evaluate(np.array([[float('nan')]]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
