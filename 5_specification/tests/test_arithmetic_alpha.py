"""Scaling must be validated before element evaluation, including empty tensors."""
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid


class ArithmeticAlpha(unittest.TestCase):
    def test_integer_scaling_is_not_truncated(self):
        for values in ([2], []):
            x = A.tensor(values, "i64")
            for op in (A.add, A.sub, A.rsub):
                for alpha in (Fraction(3, 2), 1.5, 1.0, True):
                    with self.subTest(op=op.__name__, values=values, alpha=alpha), self.assertRaises(Invalid):
                        op(x, x, alpha=alpha)

    def test_boolean_scaling_preserves_zero(self):
        a, b = A.tensor([False, True], "b8"), A.tensor([True, False], "b8")
        for alpha in (0, False):
            self.assertEqual(A.add(a, b, alpha=alpha).a.tolist(), [False, True])
        for alpha in (1, -1, 2, True):
            self.assertEqual(A.add(a, b, alpha=alpha).a.tolist(), [True, True])

    def test_valid_numeric_scaling(self):
        for dtype in ("i64", "f32"):
            a, b = A.tensor([2, -3], dtype), A.tensor([5, 7], dtype)
            self.assertEqual(A.plain(A.add(a, b, alpha=-2)), [-8, -17])
            self.assertEqual(A.plain(A.sub(a, b, alpha=-2)), [12, 11])
        a, b = A.tensor([2], "f32"), A.tensor([3], "f32")
        self.assertEqual(A.plain(A.add(a, b, alpha=Fraction(1, 3))), [3])


if __name__ == "__main__":
    unittest.main()
