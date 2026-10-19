"""Reduction coordinates and empty output domains must not be inferred by NumPy."""
import sys
import unittest
from pathlib import Path
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid, Undefined


class ReductionDomains(unittest.TestCase):
    def test_empty_extremum_axis_is_rejected_even_without_output_rows(self):
        for op in (A.amax, A.amin, A.argmax, A.argmin, A.max_dim, A.min_dim):
            x = A.view(A.tensor([], "f32"), [0, 0])
            for axis in (0, 1):
                with self.subTest(op=op.__name__, axis=axis), self.assertRaises(Invalid):
                    op(x, axis)
        with self.assertRaises(Invalid):
            A.prepare_softmax_online(x, 1)

    def test_arg_reduction_requires_one_axis_or_none(self):
        x = A.tensor([[1, 3], [2, 0]], "f32")
        for op in (A.max_dim, A.min_dim):
            with self.assertRaises(Invalid):
                op(x, None)
        for op in (A.argmax, A.argmin, A.max_dim, A.min_dim):
            for axis in ([], [0], [0, 1]):
                with self.subTest(op=op.__name__, axis=axis), self.assertRaises(Invalid):
                    op(x, axis)

    def test_arg_reduction_over_all_axes_retains_requested_dimensions(self):
        x = A.tensor([[1, 3], [2, 0]], "f32")
        for op, expected in ((A.argmax, 1), (A.argmin, 3)):
            for keep in (True, False):
                result = op(x, None, keep)
                self.assertEqual(result.shape, (1, 1) if keep else ())
                self.assertEqual(A.plain(result), [[expected]] if keep else expected)

    def test_explicit_dtype_converts_before_reduction(self):
        x = A.tensor([Fraction(7, 4), Fraction(7, 4)], "f32")
        self.assertEqual(A.plain(A.sum_(x, dtype="i64")), 2)
        self.assertEqual(A.plain(A.prod(x, dtype="i64")), 1)
        x = A.tensor([Fraction(4097, 4096)] * 2, "f32")
        for op, expected in ((A.sum_, 2), (A.mean, 1), (A.prod, 1)):
            with self.subTest(op=op.__name__):
                y = op(x, dtype="f16")
                self.assertEqual(y.dtype, "f16")
                self.assertEqual(A.plain(y), expected)

    def test_mean_requires_a_floating_result_type(self):
        for values in ([1, 2], []):
            with self.assertRaises(Invalid):
                A.mean(A.tensor(values, "i64"))
            with self.assertRaises(Invalid):
                A.mean(A.tensor(values, "f32"), dtype="i64")
        self.assertEqual(A.plain(A.mean(A.tensor([1, 2], "i64"), dtype="f32")), Fraction(3, 2))

    def test_product_promotes_integer_inputs(self):
        for dtype in ("i8", "i32", "b8"):
            self.assertEqual(A.prod(A.tensor([1, 1], dtype)).dtype, "i64")

    def test_explicit_boolean_reduction_preserves_boolean_values(self):
        for values, total, product in (([1, 1], True, True), ([1, 0], True, False), ([], False, True)):
            x = A.tensor(values, "i64")
            for op, expected in ((A.sum_, total), (A.prod, product)):
                result = A.plain(op(x, dtype="b8"))
                self.assertIs(result, expected)

    def test_axes_are_not_wrapped_or_deduplicated(self):
        x = A.tensor([[1, 2, 3], [4, 5, 6]], "f32")
        for op in (A.sum_, A.mean, A.amax, A.amin, A.prod):
            for dim in (2, -3, [0, 0], [0, -2], True, 0.0):
                with self.subTest(op=op.__name__, dim=dim), self.assertRaises(Invalid):
                    op(x, dim)
        self.assertEqual(A.plain(A.sum_(x, [-1])), [6, 15])
        self.assertEqual(A.plain(A.sum_(x, [])), 21)

    def test_scalar_axes_are_validated(self):
        x = A.tensor(7, "f32")
        for op in (A.sum_, A.mean, A.amax, A.amin, A.prod):
            for dim in (0, -1, [], None):
                self.assertEqual(A.plain(op(x, dim)), 7)
            for dim in (1, -2, [0, -1]):
                with self.subTest(op=op.__name__, dim=dim), self.assertRaises(Invalid):
                    op(x, dim)

    def test_no_output_rows_is_not_an_empty_reduction_row(self):
        x = A.view(A.tensor([], "f32"), [0, 3])
        for op in (A.sum_, A.mean, A.amax, A.amin, A.prod):
            for keep in (False, True):
                with self.subTest(op=op.__name__, keep=keep):
                    result = op(x, 1, keep)
                    self.assertEqual(result.shape, (0, 1) if keep else (0,))
                    self.assertEqual(A.plain(result), [])
        y = A.view(A.tensor([], "f32"), [2, 0])
        self.assertEqual(A.plain(A.sum_(y, 1)), [0, 0])
        self.assertEqual(A.plain(A.prod(y, 1)), [1, 1])
        with self.assertRaises(Undefined):
            A.mean(y, 1)
        with self.assertRaises(Invalid):
            A.amax(y, 1)


if __name__ == "__main__":
    unittest.main()
