import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid


class SoftmaxBackwardDomains(unittest.TestCase):
    def test_dense_jacobian_contraction_independently_matches_all_coordinates(self):
        # d y_i / d x_j = delta_ij*y_i - y_i*y_j. Contract the full
        # Jacobian with g, rather than reproducing the evaluator's reduction.
        rows = [[Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)],
                [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)]]
        grads = [[3, -2, 7], [-1, 5, 2]]
        expected = [[sum(Fraction(grads[r][i]) * ((p[i] if i == j else 0) - p[i] * p[j])
                         for i in range(3)) for j in range(3)] for r, p in enumerate(rows)]
        for dtype in ("f16", "bf16", "f32", "f64"):
            for transpose in (False, True):
                y, g = A.tensor(rows, "f32"), A.tensor(grads, "f32")
                if transpose:
                    y, g = A.transpose(y, 0, 1), A.transpose(g, 0, 1)
                result = A.softmax_backward_data(g, y, 0 if transpose else 1, dtype)
                if transpose:
                    result = A.transpose(result, 0, 1)
                self.assertEqual(A.plain(result), expected)
                self.assertEqual(result.dtype, dtype)

    def test_explicit_input_dtype_describes_the_real_gradient(self):
        y = A.tensor([Fraction(129, 256), Fraction(127, 256)], "f32")
        g = A.tensor([1, 0], "f32")
        result = A.softmax_backward_data(g, y, 0, "f16")
        expected = Fraction(16383, 65536)
        self.assertEqual(result.dtype, "f16")
        self.assertEqual(A.plain(result), [expected, -expected])
        # An explicit graph conversion is a separate task operation.
        self.assertNotEqual(A.plain(A.convert_element_type(result, "f16"))[0], expected)
        log_result = A.log_softmax_backward_data(A.tensor([1, 2], "f32"),
                                               A.tensor([0, "-inf"], "f32"), 0, "bf16")
        self.assertEqual(log_result.dtype, "bf16")
        self.assertEqual(A.plain(log_result), [-2, 2])

    def test_shape_mismatch_cannot_broadcast_a_gradient(self):
        for op in (A.softmax_backward_data, A.log_softmax_backward_data):
            for shapes in (([2], [1]), ([2], [1, 2]), ([0, 2], [0, 1])):
                g = A.full(shapes[0], 0, "f32")
                y = A.full(shapes[1], 0, "f32")
                with self.subTest(op=op.__name__, shapes=shapes), self.assertRaises(Invalid):
                    op(g, y, 0, "f32")

    def test_types_and_scalar_axis_validated_even_when_empty(self):
        for op in (A.softmax_backward_data, A.log_softmax_backward_data):
            for values in ([0, 0], []):
                x = A.tensor(values, "f32")
                for dtype in (None, "i32", "b8", "unknown"):
                    with self.subTest(op=op.__name__, dtype=dtype), self.assertRaises(Invalid):
                        op(x, x, 0, dtype)
                for dtype in ("i64", "b8", "f64"):
                    with self.assertRaises(Invalid):
                        op(x, A.tensor(values, dtype), 0, "f32")
                for axis in (None, [], [0], True, 0.0, 1, -2):
                    with self.subTest(axis=axis), self.assertRaises(Invalid):
                        op(x, x, axis, "f32")

    def test_scalar_and_empty_gradients_keep_shape_and_type(self):
        for op in (A.softmax_backward_data, A.log_softmax_backward_data):
            y = A.tensor(1 if op is A.softmax_backward_data else 0, "f32")
            result = op(A.tensor(3, "f32"), y, -1, "f16")
            self.assertEqual((result.shape, result.dtype, A.plain(result)), ((), "f16", 0))
            for shape in ([0, 2], [2, 0]):
                x = A.full(shape, 0, "f32")
                for dim in (0, 1):
                    result = op(x, x, dim, "bf16")
                    self.assertEqual((result.shape, result.dtype), (tuple(shape), "bf16"))


if __name__ == "__main__":
    unittest.main()
