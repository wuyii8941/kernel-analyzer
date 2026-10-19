import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid, NotEstablished


class CumsumDomains(unittest.TestCase):
    def test_axes_and_scalar(self):
        for values in ([1, 2], []):
            for dim in (1, -2, True, 0.0):
                with self.subTest(values=values, dim=dim), self.assertRaises(Invalid):
                    A.cumsum(A.tensor(values, "i64"), dim)
        for dim in (0, -1):
            self.assertEqual(A.plain(A.cumsum(A.tensor(7, "i32"), dim)), 7)

    def test_conversion_precedes_every_prefix(self):
        x = A.tensor([Fraction(7, 4), Fraction(7, 4)], "f32")
        self.assertEqual(A.plain(A.cumsum(x, 0, dtype="i64")), [1, 2])
        x = A.tensor([Fraction(4097, 4096)] * 2, "f32")
        self.assertEqual(A.plain(A.cumsum(x, 0, dtype="f16")), [1, 2])

    def test_all_prefix_coordinates_and_empty_shapes(self):
        x = A.tensor([[1, 2, 3], [4, 5, 6]], "i32")
        result = A.cumsum(x, -1)
        self.assertEqual(result.dtype, "i64")
        self.assertEqual(A.plain(result), [[1, 3, 6], [4, 9, 15]])
        self.assertEqual(A.plain(A.cumsum(x, 0)), [[1, 2, 3], [5, 7, 9]])
        for shape in ([0, 3], [2, 0]):
            empty = A.view(A.tensor([], "f32"), shape)
            for axis in (0, 1):
                self.assertEqual(A.cumsum(empty, axis).shape, tuple(shape))

    def test_unestablished_boolean_output_is_not_integer_counts(self):
        with self.assertRaises(NotEstablished):
            A.cumsum(A.tensor([1, 1], "i64"), 0, dtype="b8")


if __name__ == "__main__":
    unittest.main()
