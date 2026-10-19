import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, lazy as L
from kernel_spec.num import Invalid, NotEstablished


class LazyCumsum(unittest.TestCase):
    def test_axes_scalar_and_empty_metadata(self):
        for shape in ([2], [0]):
            x = L.as_lazy(A.full(shape, 1, "i32"))
            for axis in (1, -2, True, 0.0):
                with self.subTest(shape=shape, axis=axis), self.assertRaises(Invalid):
                    L.l_cumsum(x, axis)
        for axis in (0, -1):
            y = L.l_cumsum(L.as_lazy(A.tensor(7, "i32")), axis)
            self.assertEqual((y.shape, y.dtype, y.get(())), ((), "i64", 7))
        for shape in ([0, 3], [2, 0]):
            x = L.as_lazy(A.full(shape, 1, "f32"))
            for axis in (0, 1):
                self.assertEqual(L.l_cumsum(x, axis, dtype="f16").shape, tuple(shape))

    def test_conversion_and_boolean_domain(self):
        for dtype, value in (("i64", Fraction(7, 4)), ("f16", Fraction(4097, 4096))):
            x = L.as_lazy(A.tensor([value, value], "f32"))
            y = L.l_cumsum(x, 0, dtype=dtype)
            self.assertEqual(y.dtype, dtype)
            self.assertEqual([A.plain(y.get((j,))) for j in (0, 1)], [1, 2])
        with self.assertRaises(NotEstablished):
            L.l_cumsum(L.as_lazy(A.tensor([1, 1], "i64")), 0, dtype="b8")

    def test_every_prefix_matches_independent_native_execution(self):
        import torch
        x = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4) - 12
        for axis in (0, 1, -1):
            expected = torch.cumsum(x, axis)
            result = L.l_cumsum(L.as_lazy(A.tensor(x, "i32")), axis)
            self.assertEqual(result.dtype, "i64")
            for index in reversed(list(np.ndindex(result.shape))):
                self.assertEqual(result.get(index), expected[index].item())

    def test_long_prefix_uses_budget_instead_of_python_recursion_limit(self):
        x = L.Lazy((3000,), "i64", lambda index: 1)
        y = L.l_cumsum(x, 0)
        with L.evaluation_budget(5):
            with self.assertRaises(NotEstablished):
                y.get((2999,))
        self.assertNotIn((2999,), y._memo)
        with L.evaluation_budget(6000):
            self.assertEqual(y.get((2999,)), 3000)
        with L.evaluation_budget(1) as budget:
            self.assertEqual(y.get((1000,)), 1001)
        self.assertEqual(budget.used, 0)


if __name__ == "__main__":
    unittest.main()
