import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, lazy as L
from kernel_spec.num import Invalid


class LazyReductions(unittest.TestCase):
    def test_invalid_axes_are_rejected_before_coordinate_queries(self):
        x = L.as_lazy(A.full([0, 2], 1, "f32"))
        for op in (L.l_sum, L.l_mean, L.l_amax, L.l_amin, L.l_any):
            for dim in (2, -3, [0, 0], [0, -2], 0.5, True):
                with self.subTest(op=op.__name__, dim=dim), self.assertRaises(Invalid):
                    op(x, dim)

    def test_empty_extrema_are_invalid_even_without_output_coordinates(self):
        x = L.as_lazy(A.full([0, 0], 1, "f32"))
        for op in (L.l_amax, L.l_amin, L.l_max_dim):
            with self.subTest(op=op.__name__), self.assertRaises(Invalid):
                op(x, 1)

    def test_explicit_dtype_casts_before_reduction(self):
        for op in (L.l_sum, L.l_mean):
            x = L.as_lazy(A.tensor([1.00048828125], "f32"))
            y = op(x, dtype="f16")
            self.assertEqual(y.dtype, "f16")
            self.assertEqual(y.get(()).v, 1)
        with self.assertRaises(Invalid):
            L.l_mean(L.as_lazy(A.full([0], 1, "i32")))
        y = L.l_sum(L.as_lazy(A.tensor([1, 2], "i32")), dtype="b8")
        self.assertIs(y.get(()), True)

    def test_scalar_and_full_coordinate_results_match_native(self):
        import torch
        for shape, dim in (((), 0), ((), -1), ((2, 3), 1), ((2, 3), [0, 1]), ((0, 3), 1)):
            for name in ("sum", "mean", "amax", "amin"):
                for keep in (False, True):
                    with self.subTest(shape=shape, dim=dim, op=name, keep=keep):
                        x = torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape) * 2
                        expected = getattr(torch, name)(x, dim=dim, keepdim=keep)
                        result = getattr(L, "l_" + name)(L.as_lazy(A.tensor(x, "f64")), dim, keep)
                        self.assertEqual(result.dtype, "f64")
                        self.assertEqual(result.shape, tuple(expected.shape))
                        for index in np.ndindex(result.shape):
                            self.assertEqual(A.plain(result.get(index)), expected[index].item())


if __name__ == "__main__":
    unittest.main()
