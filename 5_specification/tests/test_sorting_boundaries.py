"""Sorting boundary regressions; scalar top-k behavior also checked against local PyTorch."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, sorting as S
from kernel_spec.num import Invalid


class SortingBoundaries(unittest.TestCase):
    def test_invalid_dimensions_are_not_wrapped(self):
        for shape in ((), (2,), (2, 3)):
            a = A.tensor(np.zeros(shape, dtype=int).tolist(), "f32")
            rank = max(1, len(shape))
            for dim in (-rank-1, rank):
                for fn in (S.sort, S.argsort, lambda a, dim: S.topk(a, 1, dim)):
                    with self.subTest(shape=shape, dim=dim, fn=fn), self.assertRaises(Invalid):
                        fn(a, dim=dim)

    def test_boolean_counts_and_dimensions_are_rejected(self):
        a = A.tensor([1, 2], "f32")
        with self.assertRaises(Invalid):
            S.topk(a, True)
        with self.assertRaises(Invalid):
            S.sort(a, dim=True)

    def test_empty_outer_dimensions_preserve_output_extent(self):
        a = A.Ten(np.empty((0, 3), dtype=object), "f32")
        for result, want in ((S.sort(a, 1), (0, 3)), (S.topk(a, 2, 1), (0, 2))):
            self.assertEqual(result[0].a.shape, want)
            self.assertEqual(result[1].a.shape, want)
        self.assertEqual(S.sort(a, 1)[0].a.shape, (0, 3))
        self.assertEqual(S.topk(a, 2, 1)[0].a.shape, (0, 2))

    def test_scalar_topk_zero_keeps_scalar_output(self):
        # Observed PyTorch behavior: k=0 and k=1 both preserve scalar value/index.
        for k in (0, 1):
            values, indices = S.topk(A.tensor(3, "f32"), k)
            self.assertEqual(values.a.shape, ())
            self.assertEqual(values.a.item().v, 3)
            self.assertEqual(indices.a.item(), 0)


if __name__ == "__main__":
    unittest.main()
