import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import lazy as L
from kernel_spec.num import Invalid


class LazyCoordinates(unittest.TestCase):
    def test_shape_is_not_truncated_or_negative(self):
        for shape in ((3.9,), (-1,), (True,), (np.float64(2),), "23"):
            with self.subTest(shape=shape), self.assertRaises(Invalid):
                L.Lazy(shape, "i64", lambda index: 0)
        self.assertEqual(L.Lazy((np.int64(2), 3), "i64", lambda index: 0).shape, (2, 3))

    def test_invalid_coordinates_do_not_read_cache_or_spend_budget(self):
        reads = []
        x = L.Lazy((2, 3), "i64", lambda index: reads.append(index) or 7)
        self.assertEqual(x.get((1, 2)), 7)
        for index in ((1.9, 2), (True, 2), (-1, 2), (2, 0), (0, 3), (1,), (1, 2, 0), "12", 1):
            with self.subTest(index=index), L.evaluation_budget(1) as budget:
                with self.assertRaises(Invalid):
                    x.get(index)
                self.assertEqual(budget.used, 0)
        self.assertEqual(reads, [(1, 2)])
        self.assertEqual(x._memo, {(1, 2): 7})

    def test_scalar_and_empty_query_domains(self):
        scalar = L.Lazy((), "i64", lambda index: 5)
        self.assertEqual(scalar.get(()), 5)
        with self.assertRaises(Invalid):
            scalar.get((0,))
        empty = L.Lazy((0, 2), "i64", lambda index: self.fail("empty tensor was read"))
        with self.assertRaises(Invalid):
            empty.get((0, 0))

    def test_element_count_uses_exact_integer_arithmetic(self):
        self.assertEqual(L.Lazy((2**32, 2**32), "i64", lambda index: 0).numel(), 2**64)
        self.assertEqual(L.Lazy((2**32, 2**32, 0), "i64", lambda index: 0).numel(), 0)
        self.assertEqual(L.Lazy((), "i64", lambda index: 0).numel(), 1)


if __name__ == "__main__":
    unittest.main()
