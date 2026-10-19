"""Layout preconditions must hold before any coordinate is requested."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, lazy as L
from kernel_spec.num import Invalid

class LazyLayouts(unittest.TestCase):
    def test_invalid_layouts_rejected_before_evaluation(self):
        cases = [(L.l_view, (0,), ([0, -1],)),
                 (L.l_view, (0,), ([-1, -1],)),
                 (L.l_permute, (2, 3), ([2, 1],)),
                 (L.l_permute, (2, 3), ([0, 0],)),
                 (L.l_permute, (2, 3), ([0],)),
                 (L.l_expand, (2,), ([-1, 2],)),
                 (L.l_expand, (2,), ([3],)),
                 (L.l_expand, (2,), ([0, 3],)),
                 (L.l_unsqueeze, (2,), (2,)),
                 (L.l_squeeze, (1, 2), ([0, -2],)),
                 (L.l_squeeze, (), (1,)),
                 (L.l_transpose, (2, 3), (2, 0)),
                 (L.l_t, (1, 2, 3), ())]
        for fn, shape, args in cases:
            with self.subTest(op=fn.__name__, shape=shape, args=args):
                x = L.Lazy(shape, 'i64', lambda _: self.fail('input read during validation'))
                with self.assertRaises(Invalid):
                    fn(x, *args)

    def test_valid_values_scalars_and_empty_shapes(self):
        cases = [('view', (2, 3), ([3, -1],)), ('view', (0, 3), ([-1, 3],)),
                 ('permute', (2, 3), ([-1, -2],)), ('permute', (), ([],)),
                 ('expand', (1, 2), ([3, -1],)), ('expand', (1, 2), ([0, 2],)),
                 ('unsqueeze', (), (-1,)), ('squeeze', (), (-1,)),
                 ('squeeze', (1, 2, 1), ([0, -1],)), ('transpose', (), (-1, 0)),
                 ('t', (2, 3), ())]
        for name, shape, args in cases:
            with self.subTest(op=name, shape=shape):
                array = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
                actual = getattr(L, 'l_' + name)(L.from_array(array, 'i64'), *args)
                expected = getattr(A, name)(A.Ten(array.astype(object), 'i64'), *args)
                self.assertEqual(actual.shape, expected.shape)
                for idx in np.ndindex(expected.shape):
                    self.assertEqual(actual.get(idx), expected.a[idx])

    def test_large_shape_products_do_not_overflow(self):
        x = L.Lazy((2**32, 2**32), 'i64', lambda idx: idx[0] + idx[1])
        y = L.l_view(x, [2**64])
        self.assertEqual(y.shape, (2**64,))
        self.assertEqual(y.get((2**64 - 1,)), 2 * (2**32 - 1))

if __name__ == '__main__':
    unittest.main()
