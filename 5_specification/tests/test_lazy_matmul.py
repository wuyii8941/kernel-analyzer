import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, lazy as L
from kernel_spec.num import Invalid, NotEstablished


class LazyMatmul(unittest.TestCase):
    def unread(self, shape, dtype="f32"):
        def read(index):
            self.fail("metadata validation read an input coordinate")
        return L.Lazy(shape, dtype, read)

    def test_shapes_and_types_are_checked_without_reading_inputs(self):
        for left, right in (((0, 2, 3), (0, 4, 5)), ((2, 2, 3), (1, 3, 4)), ((2, 3), (3, 4))):
            with self.subTest(left=left, right=right), self.assertRaises(Invalid):
                L.l_bmm(self.unread(left), self.unread(right))
        with self.assertRaises(Invalid):
            L.l_mm(self.unread((2, 3)), self.unread((3, 4), "f64"))
        with self.assertRaises(NotEstablished):
            L.l_bmm(self.unread((0, 2, 3)), self.unread((0, 3, 4), "f64"))

    def test_beta_zero_validates_metadata_and_scalars_but_does_not_read_addend(self):
        a, b = L.as_lazy(A.full([2, 3], 2, "f32")), L.as_lazy(A.full([3, 4], 3, "f32"))
        for c in (self.unread((1, 2, 4)), self.unread((2, 4), "f64")):
            with self.assertRaises(Invalid):
                L.l_addmm(c, a, b, beta=0)
        c = self.unread((2, 4))
        y = L.l_addmm(c, a, b, beta=L.as_lazy(A.tensor(0, "f32")))
        self.assertEqual(y.get((1, 3)).v, 18)
        for params in ({"beta": [0]}, {"beta": 0, "alpha": [1]}, {"beta": 0, "alpha": float("inf")}):
            with self.assertRaises(Invalid):
                L.l_addmm(c, a, b, **params)

    def test_all_integer_coordinates_and_dtype_match_native(self):
        import torch
        for name, shape in (("mm", [2, 2]), ("bmm", [2, 2, 2]), ("addmm", [2, 2]), ("baddbmm", [2, 2, 2])):
            for dtype, td in (("i32", torch.int32), ("f64", torch.float64)):
                with self.subTest(op=name, dtype=dtype):
                    x = torch.arange(np.prod(shape)).reshape(shape).to(td)
                    args, kw = [x, x], {}
                    if name in ("addmm", "baddbmm"):
                        args.insert(0, torch.tensor(2, dtype=td))
                        kw = {"alpha": 3, "beta": 2}
                    expected = getattr(torch, name)(*args, **kw)
                    actual = getattr(L, "l_" + name)(*[L.as_lazy(A.tensor(v, dtype)) for v in args], **kw)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.shape, tuple(expected.shape))
                    for index in np.ndindex(actual.shape):
                        value = actual.get(index)
                        self.assertEqual(A.plain(value), expected[index].item())
                        if dtype == "i32":
                            self.assertIs(type(value), int)

    def test_integer_fractional_scaling_remains_unestablished(self):
        x = L.as_lazy(A.full([1, 1], 2, "i32"))
        with self.assertRaises(NotEstablished):
            L.l_addmm(x, x, x, alpha=0.5)


if __name__ == "__main__":
    unittest.main()
