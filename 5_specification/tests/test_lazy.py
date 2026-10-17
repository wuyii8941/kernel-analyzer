"""kernel_spec.lazy must agree with kernel_spec.aten (the contracts' reference) element by element: every operation of
the lazy evaluator is run on small random tensors and compared exactly with the eager result."""
import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernel_spec import aten as A  # noqa: E402
from kernel_spec import lazy as L  # noqa: E402
from kernel_spec.num import ends  # noqa: E402

rng = random.Random(20261016)


def rnd(shape, lo=-4, hi=4, pos=False):
    vals = np.array([Fraction(rng.randint(1 if pos else lo, hi), rng.choice([1, 2, 4])) for _ in range(int(np.prod(shape)))],
                    dtype=object).reshape(shape)
    return A.tensor(vals.tolist() if shape else vals[()], "f32")


def same(x, y):
    if isinstance(x, A.AllowedSet) or isinstance(y, A.AllowedSet):
        return isinstance(x, A.AllowedSet) and isinstance(y, A.AllowedSet)
    if isinstance(x, A.R) and isinstance(y, A.R):
        if x.exact and y.exact:
            return x.v == y.v
        (a0, a1), (b0, b1) = x.bounds(), y.bounds()
        return a0 <= b1 and b0 <= a1
    return x == y


def check(test, eager, lazy):
    eager = eager if isinstance(eager, (list, tuple)) else [eager]
    lazy = lazy if isinstance(lazy, (list, tuple)) else [lazy]
    for e, l in zip(eager, lazy):
        test.assertEqual(tuple(e.shape), tuple(l.shape))
        for i in np.ndindex(*e.shape):
            test.assertTrue(same(e.a[i], l.get(i)), f"{i}: {e.a[i]} vs {l.get(i)}")


class LazyMatchesAten(unittest.TestCase):
    def test_elementwise(self):
        x, y = rnd((2, 3)), rnd((3,))
        p = rnd((2, 3), pos=True)
        for op in ("aten.add", "aten.sub", "aten.mul", "aten.maximum", "aten.minimum", "aten.eq", "aten.lt", "aten.ge"):
            check(self, A.OPS[op](x, y), L.OPS[op](x, y))
        check(self, A.div(x, p), L.OPS["aten.div"](x, p))
        check(self, A.div(x, p, rounding_mode="floor"), L.OPS["aten.div"](x, p, rounding_mode="floor"))
        for op in ("aten.exp", "aten.tanh", "aten.sigmoid", "aten.erf", "aten.sin", "aten.cos", "aten.neg", "aten.abs",
                   "aten.sign", "aten.floor", "aten.ceil", "aten.relu"):
            check(self, A.OPS[op](x), L.OPS[op](x))
        for op in ("aten.log", "aten.sqrt", "aten.rsqrt", "aten.reciprocal", "aten.log1p"):
            check(self, A.OPS[op](p), L.OPS[op](p))
        check(self, A.pow_(x, 2), L.OPS["aten.pow"](x, 2))
        check(self, A.where(A.gt(x, 0), x, y), L.OPS["aten.where"](A.gt(x, 0), x, y))
        check(self, A.convert_element_type(A.div(x, 3), "bf16"), L.OPS["prims.convert_element_type"](A.div(x, 3), "bf16"))
        check(self, A.fma(x, y, 1), L.OPS["prims.fma"](x, y, 1))

    def test_layout(self):
        x = rnd((2, 3, 4))
        cases = [("aten.view", (x, [6, 4])), ("aten.permute", (x, [2, 0, 1])), ("aten.transpose", (x, 0, 2)),
                 ("aten.expand", (rnd((1, 4)), [3, 4])), ("aten.unsqueeze", (x, 1)), ("aten.squeeze", (rnd((1, 3, 1)),)),
                 ("aten.select", (x, 1, -1)), ("aten.slice", (x, 2, 1, 4, 2)), ("aten.constant_pad_nd", (x, [1, 2], 0)),
                 ("aten.cat", ([x, x], 1)), ("aten.split_with_sizes", (x, [1, 3], 2)),
                 ("aten.select_scatter", (x, rnd((2, 4)), 1, 0)), ("aten.slice_scatter", (x, rnd((2, 3, 2)), 2, 0, 4, 2)),
                 ("aten.full", ([2, 2], Fraction(1, 3))), ("prims.iota", (5,))]
        for op, args in cases:
            check(self, A.OPS[op](*args), L.OPS[op](*args))

    def test_indexing(self):
        x = rnd((4, 5))
        idx = A.tensor([[0, 3], [1, 1]], "i64")
        check(self, A.gather(x, 1, idx), L.OPS["aten.gather"](x, 1, idx))
        check(self, A.index(x, [None, A.tensor([4, 0, 4], "i64")]), L.OPS["aten.index"](x, [None, A.tensor([4, 0, 4], "i64")]))
        check(self, A.index(x, [A.tensor([1, 2], "i64")]), L.OPS["aten.index"](x, [A.tensor([1, 2], "i64")]))
        src = A.tensor([[1, 2], [3, 4]], "f32")
        check(self, A.scatter_add(x, 1, idx, src), L.OPS["aten.scatter_add"](x, 1, idx, src))
        v = rnd((3, 5))
        check(self, A.index_put(x, [A.tensor([0, 0, 2], "i64")], v, True),
              L.OPS["aten.index_put"](x, [A.tensor([0, 0, 2], "i64")], v, True))
        check(self, A.embedding(x, A.tensor([3, 0], "i64")), L.OPS["aten.embedding"](x, A.tensor([3, 0], "i64")))

    def test_reductions(self):
        x = rnd((3, 4))
        for op, args in [("aten.sum", (x, [1])), ("aten.sum", (x, None)), ("aten.mean", (x, [0], True)),
                         ("aten.amax", (x, [1])), ("aten.amin", (x, [0, 1])), ("aten.var_mean", (x, [1])),
                         ("aten.max", (x, 1)), ("aten.cumsum", (x, 1)), ("prims.prepare_softmax_online", (x, 1)),
                         ("aten.any", (A.gt(x, 2), [1]))]:
            check(self, A.OPS[op](*args), L.OPS[op](*args))

    def test_products(self):
        a, b = rnd((3, 4)), rnd((4, 2))
        check(self, A.mm(a, b), L.OPS["aten.mm"](a, b))
        c = rnd((2,))
        check(self, A.addmm(c, a, b, beta=2, alpha=3), L.OPS["aten.addmm"](c, a, b, beta=2, alpha=3))
        p, q = rnd((2, 3, 4)), rnd((2, 4, 2))
        check(self, A.bmm(p, q), L.OPS["aten.bmm"](p, q))
        check(self, A.mm_plus_mm(a, b, a, b), L.OPS["inductor.mm_plus_mm"](a, b, a, b))
        x, w = rnd((1, 2, 5)), rnd((2, 1, 3))
        check(self, A.convolution(x, w, None, [1], [1], [1], False, [0], 2),
              L.OPS["aten.convolution"](x, w, None, [1], [1], [1], False, [0], 2))


if __name__ == "__main__":
    unittest.main()
