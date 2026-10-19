import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid, NotEstablished, Unsupported


class LinearDomains(unittest.TestCase):
    def test_complete_forward_and_gradients_against_autograd(self):
        import torch

        cases = [((3,), (3,), ()), ((2, 3), (3,), ()),
                 ((2, 1, 3), (3,), None), ((3,), (2, 3), (2,)),
                 ((2, 3), (4, 3), ()), ((2, 1, 3), (4, 3), (4,)),
                 ((2, 0), (3, 0), (3,)), ((0, 3), (2, 3), (2,)),
                 ((2, 3), (0, 3), (0,)), ((0,), (0,), ()),
                 ((2, 0), (0,), ()), ((0, 3), (3,), ())]
        for xs, ws, bs in cases:
            with self.subTest(x=xs, w=ws, b=bs):
                def values(shape):
                    return ((torch.arange(math.prod(shape), dtype=torch.float64) % 5) - 2).reshape(shape)
                x, w = values(xs).requires_grad_(), values(ws).requires_grad_()
                b = None if bs is None else values(bs).requires_grad_()
                # The selected real profile is the documented formula. Native
                # linear rejects some vector-weight + scalar-bias combinations
                # on 2-D inputs, despite those shapes appearing in its docs.
                # Use an independent contraction and its autograd here.
                y = torch.einsum("...i,i->...", x, w) if w.ndim == 1 else torch.einsum("...i,oi->...o", x, w)
                if b is not None:
                    y = y + b
                g = values(tuple(y.shape))
                inputs = (x, w) if b is None else (x, w, b)
                grads = torch.autograd.grad(y, inputs, g)
                rx, rw = A.tensor(x, "f64"), A.tensor(w, "f64")
                rb = None if b is None else A.tensor(b, "f64")
                actual = [A.linear(rx, rw, rb), *A.vjp_linear(A.tensor(g, "f64"), rx, rw, rb)]
                expected = [y, *grads] + ([None] if b is None else [])
                self.assertEqual(len(actual), len(expected))
                for result, target in zip(actual, expected):
                    if target is None:
                        self.assertIsNone(result)
                    else:
                        self.assertEqual(result.dtype, "f64")
                        self.assertEqual(result.shape, tuple(target.shape))
                        self.assertEqual(A.plain(result), target.detach().tolist())

    def test_upstream_shape_and_type_are_validated(self):
        x, w = A.full([2, 3], 1, "f32"), A.full([4, 3], 1, "f32")
        for g in (A.full([1, 2, 4], 1, "f32"), A.full([2, 4], 1, "f64")):
            with self.assertRaises(Invalid):
                A.vjp_linear(g, x, w)
        with self.assertRaises(Unsupported):
            A.vjp_linear(A.full([2, 4], 1, "i32"), A.full([2, 3], 1, "i32"), A.full([4, 3], 1, "i32"))

    def test_bias_cannot_enlarge_output_and_mixed_dtype_is_unresolved(self):
        x, w = A.full([2, 3], 1, "f32"), A.full([4, 3], 1, "f32")
        for op, args in ((A.linear, [x, w]), (A.vjp_linear, [A.full([2, 4], 1, "f32"), x, w])):
            with self.assertRaises(Invalid):
                op(*args, A.full([1, 2, 4], 1, "f32"))
            with self.assertRaises(NotEstablished):
                op(*args, A.full([4], 1, "f64"))
            with self.assertRaises(NotEstablished):
                op(*args, A.full([1, 4], 1, "f32"))

    def test_ranks_are_checked_before_empty_computation(self):
        for xs, ws in (([], [1]), ([0, 3], []), ([0, 3], [1, 2, 3])):
            with self.subTest(x=xs, w=ws), self.assertRaises(Invalid):
                A.linear(A.full(xs, 1, "f32"), A.full(ws, 1, "f32"))


if __name__ == "__main__":
    unittest.main()
