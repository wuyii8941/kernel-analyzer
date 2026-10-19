import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid


def zeros(shape):
    return A.full(shape, 0, "f32")


class MatmulDomains(unittest.TestCase):
    def test_empty_batches_do_not_bypass_inner_dimension_checks(self):
        for op in (A.bmm, A.matmul):
            with self.subTest(op=op.__name__), self.assertRaises(Invalid):
                op(zeros([0, 2, 3]), zeros([0, 4, 5]))
        self.assertEqual(A.bmm(zeros([0, 2, 3]), zeros([0, 3, 5])).shape, (0, 2, 5))
        self.assertEqual(A.matmul(zeros([0, 2, 3]), zeros([1, 3, 5])).shape, (0, 2, 5))

    def test_scalar_and_nonbroadcastable_batch_inputs_are_invalid(self):
        for a, b in ((zeros([]), zeros([2])), (zeros([2]), zeros([])),
                     (zeros([2, 2, 3]), zeros([3, 3, 4]))):
            with self.subTest(a=a.shape, b=b.shape), self.assertRaises(Invalid):
                A.matmul(a, b)

    def test_addend_must_broadcast_to_the_fixed_product_shape_even_at_beta_zero(self):
        for op, a, b, bad in ((A.addmm, zeros([2, 3]), zeros([3, 4]), ([1, 2, 4], [5])),
                              (A.baddbmm, zeros([2, 3, 4]), zeros([2, 4, 5]), ([1, 2, 3, 5], [4]))):
            for beta in (0, 1):
                for shape in bad:
                    with self.subTest(op=op.__name__, beta=beta, shape=shape), self.assertRaises(Invalid):
                        op(zeros(shape), a, b, beta=beta)
        self.assertEqual(A.addmm(zeros([]), zeros([2, 0]), zeros([0, 4]), beta=0).shape, (2, 4))

    def test_dot_product_vjp_has_both_complete_vector_gradients(self):
        a, b, g = A.tensor([1, 2, 3], "f32"), A.tensor([4, 5, 6], "f32"), A.tensor(2, "f32")
        da, db = A.vjp_matmul(g, a, b)
        self.assertEqual(A.plain(da), [8, 10, 12])
        self.assertEqual(A.plain(db), [2, 4, 6])

    def test_vjp_upstream_cannot_broadcast_to_a_different_forward_output(self):
        a, b = zeros([2, 3]), zeros([3, 4])
        with self.assertRaises(Invalid):
            A.vjp_matmul(zeros([1, 2, 4]), a, b)
        with self.assertRaises(Invalid):
            A.vjp_matmul(zeros([1]), zeros([3]), zeros([3]))
        for op, a, b, g in ((A.vjp_mm, zeros([2, 3]), zeros([4, 5]), zeros([2, 5])),
                             (A.vjp_bmm, zeros([0, 2, 3]), zeros([0, 4, 5]), zeros([0, 2, 5]))):
            with self.subTest(op=op.__name__), self.assertRaises(Invalid):
                op(g, a, b)

    def test_complete_products_and_vjps_against_independent_autograd(self):
        import math
        import torch

        def values(shape):
            return ((torch.arange(math.prod(shape), dtype=torch.float64) % 5) - 2).reshape(shape)

        shapes = [((3,), (3,)), ((3,), (3, 2)), ((2, 3), (3,)),
                  ((3,), (2, 3, 4)), ((2, 4, 3), (3,)), ((2, 3), (3, 4)),
                  ((2, 1, 3, 4), (3, 4, 2)), ((3, 4), (2, 4, 5)),
                  ((0,), (0,)), ((2, 0), (0, 3)), ((0, 2, 3), (1, 3, 4)),
                  ((0, 1, 2, 3), (2, 3, 4))]
        for sa, sb in shapes:
            with self.subTest(a=sa, b=sb):
                a, b = values(sa).requires_grad_(), values(sb).requires_grad_()
                y = torch.matmul(a, b)
                g = values(tuple(y.shape))
                da, db = torch.autograd.grad(y, (a, b), g)
                ra, rb = A.tensor(a, "f64"), A.tensor(b, "f64")
                actual = [A.matmul(ra, rb), *A.vjp_matmul(A.tensor(g, "f64"), ra, rb)]
                for result, expected in zip(actual, (y, da, db)):
                    self.assertEqual(result.shape, tuple(expected.shape))
                    self.assertEqual(A.plain(result), expected.detach().tolist())


if __name__ == "__main__":
    unittest.main()
