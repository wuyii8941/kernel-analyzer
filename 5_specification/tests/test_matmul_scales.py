import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid


class MatmulScales(unittest.TestCase):
    def test_integer_noninteger_scaling_waits_for_a_task_choice(self):
        from kernel_spec.num import NotEstablished
        for op, shape in ((A.addmm, [1, 1]), (A.baddbmm, [1, 1, 1])):
            x = A.full(shape, 2, "i64")
            for key in ("alpha", "beta"):
                with self.subTest(op=op.__name__, key=key), self.assertRaises(NotEstablished):
                    op(x, x, x, **{key: Q(1, 2)})

    def test_scales_are_scalar_even_for_empty_results_and_zero_beta(self):
        for op, shape in ((A.addmm, [1, 1]), (A.baddbmm, [1, 1, 1]),
                          (A.addmm, [0, 0]), (A.baddbmm, [0, 1, 1])):
            x = A.full(shape, 1, "f32")
            for bad in ([1], {"value": 1}, A.tensor([1, 2], "f32"), A.tensor([[1]], "f32"), None):
                for key in ("alpha", "beta"):
                    with self.subTest(op=op.__name__, shape=shape, key=key, bad=bad), self.assertRaises(Invalid):
                        op(x, x, x, **dict({"beta": 0}, **{key: bad}))

    def test_scalar_zero_beta_ignores_infinite_values_but_not_input_shape(self):
        for op, values in ((A.addmm, [[2]]), (A.baddbmm, [[[2]]])):
            x = A.tensor(values, "f32")
            c = A.full(x.shape, "+inf", "f32")
            for beta in (0, False, A.R(0), A.tensor(0, "f32"), A.tensor(0, "i64")):
                result = op(c, x, x, beta=beta)
                self.assertEqual(A.plain(result), [[4]] if op is A.addmm else [[[4]]])

    def test_rational_and_zero_dimensional_scales_keep_exact_values(self):
        x = A.tensor([[2]], "f64")
        result = A.addmm(x, x, x, alpha=Q(1, 3), beta=A.tensor(Q(-1, 2), "f64"))
        self.assertEqual(A.plain(result), [[Q(1, 3)]])
        self.assertEqual(result.dtype, "f64")
        self.assertEqual(A.plain(A.addmm(x, x, x, alpha=False, beta=True)), [[2]])

    def test_addmm_vjp_uses_the_same_scalar_parameter_rules(self):
        x = A.tensor([[2]], "f32")
        with self.assertRaises(Invalid):
            A.vjp_addmm(x, x, x, x, alpha=A.tensor([1, 2], "f32"))
        dc, da, db = A.vjp_addmm(x, x, x, x, beta=A.tensor(0, "f32"), alpha=A.tensor(3, "f32"))
        self.assertEqual([A.plain(t) for t in (dc, da, db)], [[[0]], [[12]], [[12]]])

    def test_unknown_scalar_does_not_guess_a_zero_branch(self):
        from kernel_spec.num import NotEstablished, hull
        x = A.tensor([[2]], "f32")
        with self.assertRaises(NotEstablished):
            A.addmm(x, x, x, beta=A.R(hull(0, 1)))
        self.assertEqual(A.plain(A.addmm(A.tensor([["+inf"]], "f32"), x, x,
                                        beta=A.R(hull(0, 0)))), [[4]])


if __name__ == "__main__":
    unittest.main()
