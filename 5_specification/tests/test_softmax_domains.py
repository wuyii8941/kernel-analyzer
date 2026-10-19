import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid, Undefined


class SoftmaxDomains(unittest.TestCase):
    def test_infinite_rows_follow_the_declared_domain(self):
        for op in (A.softmax, A.log_softmax, A.safe_softmax, A.prepare_softmax_online):
            for values in ([0, "+inf"], ["+inf", "+inf"]):
                with self.subTest(op=op.__name__, values=values), self.assertRaises(Undefined):
                    op(A.tensor(values, "f32"), 0)
        for op in (A.softmax, A.log_softmax, A.prepare_softmax_online):
            with self.subTest(op=op.__name__), self.assertRaises(Undefined):
                op(A.tensor(["-inf", "-inf"], "f32"), 0)

    def test_nan_input_is_invalid_not_an_undefined_formula(self):
        from kernel_spec.aten_adapter import make_reference
        from kernel_spec.num import evaluate
        for op in ("aten._softmax", "aten._log_softmax", "aten._safe_softmax",
                   "prims.prepare_softmax_online"):
            reference = make_reference([op])
            args = [{"t": "f32", "v": [float("nan")]}, 0]
            if op in ("aten._softmax", "aten._log_softmax"):
                args.append(False)
            result = evaluate(reference, "real", {"op": op, "args": args})
            self.assertEqual(result.outcome, "invalid_input")

    def test_axes_are_validated_before_row_evaluation(self):
        for op in (A.softmax, A.log_softmax, A.safe_softmax):
            for values in ([0, 0], []):
                for dim in (1, -2, True, 0.0):
                    with self.subTest(op=op.__name__, values=values, dim=dim), self.assertRaises(Invalid):
                        op(A.tensor(values, "f32"), dim)

    def test_scalar_is_a_single_entry_row(self):
        for op, expected in ((A.softmax, 1), (A.log_softmax, 0), (A.safe_softmax, 1)):
            for dim in (0, -1):
                self.assertEqual(A.plain(op(A.tensor(7, "f32"), dim)), expected)
            with self.assertRaises(Invalid):
                op(A.tensor(7, "f32"), 1)

    def test_empty_output_shapes_are_preserved(self):
        for shape in ([0, 3], [2, 0]):
            x = A.view(A.tensor([], "f32"), shape)
            for op in (A.softmax, A.log_softmax, A.safe_softmax):
                for dim in (0, 1):
                    self.assertEqual(op(x, dim).shape, tuple(shape))

    def test_row_result_cannot_be_truncated_or_left_unwritten(self):
        x = A.tensor([0, 0], "f32")
        for count in (0, 1, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                A._rowwise(x, 0, lambda row: [A.R(0)] * count)

    def test_half_to_float_validates_input_and_preserves_real_values(self):
        for op in (A.softmax, A.log_softmax):
            self.assertEqual(op(A.tensor([0, 0], "f16"), 0, True).dtype, "f32")
            self.assertEqual(op(A.tensor([], "f16"), 0, True).dtype, "f32")
            for dtype in ("f32", "bf16", "i64", "b8"):
                with self.subTest(op=op.__name__, dtype=dtype), self.assertRaises(Invalid):
                    op(A.tensor([], dtype), 0, True)
            with self.assertRaises(Invalid):
                op(A.tensor([0], "f16"), 0, "false")

    def test_integer_input_requires_an_explicit_floating_conversion(self):
        for op in (A.softmax, A.log_softmax, A.safe_softmax):
            for values in ([0, 0], []):
                with self.subTest(op=op.__name__), self.assertRaises(Invalid):
                    op(A.tensor(values, "i64"), 0)
        result = A.safe_softmax(A.tensor([0, 0], "i64"), 0, "f64")
        self.assertEqual(result.dtype, "f64")
        self.assertEqual(A.plain(result), [A.Fraction(1, 2)] * 2)

    def test_safe_softmax_cast_precedes_the_formula(self):
        x = A.tensor([1, A.Fraction(4097, 4096)], "f64")
        result = A.safe_softmax(x, 0, "f16")
        self.assertEqual(result.dtype, "f16")
        self.assertEqual(A.plain(result), [A.Fraction(1, 2)] * 2)
        with self.assertRaises(Invalid):
            A.safe_softmax(A.tensor([], "f32"), 0, "i64")
        # The mask is computed from the original input, before conversion.
        from kernel_spec.num import Undefined
        with self.assertRaises(Undefined):
            A.safe_softmax(A.tensor([-100000, -100000], "f64"), 0, "f16")
        self.assertEqual(A.plain(A.safe_softmax(A.tensor(["-inf", "-inf"], "f64"), 0, "f16")), [0, 0])


if __name__ == "__main__":
    unittest.main()
