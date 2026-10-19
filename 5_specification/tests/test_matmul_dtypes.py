import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid, Unsupported, NotEstablished


class MatmulDtypes(unittest.TestCase):
    def test_complete_small_products_against_native_execution(self):
        import math
        import torch

        # Small integers are exact in every tested dtype; this checks metadata
        # and values without treating native rounding or overflow as the target.
        dtypes = (("i8", torch.int8), ("i32", torch.int32), ("i64", torch.int64),
                  ("f16", torch.float16), ("bf16", torch.bfloat16),
                  ("f32", torch.float32), ("f64", torch.float64))
        cases = (("mm", (2, 3), (3, 4)), ("mm", (2, 0), (0, 3)),
                 ("bmm", (2, 2, 3), (2, 3, 4)), ("bmm", (0, 2, 3), (0, 3, 4)),
                 ("matmul", (3,), (3,)), ("matmul", (0,), (0,)),
                 ("addmm", (2, 3), (3, 4)), ("baddbmm", (2, 2, 3), (2, 3, 4)))
        for dtype, native_dtype in dtypes:
            for name, left, right in cases:
                with self.subTest(dtype=dtype, op=name, left=left):
                    def values(shape):
                        return ((torch.arange(math.prod(shape)) % 3) - 1).reshape(shape).to(native_dtype)
                    a, b = values(left), values(right)
                    args, kwargs = [a, b], {}
                    if name in ("addmm", "baddbmm"):
                        args.insert(0, torch.tensor(2, dtype=native_dtype))
                        kwargs = {"alpha": 3, "beta": 2}
                    expected = getattr(torch, name)(*args, **kwargs)
                    actual = getattr(A, name)(*[A.tensor(x, dtype) for x in args], **kwargs)
                    self.assertEqual(expected.dtype, native_dtype)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.shape, tuple(expected.shape))
                    self.assertEqual(A.plain(actual), expected.tolist())

    def test_integer_products_retain_type_and_integer_storage(self):
        for dtype in ("i8", "i32", "i64"):
            for op, shape in ((A.mm, [2, 2]), (A.bmm, [1, 2, 2]), (A.matmul, [2])):
                x = A.full(shape, 2, dtype)
                result = op(x, x)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(all(type(value) is int and value == 8 for value in result.a.flat))
            empty = A.mm(A.full([2, 0], 0, dtype), A.full([0, 3], 0, dtype))
            self.assertEqual(empty.dtype, dtype)
            self.assertTrue(all(type(value) is int and value == 0 for value in empty.a.flat))

    def test_mixed_operand_types_are_not_silently_promoted(self):
        for op, shape in ((A.mm, [1, 1]), (A.bmm, [1, 1, 1]), (A.matmul, [0])):
            for left, right in (("f32", "f64"), ("i32", "f32"), ("i32", "i64")):
                with self.subTest(op=op.__name__, left=left, right=right), self.assertRaises(Invalid):
                    op(A.full(shape, 1, left), A.full(shape, 1, right))
        # Native empty-batch behavior differs from the nonempty dtype rejection.
        for op in (A.bmm, A.matmul):
            with self.assertRaises(NotEstablished):
                op(A.full([0, 1, 1], 1, "f32"), A.full([0, 1, 1], 1, "f64"))

    def test_zero_beta_does_not_erase_addend_type(self):
        for op, shape in ((A.addmm, [1, 1]), (A.baddbmm, [0, 1, 1])):
            x = A.full(shape, 1, "f32")
            with self.assertRaises(Invalid):
                op(A.full(shape, 1, "f64"), x, x, beta=0)

    def test_integral_scaling_does_not_promote_integer_results(self):
        for op, shape in ((A.addmm, [1, 1]), (A.baddbmm, [1, 1, 1])):
            x = A.full(shape, 2, "i32")
            result = op(x, x, x, beta=2, alpha=3)
            self.assertEqual(result.dtype, "i32")
            self.assertTrue(all(type(value) is int and value == 16 for value in result.a.flat))

    def test_vjp_cannot_silently_cast_mismatched_or_integer_inputs(self):
        x = A.full([1, 1], 1, "f32")
        with self.assertRaises(Invalid):
            A.vjp_mm(A.full([1, 1], 1, "f64"), x, x)
        y = A.full([1, 1], 1, "i64")
        with self.assertRaises(Unsupported):
            A.vjp_mm(y, y, y)


if __name__ == "__main__":
    unittest.main()
