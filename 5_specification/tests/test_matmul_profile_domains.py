"""Admission boundaries of the finite floating MATMUL contract profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


def tensor(value, shape, dtype="f32"):
    return {"t": dtype, "v": value, "shape": shape}


class MatmulProfileDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("MATMUL").adapter

    def outcome(self, call):
        return evaluate(self.adapter.reference, "real", call).outcome

    def test_nonfloating_and_float8_operands_are_outside_the_profile(self):
        for dtype in ("i32", "b8", "f8e4m3", "f8e5m2", "complex64"):
            call = {"op": "aten.mm", "args": [tensor([[1]], [1, 1], dtype), tensor([[2]], [1, 1], dtype)]}
            with self.subTest(dtype=dtype):
                self.assertEqual(self.outcome(call), "unsupported")

    def test_infinite_operands_are_unsupported_and_nan_is_invalid(self):
        for value, expected in (("inf", "unsupported"), (float("inf"), "unsupported"),
                                ("-inf", "unsupported"), (float("nan"), "invalid_input")):
            call = {"op": "aten.mm", "args": [tensor([[value]], [1, 1]), tensor([[2]], [1, 1])]}
            with self.subTest(value=value):
                self.assertEqual(self.outcome(call), expected)

    def test_beta_zero_is_the_only_nonfinite_addend_exception(self):
        product = [tensor([[2]], [1, 1]), tensor([[3]], [1, 1])]
        for value in ("nan", float("nan"), "inf", float("inf"), "-inf"):
            call = {"op": "aten.addmm", "args": [tensor([[value]], [1, 1]), *product],
                    "kwargs": {"beta": 0}}
            with self.subTest(value=value):
                self.assertEqual(self.outcome(call), "value")
                self.assertTrue(self.adapter.check_output("real", call,
                                                           {"out": tensor([[6]], [1, 1])}))
            call["kwargs"]["beta"] = 1
            self.assertIn(self.outcome(call), ("invalid_input", "unsupported"))

    def test_default_aten_overload_does_not_accept_dtype_or_positional_scales(self):
        args = [tensor([[0]], [1, 1]), tensor([[2]], [1, 1]), tensor([[3]], [1, 1])]
        self.assertEqual(self.outcome({"op": "aten.addmm", "args": [*args, 0]}), "invalid_input")
        self.assertEqual(self.outcome({"op": "aten.addmm", "args": args,
                                      "kwargs": {"out_dtype": "torch.float32"}}), "unsupported")
        self.assertEqual(self.outcome({"op": "aten.addmm.dtype", "args": args}), "unsupported")


if __name__ == "__main__":
    unittest.main()
