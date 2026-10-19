"""Admission boundary of the finite floating REDUCE profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


def tensor(values, shape=None, dtype="f32"):
    out = {"t": dtype, "v": values}
    if shape is not None:
        out["shape"] = shape
    return out


class ReductionProfileDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("REDUCE").adapter

    def outcome(self, call):
        return evaluate(self.adapter.reference, "real", call).outcome

    def test_only_finite_binary_floating_tensors_are_admitted(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            with self.subTest(dtype=dtype):
                self.assertEqual(self.outcome({"op": "aten.sum", "args": [tensor([1, 2], dtype=dtype)]}), "value")
        for dtype in ("i32", "b8", "f8e4m3", "f8e5m2", "complex64"):
            with self.subTest(dtype=dtype):
                self.assertEqual(self.outcome({"op": "aten.sum", "args": [tensor([1, 2], dtype=dtype)]}), "unsupported")
        for value, outcome in (("inf", "unsupported"), (float("inf"), "unsupported"),
                               ("-inf", "unsupported"), (float("nan"), "invalid_input")):
            with self.subTest(value=value):
                self.assertEqual(self.outcome({"op": "aten.sum", "args": [tensor([1, value])]}), outcome)

    def test_retained_default_schemas_are_enforced(self):
        x = tensor([[1, 2], [3, 4]])
        calls = [
            {"op": "aten.sum", "args": [x, [1], False, "f64"]},
            {"op": "aten.mean", "args": [x, [1], False, "f64"]},
            {"op": "aten.prod", "args": [x, [0, 1]]},
            {"op": "aten.prod", "args": [x, 1, False, "f64"]},
            {"op": "aten.logsumexp", "args": [x, None]},
            {"op": "aten.logsumexp", "args": [x, []]},
        ]
        for call in calls:
            with self.subTest(call=call):
                self.assertEqual(self.outcome(call), "invalid_input")
        self.assertEqual(self.outcome({"op": "aten.sum", "args": [x], "kwargs": {"dtype": "i64"}}),
                         "unsupported")
        self.assertEqual(self.outcome({"op": "aten.sum", "args": [x],
                                      "kwargs": {"dtype": "torch.float16"}}), "value")

    def test_vjp_requires_matching_complete_upstream(self):
        call = {"op": "vjp.aten.sum", "args": [tensor([1], dtype="f64"), tensor([[1, 2]]), 1]}
        self.assertEqual(self.outcome(call), "invalid_input")
        call["args"][0] = tensor([1, 2])
        self.assertEqual(self.outcome(call), "invalid_input")

    def test_nonboolean_keepdim_is_rejected(self):
        for value in (1, "false", None):
            call = {"op": "aten.sum", "args": [tensor([[1, 2]]), [1], value]}
            with self.subTest(value=value):
                self.assertEqual(self.outcome(call), "invalid_input")

    def test_selected_variance_correction_and_domain_are_explicit(self):
        x = tensor([[1, 3], [2, 6]])
        for correction in (0, 1, None):
            call = {"op": "aten.var", "args": [x, 1], "kwargs": {"correction": correction}}
            with self.subTest(correction=correction):
                self.assertEqual(self.outcome(call), "value")
        for correction in (-1, 2, 0.5):
            call = {"op": "aten.var", "args": [x, 1], "kwargs": {"correction": correction}}
            with self.subTest(correction=correction):
                self.assertEqual(self.outcome(call), "unsupported")
        for correction in (True, "one"):
            call = {"op": "aten.var", "args": [x, 1], "kwargs": {"correction": correction}}
            with self.subTest(correction=correction):
                self.assertEqual(self.outcome(call), "invalid_input")
        self.assertEqual(self.outcome({"op": "aten.var", "args": [x, []],
                                      "kwargs": {"correction": 0}}), "unsupported")
        self.assertEqual(self.outcome({"op": "aten.var", "args": [tensor([[2]]), 1],
                                      "kwargs": {"correction": 1}}), "domain_undefined")

    def test_variance_vjp_requires_the_complete_matching_upstream(self):
        x = tensor([[1, 3], [2, 6]])
        valid = {"op": "vjp.aten.var", "args": [tensor([3, -2]), x, 1],
                 "kwargs": {"correction": 1}}
        self.assertEqual(self.outcome(valid), "value")
        wrong_shape = {**valid, "args": [tensor([[3, -2]]), x, 1]}
        wrong_dtype = {**valid, "args": [tensor([3, -2], dtype="f64"), x, 1]}
        self.assertEqual(self.outcome(wrong_shape), "invalid_input")
        self.assertEqual(self.outcome(wrong_dtype), "invalid_input")
        for op in ("aten.norm", "vjp.aten.norm"):
            with self.subTest(op=op):
                self.assertEqual(self.outcome({"op": op, "args": [x, 1]}), "unsupported")

    def test_selected_public_vector_norm_domain_is_explicit(self):
        x = tensor([[3, 4], [5, 12]])
        for dim in (None, 1, [1], (1,)):
            with self.subTest(dim=dim):
                self.assertEqual(self.outcome({
                    "op": "aten.linalg_vector_norm", "args": [x, 2, dim]
                }), "value")
        valid_vjp = {
            "op": "vjp.aten.linalg_vector_norm",
            "args": [tensor([2, -3]), x, 2, 1],
        }
        self.assertEqual(self.outcome(valid_vjp), "value")
        self.assertEqual(self.outcome({
            "op": "vjp.aten.linalg_vector_norm",
            "args": [tensor([1]), tensor([[0, 0]]), 2, 1],
        }), "domain_undefined")
        for call in (
            {"op": "aten.linalg_vector_norm", "args": [x, 1, 1]},
            {"op": "aten.linalg_vector_norm", "args": [x, 2, []]},
            {"op": "aten.linalg_vector_norm", "args": [tensor([3, 4], dtype="bf16"), 2]},
            {"op": "aten.linalg_vector_norm", "args": [x, 2],
             "kwargs": {"dtype": "f64"}},
        ):
            with self.subTest(call=call):
                self.assertEqual(self.outcome(call), "unsupported")
        wrong_upstream = {**valid_vjp, "args": [tensor([[2, -3]]), x, 2, 1]}
        wrong_dtype = {**valid_vjp,
                       "args": [tensor([2, -3], dtype="f64"), x, 2, 1]}
        self.assertEqual(self.outcome(wrong_upstream), "invalid_input")
        self.assertEqual(self.outcome(wrong_dtype), "invalid_input")

    def test_standard_deviation_has_an_explicit_smooth_vjp_domain(self):
        x = tensor([[1, 3], [2, 6]])
        for correction in (0, 1, None):
            call = {"op": "aten.std", "args": [x, 1],
                    "kwargs": {"correction": correction}}
            with self.subTest(correction=correction):
                self.assertEqual(self.outcome(call), "value")
        self.assertEqual(self.outcome({
            "op": "vjp.aten.std", "args": [tensor([3, -2]), x, 1],
            "kwargs": {"correction": 1},
        }), "value")
        self.assertEqual(self.outcome({
            "op": "vjp.aten.std", "args": [tensor([1]), tensor([[2, 2]]), 1],
            "kwargs": {"correction": 0},
        }), "domain_undefined")
        self.assertEqual(self.outcome({
            "op": "vjp.aten.std", "args": [tensor([7]), tensor([[2]]), 1],
            "kwargs": {"correction": 0},
        }), "value")
        for correction in (-1, 2, 0.5):
            with self.subTest(correction=correction):
                self.assertEqual(self.outcome({
                    "op": "aten.std", "args": [x, 1],
                    "kwargs": {"correction": correction},
                }), "unsupported")

    def test_var_mean_requires_both_outputs_and_both_matching_upstreams(self):
        x = tensor([[1, 3], [2, 6]])
        for correction in (0, 1, None):
            with self.subTest(correction=correction):
                self.assertEqual(self.outcome({
                    "op": "aten.var_mean", "args": [x, 1],
                    "kwargs": {"correction": correction},
                }), "value")
        valid = {
            "op": "vjp.aten.var_mean",
            "args": [tensor([3, -2]), tensor([5, 7]), x, 1],
            "kwargs": {"correction": 1},
        }
        self.assertEqual(self.outcome(valid), "value")
        wrong_var_shape = {**valid, "args": [tensor([[3, -2]]), tensor([5, 7]), x, 1]}
        wrong_mean_shape = {**valid, "args": [tensor([3, -2]), tensor([[5, 7]]), x, 1]}
        wrong_mean_dtype = {**valid, "args": [tensor([3, -2]),
                                               tensor([5, 7], dtype="f64"), x, 1]}
        for call in (wrong_var_shape, wrong_mean_shape, wrong_mean_dtype):
            with self.subTest(call=call):
                self.assertEqual(self.outcome(call), "invalid_input")
        self.assertEqual(self.outcome({
            "op": "aten.var_mean", "args": [x, []],
            "kwargs": {"correction": 0},
        }), "unsupported")


if __name__ == "__main__":
    unittest.main()
