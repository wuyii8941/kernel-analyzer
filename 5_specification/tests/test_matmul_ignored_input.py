import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, NotEstablished, Unsupported
from kernel_spec.real import R
from kernel_spec.num import hull


def tensor(values, shape, dtype="f32"):
    return {"t": dtype, "v": values, "shape": shape}


class IgnoredInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("MATMUL").adapter

    def test_documented_nan_ignored_for_both_products(self):
        import torch
        for op, shape in (("addmm", [1, 1]), ("baddbmm", [1, 1, 1])):
            for nan in (float("nan"), "nan"):
                for beta in (0, tensor(0, [])):
                    with self.subTest(op=op, nan=nan, beta=beta):
                        call = {"op": "aten." + op, "args": [tensor(nan, []), tensor([2], shape), tensor([3], shape)],
                                "kwargs": {"beta": beta}}
                        expected = getattr(torch, op)(torch.tensor(float("nan")), torch.full(shape, 2.),
                                                      torch.full(shape, 3.), beta=0)
                        self.assertTrue(self.adapter.check_output("real", call, {"out": tensor(expected.tolist(), shape)}))
                        self.assertFalse(self.adapter.check_output("real", call, {"out": tensor(float("nan"), shape)}))

    def test_vjp_preserves_all_gradients_with_nan_addend(self):
        call = {"op": "vjp.aten.addmm", "kwargs": {"g": tensor([[2]], [1, 1]), "c": tensor("nan", []),
                "a": tensor([[3]], [1, 1]), "b": tensor([[4]], [1, 1]), "beta": 0}}
        output = [tensor(0, []), tensor([[8]], [1, 1]), tensor([[6]], [1, 1])]
        self.assertTrue(self.adapter.check_output("real", call, {"out": output}))

    def test_empty_outputs_obey_beta_zero_and_positional_beta_is_rejected(self):
        for op, left, right, output in (("addmm", [0, 2], [2, 3], [0, 3]),
                                        ("baddbmm", [0, 2, 3], [0, 3, 4], [0, 2, 4])):
            args = [tensor("nan", []), tensor([], left),
                    tensor([1] * 6 if op == "addmm" else [], right)]
            call = {"op": "aten." + op, "args": args, "kwargs": {"beta": 0}}
            self.assertTrue(self.adapter.check_output("real", call, {"out": tensor([], output)}))
            call = {"op": "aten." + op, "args": [*args, 0]}
            with self.assertRaises(Invalid):
                self.adapter.reference("real", call)

    def test_zero_beta_does_not_disable_input_validation(self):
        base = [tensor("nan", []), tensor([[2]], [1, 1]), tensor([[3]], [1, 1])]
        for bad in (tensor("nan", [2]), tensor(["nan", "nan"], [2]), tensor("nan", [], "f64"),
                    tensor("not-a-number", [])):
            with self.subTest(bad=bad), self.assertRaises(Invalid):
                self.adapter.reference("real", {"op": "aten.addmm", "args": [bad, *base[1:]], "kwargs": {"beta": 0}})
        with self.assertRaises(Unsupported):
            self.adapter.reference("real", {"op": "aten.addmm",
                                              "args": [tensor("nan", [], "i32"), *base[1:]],
                                              "kwargs": {"beta": 0}})
        for parameter in ("alpha", "beta"):
            with self.assertRaises(Invalid):
                self.adapter.reference("real", {"op": "aten.addmm", "args": base,
                                                  "kwargs": {"beta": 0, parameter: [0]}})

    def test_nan_in_used_inputs_and_uncertain_beta_are_not_ignored(self):
        a = tensor([[2]], [1, 1])
        for args, kwargs in (([tensor("nan", []), a, a], {}),
                             ([tensor(0, []), tensor([[float("nan")]], [1, 1]), a], {"beta": 0})):
            with self.assertRaises(Invalid):
                self.adapter.reference("real", {"op": "aten.addmm", "args": args, "kwargs": kwargs})
        with self.assertRaises(NotEstablished):
            self.adapter.reference("real", {"op": "aten.addmm", "args": [tensor("nan", []), a, a],
                                              "kwargs": {"beta": R(hull(-1, 1))}})


if __name__ == "__main__":
    unittest.main()
