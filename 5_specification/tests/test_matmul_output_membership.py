import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Unsupported


def tensor(values, shape, dtype="f32"):
    return {"t": dtype, "v": values, "shape": shape}


class MatmulOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("MATMUL").adapter

    def test_complete_product_metadata_and_coordinates(self):
        call = {"op": "aten.mm", "args": [tensor([[1, 2]], [1, 2]), tensor([[3, 4], [5, 6]], [2, 2])]}
        self.assertTrue(self.adapter.check_output("real", call, {"out": tensor([[13, 16]], [1, 2])}))
        for output in (tensor([[13, 17]], [1, 2]), tensor([13, 16], [2]),
                       tensor([[13, 16]], [1, 2], "f64"), [[13, 16]]):
            self.assertFalse(self.adapter.check_output("real", call, {"out": output}))

    def test_empty_results_still_require_shape_and_dtype(self):
        call = {"op": "aten.bmm", "args": [tensor([], [0, 2, 3]), tensor([], [0, 3, 4])]}
        self.assertTrue(self.adapter.check_output("real", call, {"out": tensor([], [0, 2, 4])}))
        for output in (tensor([], [0, 4, 2]), tensor([], [0, 2, 4], "f64"), []):
            self.assertFalse(self.adapter.check_output("real", call, {"out": output}))

    def test_all_gradients_and_absent_bias_are_observed(self):
        call = {"op": "vjp.aten.linear", "args": [tensor([2], [1]), tensor([3, 4], [2]), tensor([[5, 6]], [1, 2])]}
        output = [tensor([10, 12], [2]), tensor([[6, 8]], [1, 2]), None]
        self.assertTrue(self.adapter.check_output("real", call, {"out": output}))
        for bad in (output[:2], [*output[:2], tensor(0, [])],
                    [None, *output[1:]], [output[0], tensor([[6, 9]], [1, 2]), None]):
            self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))
        call["args"].append(tensor(7, []))
        self.assertTrue(self.adapter.check_output("real", call, {"out": [*output[:2], tensor(2, [])]}))
        for bias in (None, tensor([2], [1]), tensor(0, [])):
            self.assertFalse(self.adapter.check_output("real", call, {"out": [*output[:2], bias]}))

    def test_profile_is_selected_and_admitted_comparison_is_not_a_kernel_verdict(self):
        call = {"op": "aten.mm", "args": [tensor([[2]], [1, 1]), tensor([[3]], [1, 1])]}
        observed = {"out": tensor([[6]], [1, 1])}
        for fn, args in ((self.adapter.reference, (call,)), (self.adapter.check_output, (call, observed))):
            with self.assertRaises(Unsupported):
                fn("unselected", *args)
        verdict, _, labels = registry.check_output("MATMUL", "real", call, observed)
        self.assertEqual(verdict, "consistent")
        self.assertEqual(labels["semantic_verdict"], "not_established")


if __name__ == "__main__":
    unittest.main()
