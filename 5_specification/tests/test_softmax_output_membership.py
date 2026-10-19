import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import NotEstablished, Unsupported, ends


def tensor(values, shape, dtype="f32"):
    return {"t": dtype, "v": values, "shape": shape}


class SoftmaxOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("SOFTMAX").adapter

    def test_exact_outputs_and_masked_log_probability(self):
        for op, values, expected in (("aten._softmax", [0, 0], [Q(1, 2)]*2),
                                    ("aten._safe_softmax", ["-inf", "-inf"], [0, 0]),
                                    ("aten._log_softmax", [0, "-inf"], [0, "-inf"])):
            call = {"op": op, "args": [tensor(values, [2]), 0] + ([] if op == "aten._safe_softmax" else [False])}
            self.assertTrue(self.adapter.check_output("real", call, {"out": tensor(expected, [2])}))
            self.assertFalse(self.adapter.check_output("real", call, {"out": tensor([0, 1], [2])}))

    def test_both_online_outputs_and_all_coordinates_are_required(self):
        call = {"op": "prims.prepare_softmax_online", "args": [tensor([[2, 2], [3, 3]], [2, 2]), 1]}
        output = [tensor([[2], [3]], [2, 1]), tensor([[2], [2]], [2, 1])]
        self.assertTrue(self.adapter.check_output("real", call, {"out": output}))
        for bad in (output[:1], [output[0], tensor([[2], [3]], [2, 1])],
                    [output[0], tensor([2], [1, 1])]):
            self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))

    def test_empty_shape_and_dtype_cannot_disappear_in_nested_lists(self):
        call = {"op": "aten._softmax", "args": [tensor([], [0, 3], "f16"), 1, True]}
        self.assertTrue(self.adapter.check_output("real", call, {"out": tensor([], [0, 3])}))
        for bad in (tensor([], [0, 4]), tensor([], [0, 3], "f16"), []):
            self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))
        nonempty = {"op": "aten._softmax", "args": [tensor([0, 0], [2]), 0, False]}
        for shape in ([-1], [2.0], [True, 2]):
            self.assertFalse(self.adapter.check_output("real", nonempty, {"out": tensor([Q(1, 2)]*2, shape)}))

    def test_interval_midpoint_is_not_equality_and_later_error_still_counts(self):
        call = {"op": "aten._softmax", "args": [tensor([[0, 1], [0, "-inf"]], [2, 2]), 1, False]}
        ref = self.adapter.reference("real", call).fields["out"]
        mid = [sum(ends(value))/2 for value in ref[0]]
        with self.assertRaises(NotEstablished):
            self.adapter.check_output("real", call, {"out": tensor([mid, [1, 0]], [2, 2])})
        self.assertFalse(self.adapter.check_output("real", call, {"out": tensor([mid, [0, 0]], [2, 2])}))

    def test_unknown_profile_is_not_silently_used_by_development_calls(self):
        call = {"op": "aten._softmax", "args": [tensor([0, 0], [2]), 0, False]}
        for fn, args in ((self.adapter.reference, (call,)),
                         (self.adapter.check_output, (call, {"out": tensor([Q(1, 2)]*2, [2])}))):
            with self.assertRaises(Unsupported):
                fn("unselected", *args)

    def test_admitted_reference_comparison_is_not_a_kernel_verdict(self):
        call = {"op": "aten._softmax", "args": [tensor([0, 0], [2]), 0, False]}
        verdict, _, labels = registry.check_output("SOFTMAX", "real", call,
                                                  {"out": tensor([Q(1, 2)]*2, [2])})
        self.assertEqual(verdict, "consistent")
        self.assertEqual(labels["semantic_verdict"], "not_established")


if __name__ == "__main__":
    unittest.main()
