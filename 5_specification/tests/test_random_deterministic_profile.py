"""Admission boundaries and complete outputs for reviewed deterministic dropout profiles."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


class RandomDeterministicProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("RANDOM").adapter

    def test_complete_output_requires_dtype_shape_and_every_value(self):
        inp = {"op": "aten.native_dropout_backward", "args": [
            {"t": "f32", "v": [1, 2, 3]}, {"t": "b8", "v": [True, False, True]}, "5/4"]}
        good = {"out": {"t": "f32", "v": [1.25, 0, 3.75], "shape": [3]}}
        self.assertTrue(self.adapter.check_output("finite_backward", inp, good))
        self.assertFalse(self.adapter.check_output("finite_backward", inp, {"out": [1.25, 0, 3.75]}))
        self.assertFalse(self.adapter.check_output(
            "finite_backward", inp, {"out": {"t": "f64", "v": [1.25, 0, 3.75], "shape": [3]}}))
        self.assertFalse(self.adapter.check_output(
            "finite_backward", inp, {"out": {"t": "f32", "v": [1.25, 0], "shape": [2]}}))

    def test_stochastic_forward_is_refused_but_boundaries_are_values(self):
        x = {"t": "f32", "v": [1, -2, 3]}
        with self.assertRaises(Unsupported):
            self.adapter.reference("deterministic_forward", {"op": "aten.dropout", "args": [x, "1/2", True]})
        self.assertEqual(
            self.adapter.reference("deterministic_forward", {"op": "aten.dropout", "args": [x, 0, True]}).fields["out"],
            [1, -2, 3])
        self.assertEqual(
            self.adapter.reference("deterministic_forward", {"op": "aten.dropout", "args": [x, 1, True]}).fields["out"],
            [0, 0, 0])

    def test_profile_and_operation_must_match(self):
        x = {"t": "f32", "v": [1, 2]}
        mask = {"t": "b8", "v": [True, False]}
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_backward", {"op": "aten.dropout", "args": [x, 0, True]})
        with self.assertRaises(Unsupported):
            self.adapter.reference("deterministic_forward", {
                "op": "aten.native_dropout_backward", "args": [x, mask, 1]})

    def test_malformed_physical_inputs_are_rejected_or_refused(self):
        with self.assertRaises(Invalid):
            self.adapter.reference("finite_backward", {"op": "aten.native_dropout_backward", "args": [
                {"t": "f32", "v": [1, 2]}, {"t": "b8", "v": [True]}, 1]})
        with self.assertRaises(Invalid):
            self.adapter.reference("finite_backward", {"op": "aten.native_dropout_backward", "args": [
                {"t": "f32", "v": [1, 2]}, {"t": "b8", "v": [1, 0]}, 1]})
        with self.assertRaises(Invalid):
            self.adapter.reference("deterministic_forward", {"op": "aten.dropout", "args": [
                {"t": "f32", "v": ["1/3"]}, 0, True]})
        with self.assertRaises(Unsupported):
            self.adapter.reference("deterministic_forward", {"op": "aten.dropout", "args": [
                {"t": "f32", "v": [-0.0]}, 0, True]})


if __name__ == "__main__":
    unittest.main()
