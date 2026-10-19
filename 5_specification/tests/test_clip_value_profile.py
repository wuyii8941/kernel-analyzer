"""Admission boundaries and complete outputs for the reviewed CLIP profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


class ClipValueProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("CLIP").adapter

    def test_complete_post_state_requires_dtype_shape_and_every_value(self):
        inp = {"op": "torch.nn.utils.clip_grad_value_", "args": [
            {"t": "f32", "v": [-2, -0.5, 0.5, 2]}, 0.75], "kwargs": {"foreach": False}}
        good = {"out": {"t": "f32", "v": [-0.75, -0.5, 0.5, 0.75], "shape": [4]}}
        self.assertTrue(self.adapter.check_output("finite_value_single_gradient", inp, good))
        self.assertFalse(self.adapter.check_output(
            "finite_value_single_gradient", inp, {"out": [-0.75, -0.5, 0.5, 0.75]}))
        self.assertFalse(self.adapter.check_output(
            "finite_value_single_gradient", inp,
            {"out": {"t": "f64", "v": [-0.75, -0.5, 0.5, 0.75], "shape": [4]}}))
        self.assertFalse(self.adapter.check_output(
            "finite_value_single_gradient", inp,
            {"out": {"t": "f32", "v": [-0.75, -0.5, 0.5], "shape": [3]}}))

    def test_foreach_choices_have_one_target(self):
        expected = [-1, 0, 1]
        for foreach in (False, True, None):
            inp = {"op": "torch.nn.utils.clip_grad_value_", "args": [
                {"t": "f32", "v": [-2, 0, 2]}, 1], "kwargs": {"foreach": foreach}}
            self.assertEqual(
                self.adapter.reference("finite_value_single_gradient", inp).fields["out"], expected)

    def test_physical_and_clip_boundaries_are_enforced(self):
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_value_single_gradient", {
                "op": "torch.nn.utils.clip_grad_value_", "args": [{"t": "f32", "v": [1]}, 0]})
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_value_single_gradient", {
                "op": "torch.nn.utils.clip_grad_value_", "args": [{"t": "f32", "v": [1]}, "1/3"]})
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_value_single_gradient", {
                "op": "torch.nn.utils.clip_grad_value_", "args": [{"t": "f32", "v": [-0.0, 1]}, 1]})
        with self.assertRaises(Invalid):
            self.adapter.reference("finite_value_single_gradient", {
                "op": "torch.nn.utils.clip_grad_value_", "args": [{"t": "f32", "v": [1]}, 1],
                "kwargs": {"foreach": 1}})

    def test_norm_clipping_is_refused(self):
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_value_single_gradient", {
                "op": "torch.nn.utils.clip_grad_norm_", "args": [{"t": "f32", "v": [1]}, 1]})

    def test_internal_mutation_selector_is_not_a_task_argument(self):
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_value_single_gradient", {
                "op": "torch.nn.utils.clip_grad_value_", "args": [{"t": "f32", "v": [-2, 2]}, 1],
                "kwargs": {"mode": "upper_only"}})


if __name__ == "__main__":
    unittest.main()
