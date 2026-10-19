"""Admission boundaries and complete outputs for the reviewed INDEXREDUCE profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


class IndexAddProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("INDEXREDUCE").adapter

    def test_complete_output_requires_dtype_shape_and_every_value(self):
        inp = {"op": "aten.index_add", "args": [
            {"t": "f32", "v": [1, 1, 1]}, 0, {"t": "i64", "v": [0, 2, 0]},
            {"t": "f32", "v": [1, 2, 3]}], "kwargs": {"alpha": 2}}
        good = {"out": {"t": "f32", "v": [9, 1, 5], "shape": [3]}}
        self.assertTrue(self.adapter.check_output("finite_index_add_values", inp, good))
        self.assertFalse(self.adapter.check_output("finite_index_add_values", inp, {"out": [9, 1, 5]}))
        self.assertFalse(self.adapter.check_output(
            "finite_index_add_values", inp, {"out": {"t": "f64", "v": [9, 1, 5], "shape": [3]}}))
        self.assertFalse(self.adapter.check_output(
            "finite_index_add_values", inp, {"out": {"t": "f32", "v": [9, 1], "shape": [2]}}))

    def test_duplicate_indices_all_contribute(self):
        inp = {"op": "torch.index_add", "args": [
            {"t": "f32", "v": [1, 1, 1]}, 0, {"t": "i64", "v": [0, 2, 0]},
            {"t": "f32", "v": [1, 2, 3]}], "kwargs": {"alpha": 2}}
        self.assertEqual(self.adapter.reference("finite_index_add_values", inp).fields["out"], [9, 1, 5])

    def test_shape_range_and_physical_boundaries_are_enforced(self):
        with self.assertRaises(Invalid):
            self.adapter.reference("finite_index_add_values", {"op": "aten.index_add", "args": [
                {"t": "f32", "v": [1, 2]}, 0, {"t": "i64", "v": [0, 1]},
                {"t": "f32", "v": [3]}]})
        with self.assertRaises(Invalid):
            self.adapter.reference("finite_index_add_values", {"op": "aten.index_add", "args": [
                {"t": "f32", "v": [1, 2]}, 0, {"t": "i64", "v": [2]},
                {"t": "f32", "v": [3]}]})
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_index_add_values", {"op": "aten.index_add", "args": [
                {"t": "f32", "v": [1, 2]}, 0, {"t": "i64", "v": [0]},
                {"t": "f32", "v": [3]}], "kwargs": {"alpha": "1/3"}})
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_index_add_values", {"op": "aten.index_add", "args": [
                {"t": "f32", "v": [-0.0, 2]}, 0, {"t": "i64", "v": [0]},
                {"t": "f32", "v": [3]}]})

    def test_other_reductions_and_inplace_names_are_refused(self):
        inp = {"op": "aten.index_reduce", "args": [
            {"t": "f32", "v": [1]}, 0, {"t": "i64", "v": [0]}, {"t": "f32", "v": [2]}, "prod"]}
        with self.assertRaises(Unsupported):
            self.adapter.reference("finite_index_add_values", inp)


if __name__ == "__main__":
    unittest.main()
