"""Complete-output comparison for REDUCE, including metadata and structured extrema."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Unsupported


def tensor(values, shape, dtype="f32"):
    return {"t": dtype, "v": values, "shape": shape}


class ReductionOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("REDUCE").adapter

    def test_every_value_coordinate_and_metadata_is_checked(self):
        call = {"op": "aten.sum", "args": [tensor([[1, 2], [3, 4]], [2, 2]), 1]}
        self.assertTrue(self.adapter.check_output("real", call, {"out": tensor([3, 7], [2])}))
        for output in (tensor([3, 8], [2]), tensor([[3, 7]], [1, 2]),
                       tensor([3, 7], [2], "f64"), [3, 7]):
            with self.subTest(output=output):
                self.assertFalse(self.adapter.check_output("real", call, {"out": output}))

    def test_both_value_and_index_outputs_are_required(self):
        call = {"op": "aten.max", "args": [tensor([[1, 3, 3]], [1, 3]), 1]}
        correct = [tensor([3], [1]), tensor([1], [1], "i64")]
        self.assertTrue(self.adapter.check_output("real", call, {"out": correct}))
        for bad in (correct[:1], [correct[1], correct[0]],
                    [tensor([3], [1]), tensor([2], [1], "i64")],
                    [tensor([3], [1]), tensor([1], [1], "f32")]):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))

    def test_empty_outputs_still_require_shape_and_dtype(self):
        call = {"op": "aten.cumsum", "args": [tensor([], [0, 3]), 0]}
        self.assertTrue(self.adapter.check_output("real", call, {"out": tensor([], [0, 3])}))
        self.assertFalse(self.adapter.check_output("real", call, {"out": tensor([], [3, 0])}))
        self.assertFalse(self.adapter.check_output("real", call, {"out": tensor([], [0, 3], "f64")}))

    def test_profile_selection_is_explicit(self):
        call = {"op": "aten.sum", "args": [tensor([1, 2], [2])]}
        with self.assertRaises(Unsupported):
            self.adapter.reference("unselected", call)
        with self.assertRaises(Unsupported):
            self.adapter.check_output("unselected", call, {"out": tensor(3, [])})

    def test_variance_output_is_complete_and_exactly_checked(self):
        call = {"op": "aten.var", "args": [tensor([[1, 3], [2, 6]], [2, 2]), 1],
                "kwargs": {"correction": 1}}
        self.assertTrue(self.adapter.check_output("real", call,
                                                  {"out": tensor([2, 8], [2])}))
        for bad in (tensor([2, 7], [2]), tensor([[2, 8]], [1, 2]),
                    tensor([2, 8], [2], "f64")):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))

    def test_exact_standard_deviation_output_is_completely_checked(self):
        call = {"op": "aten.std", "args": [tensor([[1, 3], [2, 6]], [2, 2]), 1],
                "kwargs": {"correction": 0}}
        self.assertTrue(self.adapter.check_output("real", call,
                                                  {"out": tensor([1, 2], [2])}))
        for bad in (tensor([1, 3], [2]), tensor([[1, 2]], [1, 2]),
                    tensor([1, 2], [2], "f64")):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))

    def test_var_mean_requires_both_ordered_complete_outputs(self):
        call = {"op": "aten.var_mean",
                "args": [tensor([[1, 3], [2, 6]], [2, 2]), 1],
                "kwargs": {"correction": 1}}
        correct = [tensor([2, 8], [2]), tensor([2, 4], [2])]
        self.assertTrue(self.adapter.check_output("real", call, {"out": correct}))
        for bad in (
            correct[:1],
            [correct[1], correct[0]],
            [tensor([2, 7], [2]), correct[1]],
            [correct[0], tensor([[2, 4]], [1, 2])],
            [correct[0], tensor([2, 4], [2], "f64")],
        ):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))

    def test_vector_norm_output_is_complete_and_exactly_checked(self):
        call = {"op": "aten.linalg_vector_norm",
                "args": [tensor([[3, 4], [5, 12]], [2, 2]), 2, 1]}
        correct = tensor([5, 13], [2])
        self.assertTrue(self.adapter.check_output("real", call, {"out": correct}))
        for bad in (tensor([5, 12], [2]), tensor([[5, 13]], [1, 2]),
                    tensor([5, 13], [2], "f64")):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("real", call, {"out": bad}))


if __name__ == "__main__":
    unittest.main()
