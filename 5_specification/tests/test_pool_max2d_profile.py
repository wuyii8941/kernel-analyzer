"""Admission boundaries and complete outputs for the reviewed POOL profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


class PoolMax2dProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("POOL").adapter

    @staticmethod
    def tensor(dtype, values, shape):
        return {"t": dtype, "v": values, "shape": shape}

    def test_complete_values_dtype_and_shape_are_required(self):
        inp = {"op": "aten.max_pool2d", "args": [
            {"t": "f32", "v": [[[[1, 2], [3, 4]]]]}, 2]}
        good = {"out": self.tensor("f32", [[[[4]]]], [1, 1, 1, 1])}
        self.assertTrue(self.adapter.check_output("finite_max2d_values", inp, good))
        for bad in (
            {"out": [[[[4]]]]},
            {"out": self.tensor("f16", [[[[4]]]], [1, 1, 1, 1])},
            {"out": self.tensor("f32", [[[[3]]]], [1, 1, 1, 1])},
            {"out": self.tensor("f32", [[[[4]]]], [1, 1, 1, 1, 1])},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("finite_max2d_values", inp, bad))

    def test_ties_are_valid_for_values_only_forward(self):
        inp = {"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[5, 5], [1, 0]]]}, 2]}
        self.assertEqual(self.adapter.reference("finite_max2d_values", inp).fields["out"], [[[5]]])

    def test_unselected_semantics_and_malformed_inputs_are_refused(self):
        cases = [
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, 2], [3, 4]]]}, 2, 1, 1]}, Unsupported),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, 2], [3, 4]]]}, 2], "kwargs": {"ceil_mode": True}}, Unsupported),
            ({"op": "torch.nn.functional.max_pool2d", "args": [{"t": "f32", "v": [[[1, 2], [3, 4]]]}, 2], "kwargs": {"return_indices": True}}, Unsupported),
            ({"op": "aten.max_pool2d", "args": [{"t": "i64", "v": [[[1, 2], [3, 4]]]}, 2]}, Unsupported),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, "nan"], [3, 4]]]}, 2]}, Unsupported),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, "-0.0"], [3, 4]]]}, 2]}, Unsupported),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, True], [3, 4]]]}, 2]}, Invalid),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, "1/3"], [3, 4]]]}, 2]}, Invalid),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[1, 2], [3, 4]]}, 2]}, Unsupported),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, 2], [3, 4]]]}, [2]]}, Invalid),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, 2], [3, 4]]]}, 0]}, Invalid),
            ({"op": "aten.max_pool2d", "args": [{"t": "f32", "v": [[[1, 2], [3, 4]]]}, 3]}, Invalid),
        ]
        for inp, error in cases:
            with self.subTest(inp=inp), self.assertRaises(error):
                self.adapter.reference("finite_max2d_values", inp)


if __name__ == "__main__":
    unittest.main()
