"""Admission boundaries and complete outputs for the reviewed SORT profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


class SortFiniteUniqueProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("SORT").adapter

    @staticmethod
    def tensor(dtype, values, shape):
        return {"t": dtype, "v": values, "shape": shape}

    def test_complete_values_and_indices_are_required(self):
        inp = {"op": "aten.sort", "args": [{"t": "f32", "v": [[3, 1, 2]]}, 1]}
        good = {"out": [self.tensor("f32", [[1, 2, 3]], [1, 3]),
                         self.tensor("i64", [[1, 2, 0]], [1, 3])]}
        self.assertTrue(self.adapter.check_output("finite_unique", inp, good))
        for bad in (
            {"out": good["out"][0]},
            {"out": [self.tensor("f32", [[1, 2, 3]], [1, 3]),
                     self.tensor("i64", [[1, 0, 2]], [1, 3])]},
            {"out": [self.tensor("f16", [[1, 2, 3]], [1, 3]), good["out"][1]]},
            {"out": [good["out"][0], self.tensor("i64", [[1, 2, 0]], [3, 1])]},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(self.adapter.check_output("finite_unique", inp, bad))

    def test_outside_selected_semantics_are_refused(self):
        cases = [
            ({"op": "aten.sort", "args": [{"t": "f32", "v": [1, 1]}]}, Unsupported),
            ({"op": "aten.sort", "args": [{"t": "f32", "v": [0, "nan"]}]}, Unsupported),
            ({"op": "aten.sort", "args": [{"t": "f32", "v": [0, "inf"]}]}, Unsupported),
            ({"op": "aten.sort", "args": [{"t": "i64", "v": [2, 1]}]}, Unsupported),
            ({"op": "aten.sort", "args": [{"t": "f32", "v": 1}]}, Unsupported),
            ({"op": "aten.sort", "args": [{"t": "f32", "v": ["1/3", 1]}]}, Invalid),
            ({"op": "aten.sort", "args": [{"t": "f32", "v": [True, 1]}]}, Invalid),
            ({"op": "aten.sort", "args": [{"t": "f32", "v": ["-0.0", 1]}]}, Unsupported),
            ({"op": "aten.topk", "args": [{"t": "f32", "v": [2, 1]}, 1],
              "kwargs": {"sorted": False}}, Unsupported),
            ({"op": "aten.sort", "args": [{"t": "f32", "v": [2, 1]}],
              "kwargs": {"stable": 1}}, Invalid),
            ({"op": "aten.topk", "args": [{"t": "f32", "v": [2, 1]}, True]}, Invalid),
        ]
        for inp, error in cases:
            with self.subTest(inp=inp), self.assertRaises(error):
                self.adapter.reference("finite_unique", inp)

    def test_zero_length_topk_keeps_explicit_output_shapes(self):
        inp = {"op": "aten.topk", "args": [{"t": "f32", "v": [[3, 1, 2]]}, 0, 1]}
        observed = {"out": [self.tensor("f32", [[]], [1, 0]), self.tensor("i64", [[]], [1, 0])]}
        self.assertTrue(self.adapter.check_output("finite_unique", inp, observed))


if __name__ == "__main__":
    unittest.main()
