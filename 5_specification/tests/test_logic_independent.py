import importlib.util
import itertools
import math
import unittest
from pathlib import Path

import numpy as np


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "LOGIC.py"
    spec = importlib.util.spec_from_file_location("reviewed_logic_independent_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogicIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def call(self, op, args, profile="real"):
        return self.adapter.reference(profile, {"op": op, "args": args}).fields["out"]

    def test_comparisons_match_independent_numpy_broadcasting(self):
        left = np.array([[-2.0], [0.5], [2.0]])
        right = np.array([-1.0, 0.5, 3.0])
        record_left = {"t": "f32", "v": [[-2], ["1/2"], [2]]}
        record_right = {"t": "f32", "v": [-1, "1/2", 3]}
        operations = {
            "aten.eq": np.equal, "aten.ne": np.not_equal, "aten.lt": np.less,
            "aten.le": np.less_equal, "aten.gt": np.greater, "aten.ge": np.greater_equal,
        }
        for op, fn in operations.items():
            with self.subTest(op=op):
                self.assertEqual(self.call(op, [record_left, record_right]), fn(left, right).tolist())

    def test_logical_and_boolean_bitwise_values_use_independent_truth_tables(self):
        values = [False, True]
        for a, b in itertools.product(values, repeat=2):
            x, y = {"t": "b8", "v": [a]}, {"t": "b8", "v": [b]}
            self.assertEqual(self.call("aten.logical_and", [x, y]), [a and b])
            self.assertEqual(self.call("aten.logical_or", [x, y]), [a or b])
            self.assertEqual(self.call("aten.bitwise_and", [x, y]), [a and b])
            self.assertEqual(self.call("aten.bitwise_or", [x, y]), [a or b])
        self.assertEqual(self.call("aten.logical_not", [{"t": "f64", "v": [0, -2, "1/2"]}]),
                         [True, False, False])
        self.assertEqual(self.call("aten.bitwise_not", [{"t": "b8", "v": [False, True]}]),
                         [True, False])

    def test_truth_reductions_match_independent_numpy(self):
        values = np.array([[0.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
        record = {"t": "f32", "v": [[0, -1, 2], [0, 0, 0]]}
        self.assertEqual(self.call("aten.any", [record]), bool(np.any(values)))
        self.assertEqual(self.call("aten.all", [record]), bool(np.all(values)))
        self.assertEqual(self.call("aten.any", [record, 1]), np.any(values, axis=1).tolist())
        self.assertEqual(self.call("aten.all", [record, 0, True]), np.all(values, axis=0, keepdims=True).tolist())
        self.assertFalse(self.call("aten.any", [{"t": "b8", "v": []}]))
        self.assertTrue(self.call("aten.all", [{"t": "b8", "v": []}]))

    def test_isnan_classifies_nan_separately_from_infinities(self):
        values = [float("nan"), float("inf"), float("-inf"), -0.0, 1.0]
        expected = [math.isnan(value) for value in values]
        self.assertEqual(self.call("aten.isnan", [{"t": "f64", "v": values}], "ieee_values"), expected)

    def test_pytorch_crosscheck_for_all_reviewed_dtypes(self):
        import torch

        torch_dtypes = {
            "f16": torch.float16, "bf16": torch.bfloat16,
            "f32": torch.float32, "f64": torch.float64,
        }
        values = [-2, "-1/2", 0, "1/2", 2]
        native_values = [-2.0, -0.5, 0.0, 0.5, 2.0]
        for name, dtype in torch_dtypes.items():
            tensor = torch.tensor(native_values, dtype=dtype)
            record = {"t": name, "v": values}
            with self.subTest(dtype=name, op="ge"):
                self.assertEqual(self.call("aten.ge", [record, "1/2"]), torch.ge(tensor, 0.5).tolist())
            with self.subTest(dtype=name, op="logical_not"):
                self.assertEqual(self.call("aten.logical_not", [record]), torch.logical_not(tensor).tolist())
            with self.subTest(dtype=name, op="any"):
                self.assertEqual(self.call("aten.any", [record]), bool(torch.any(tensor)))
        mask = torch.tensor([[True, False], [False, True]])
        record = {"t": "b8", "v": [[True, False], [False, True]]}
        self.assertEqual(self.call("aten.bitwise_not", [record]), torch.bitwise_not(mask).tolist())


if __name__ == "__main__":
    unittest.main()
