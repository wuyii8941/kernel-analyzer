import importlib.util
import unittest
from pathlib import Path


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "MATHFN.py"
    spec = importlib.util.spec_from_file_location("reviewed_mathfn_output_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MathFunctionCompleteOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()
        cls.call = {"op": "aten.exp", "args": [{"t": "f32", "v": [0, 0]}]}

    def test_complete_dtype_shape_and_values_are_required(self):
        correct = {"out": {"t": "f32", "shape": [2], "v": [1, 1]}}
        self.assertTrue(self.adapter.check_output("real", self.call, correct))
        bad = (
            {"out": {"t": "f64", "shape": [2], "v": [1, 1]}},
            {"out": {"t": "f32", "shape": [1, 2], "v": [[1, 1]]}},
            {"out": {"t": "f32", "shape": [2], "v": [1, 0]}},
            {"out": {"t": "f32", "v": [1, 1]}},
            {},
        )
        for observed in bad:
            with self.subTest(observed=observed):
                self.assertFalse(self.adapter.check_output("real", self.call, observed))

    def test_empty_dimensions_are_not_reconstructed_from_values(self):
        call = {"op": "aten.cos", "args": [{"t": "f64", "v": [], "shape": [2, 0]}]}
        self.assertTrue(self.adapter.check_output(
            "real", call, {"out": {"t": "f64", "shape": [2, 0], "v": [[], []]}}))
        self.assertFalse(self.adapter.check_output(
            "real", call, {"out": {"t": "f64", "shape": [0, 2], "v": []}}))


if __name__ == "__main__":
    unittest.main()
