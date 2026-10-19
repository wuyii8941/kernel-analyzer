"""The CAST comparison observes dtype, shape and every coordinate."""
import importlib.util
import unittest
from pathlib import Path


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "CAST.py"
    spec = importlib.util.spec_from_file_location("reviewed_cast_output_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CastCompleteOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def test_complete_dtype_shape_and_values_are_required(self):
        call = {"op": "prims.convert_element_type",
                "args": [{"t": "f32", "v": ["257/256", "259/256"]}, "bf16"]}
        correct = {"out": {"t": "bf16", "shape": [2], "v": [1, "65/64"]}}
        self.assertTrue(self.adapter.check_output("finite_rne", call, correct))
        bad = (
            {"out": {"t": "f16", "shape": [2], "v": [1, "65/64"]}},
            {"out": {"t": "bf16", "shape": [1, 2], "v": [[1, "65/64"]]}},
            {"out": {"t": "bf16", "shape": [2], "v": [1, 1]}},
            {"out": {"t": "bf16", "v": [1, "65/64"]}},
            {},
        )
        for observed in bad:
            with self.subTest(observed=observed):
                self.assertFalse(self.adapter.check_output("finite_rne", call, observed))

    def test_empty_dimensions_are_explicit(self):
        call = {"op": "prims.convert_element_type",
                "args": [{"t": "f64", "v": [], "shape": [2, 0, 3]}, "f32"]}
        self.assertTrue(self.adapter.check_output(
            "finite_rne", call, {"out": {"t": "f32", "shape": [2, 0, 3], "v": [[], []]}}))
        self.assertFalse(self.adapter.check_output(
            "finite_rne", call, {"out": {"t": "f32", "shape": [0, 2, 3], "v": []}}))


if __name__ == "__main__":
    unittest.main()
