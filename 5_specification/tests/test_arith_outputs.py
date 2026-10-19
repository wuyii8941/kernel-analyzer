import importlib.util
import unittest
from pathlib import Path


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "ARITH.py"
    spec = importlib.util.spec_from_file_location("reviewed_arith_output_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArithmeticCompleteOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()
        cls.call = {
            "op": "aten.add",
            "args": [{"t": "f32", "v": [[1], [2]]}, {"t": "f32", "v": ["1/2", 3]}],
        }

    def test_complete_dtype_shape_and_every_value_are_required(self):
        correct = {"out": {"t": "f32", "shape": [2, 2],
                           "v": [["3/2", 4], ["5/2", 5]]}}
        self.assertTrue(self.adapter.check_output("real", self.call, correct))
        bad = (
            {"out": {"t": "f64", "shape": [2, 2], "v": correct["out"]["v"]}},
            {"out": {"t": "f32", "shape": [4], "v": ["3/2", 4, "5/2", 5]}},
            {"out": {"t": "f32", "shape": [2, 2], "v": [["3/2", 4], ["5/2", 6]]}},
            {"out": {"t": "f32", "v": correct["out"]["v"]}},
            {"out": correct["out"], "extra": 1},
            {},
        )
        for observed in bad:
            with self.subTest(observed=observed):
                self.assertFalse(self.adapter.check_output("real", self.call, observed))

    def test_empty_broadcast_result_keeps_explicit_shape_and_dtype(self):
        call = {"op": "aten.maximum", "args": [
            {"t": "bf16", "v": [[], []]},
            {"t": "bf16", "v": [], "shape": [1, 0]},
        ]}
        self.assertTrue(self.adapter.check_output(
            "real", call, {"out": {"t": "bf16", "shape": [2, 0], "v": [[], []]}}))


if __name__ == "__main__":
    unittest.main()
