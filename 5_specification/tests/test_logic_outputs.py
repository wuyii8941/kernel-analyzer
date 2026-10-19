import importlib.util
import unittest
from pathlib import Path


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "LOGIC.py"
    spec = importlib.util.spec_from_file_location("reviewed_logic_output_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogicCompleteOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()
        cls.call = {
            "op": "aten.gt",
            "args": [{"t": "f32", "v": [[1], [-1]]}, {"t": "f32", "v": [0, 1]}],
        }

    def test_complete_boolean_dtype_shape_and_values_are_required(self):
        correct = {"out": {"t": "b8", "shape": [2, 2],
                           "v": [[True, False], [False, False]]}}
        self.assertTrue(self.adapter.check_output("real", self.call, correct))
        bad = (
            {"out": {"t": "i64", "shape": [2, 2], "v": [[1, 0], [0, 0]]}},
            {"out": {"t": "b8", "shape": [4], "v": [True, False, False, False]}},
            {"out": {"t": "b8", "shape": [2, 2], "v": [[True, True], [False, False]]}},
            {"out": {"t": "b8", "shape": [2, 2], "v": [[1, 0], [0, 0]]}},
            {"out": {"t": "b8", "v": [[True, False], [False, False]]}},
            {"out": correct["out"], "extra": 1},
            {},
        )
        for observed in bad:
            with self.subTest(observed=observed):
                self.assertFalse(self.adapter.check_output("real", self.call, observed))

    def test_scalar_and_empty_boolean_outputs_keep_explicit_shapes(self):
        all_empty = {"op": "aten.all", "args": [{"t": "b8", "v": []}]}
        self.assertTrue(self.adapter.check_output(
            "real", all_empty, {"out": {"t": "b8", "shape": [], "v": True}}))
        empty = {"op": "aten.eq", "args": [
            {"t": "f32", "v": [], "shape": [0, 2]},
            {"t": "f32", "v": [], "shape": [0, 1]},
        ]}
        self.assertTrue(self.adapter.check_output(
            "real", empty, {"out": {"t": "b8", "shape": [0, 2], "v": []}}))

    def test_isnan_output_is_checked_completely(self):
        call = {"op": "aten.isnan", "args": [{"t": "f16", "v": [float("nan"), float("inf"), 0]}]}
        correct = {"out": {"t": "b8", "shape": [3], "v": [True, False, False]}}
        self.assertTrue(self.adapter.check_output("ieee_values", call, correct))
        wrong = {"out": {"t": "b8", "shape": [3], "v": [True, True, False]}}
        self.assertFalse(self.adapter.check_output("ieee_values", call, wrong))


if __name__ == "__main__":
    unittest.main()
