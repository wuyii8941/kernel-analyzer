import importlib.util
import unittest
from pathlib import Path


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "LAYOUT.py"
    spec = importlib.util.spec_from_file_location("reviewed_layout_output_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayoutCompleteOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def test_single_result_requires_complete_dtype_shape_and_values(self):
        call = {"op": "aten.transpose", "args": [{"t": "f32", "v": [[1, 2], [3, 4]]}, 0, 1]}
        correct = {"out": {"t": "f32", "shape": [2, 2], "v": [[1, 3], [2, 4]]}}
        self.assertTrue(self.adapter.check_output("real", call, correct))
        bad = (
            {"out": {"t": "f64", "shape": [2, 2], "v": correct["out"]["v"]}},
            {"out": {"t": "f32", "shape": [4], "v": [1, 3, 2, 4]}},
            {"out": {"t": "f32", "shape": [2, 2], "v": [[1, 2], [3, 4]]}},
            {"out": {"t": "f32", "v": correct["out"]["v"]}},
            {"out": correct["out"], "extra": 1},
        )
        for observed in bad:
            with self.subTest(observed=observed):
                self.assertFalse(self.adapter.check_output("real", call, observed))

    def test_split_requires_every_ordered_result_including_empty_pieces(self):
        call = {"op": "aten.split_with_sizes",
                "args": [{"t": "f32", "v": [1, 2, 3]}, [1, 0, 2]]}
        pieces = [
            {"t": "f32", "shape": [1], "v": [1]},
            {"t": "f32", "shape": [0], "v": []},
            {"t": "f32", "shape": [2], "v": [2, 3]},
        ]
        self.assertTrue(self.adapter.check_output("real", call, {"out": pieces}))
        self.assertFalse(self.adapter.check_output("real", call, {"out": pieces[:-1]}))
        self.assertFalse(self.adapter.check_output("real", call, {"out": [pieces[0], pieces[2], pieces[1]]}))
        malformed = list(pieces)
        malformed[1] = {"t": "f32", "shape": [0], "v": [], "extra": True}
        self.assertFalse(self.adapter.check_output("real", call, {"out": malformed}))


if __name__ == "__main__":
    unittest.main()
