import importlib.util
import os
import unittest
from pathlib import Path

from kernel_spec.num import Invalid, Unsupported


def load_adapter():
    # LAYOUT_ADAPTER may name a staged candidate adapter; by default the current top-level adapter is tested.
    path = Path(os.environ.get("LAYOUT_ADAPTER") or Path(__file__).resolve().parents[1] / "adapters" / "LAYOUT.py")
    spec = importlib.util.spec_from_file_location("reviewed_layout_domain_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayoutProfileDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def call(self, op, args, kwargs=None, profile="real"):
        inp = {"op": op, "args": args}
        if kwargs is not None:
            inp["kwargs"] = kwargs
        return self.adapter.reference(profile, inp)

    def test_four_reviewed_dtypes_and_empty_shapes_are_supported(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            with self.subTest(dtype=dtype):
                result = self.call("aten.reshape", [{"t": dtype, "v": [1, 2, 3, 4]}, [2, 2]])
                self.assertEqual(result.fields["out"], [[1, 2], [3, 4]])
        empty = self.call("aten.reshape", [{"t": "f32", "v": [], "shape": [0, 3]}, [3, 0]])
        self.assertEqual(empty.fields["out"], [[], [], []])

    def test_unreviewed_types_nonfinite_unrepresentable_and_negative_zero_are_refused(self):
        # 2026-10-20 extension (LY.EXACT_DATA): integer data is in the real profile; b8 belongs to the boolean profile.
        self.assertEqual(self.call("aten.clone", [{"t": "i64", "v": [1]}]).fields["out"], [1])
        for record in (
            {"t": "b8", "v": [True]},
            {"t": "f8e4m3", "v": [1]},
            {"t": "complex64", "v": [1]},
            {"t": "f32", "v": ["inf"]},
        ):
            with self.subTest(record=record), self.assertRaises(Unsupported):
                self.call("aten.clone", [record])
        for record in ({"t": "f16", "v": ["1/3"]}, {"t": "f32", "v": [-0.0]}):
            with self.subTest(record=record), self.assertRaises(Invalid):
                self.call("aten.clone", [record])

    def test_multi_tensor_dtype_and_scalar_padding_are_strict(self):
        with self.assertRaises(Unsupported):
            self.call("aten.cat", [[{"t": "f16", "v": [1]}, {"t": "f32", "v": [2]}]])
        with self.assertRaises(Unsupported):
            self.call("aten.select_scatter", [
                {"t": "f16", "v": [[0], [0]]}, {"t": "f32", "v": [1]}, 0, 0])
        with self.assertRaises(Invalid):
            self.call("aten.constant_pad_nd", [{"t": "f16", "v": [1]}, [1, 0], "1/3"])

    def test_selected_functional_schemas_reject_other_overloads(self):
        x = {"t": "f32", "v": [1, 2]}
        with self.assertRaises(Unsupported):
            self.call("aten.copy", [x, x])
        with self.assertRaises(Unsupported):
            self.call("aten.clone", [x], {"memory_format": "contiguous_format"})
        with self.assertRaises(Unsupported):
            self.call("aten.expand", [x, [2, 2]], {"implicit": True})
        with self.assertRaises(Unsupported):
            self.call("aten.reshape", [x, [2]], {"out": x})
        with self.assertRaises(Unsupported):
            self.call("aten.narrow", [x, 0, {"t": "i64", "v": 0}, 1])

    def test_shape_dimension_range_and_source_errors_are_invalid(self):
        x = {"t": "f32", "v": [[1, 2], [3, 4]]}
        bad_calls = (
            ("aten.view", [x, [3]]),
            ("aten.permute", [x, [0, 0]]),
            ("aten.transpose", [x, 0, 2]),
            ("aten.expand", [x, [3, 2]]),
            ("aten.select", [x, 0, 2]),
            ("aten.slice", [x, 0, None, None, 0]),
            ("aten.narrow", [x, 1, 1, 2]),
            ("aten.split_with_sizes", [x, [1, 2], 0]),
            ("aten.select_scatter", [x, {"t": "f32", "v": [1]}, 0, 0]),
            ("aten.triu", [{"t": "f32", "v": [1, 2]}]),
        )
        for op, args in bad_calls:
            with self.subTest(op=op), self.assertRaises(Invalid):
                self.call(op, args)


if __name__ == "__main__":
    unittest.main()
