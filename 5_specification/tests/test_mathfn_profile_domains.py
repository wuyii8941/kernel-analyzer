import importlib.util
import unittest
from pathlib import Path

from kernel_spec.num import Invalid, Undefined, Unsupported


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "MATHFN.py"
    spec = importlib.util.spec_from_file_location("reviewed_mathfn_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MathFunctionProfileDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def call(self, op, tensor, kwargs=None):
        inp = {"op": op, "args": [tensor]}
        if kwargs is not None:
            inp["kwargs"] = kwargs
        return self.adapter.reference("real", inp)

    def test_all_reviewed_floating_dtypes_and_empty_shapes_are_supported(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            with self.subTest(dtype=dtype):
                result = self.call("aten.exp", {"t": dtype, "v": [0, 1]})
                self.assertEqual(result.outcome, "value")
        empty = self.call("aten.cos", {"t": "f64", "v": [], "shape": [3, 0]})
        self.assertEqual(empty.fields["out"], [[], [], []])

    def test_unreviewed_types_values_and_magnitudes_are_refused(self):
        for tensor in ({"t": "i64", "v": [1]}, {"t": "b8", "v": [True]},
                       {"t": "f8e4m3", "v": [1]}, {"t": "f32", "v": [float("inf")]},
                       {"t": "f32", "v": [-101]}):
            with self.subTest(tensor=tensor), self.assertRaises(Unsupported):
                self.call("aten.exp", tensor)
        with self.assertRaises(Unsupported):
            self.call("aten.exp", {"t": "complex64", "v": [1]})
        with self.assertRaises(Invalid):
            self.call("aten.exp", {"t": "f32", "v": [float("nan")]})

    def test_function_specific_real_domains_are_enforced(self):
        cases = (("aten.log", 0), ("aten.log", -1), ("aten.log1p", -1),
                 ("aten.log1p", -2), ("aten.sqrt", -1), ("aten.rsqrt", 0),
                 ("aten.rsqrt", -1))
        for op, value in cases:
            with self.subTest(op=op, value=value), self.assertRaises(Undefined):
                self.call(op, {"t": "f32", "v": [value]})
        self.assertEqual(self.call("aten.sqrt", {"t": "f32", "v": [0]}).fields["out"], [0])

    def test_functional_schema_does_not_accept_out_or_extra_arguments(self):
        with self.assertRaises(Unsupported):
            self.call("aten.exp", {"t": "f32", "v": [0]}, {"out": {"t": "f32", "v": [0]}})
        with self.assertRaises(Invalid):
            self.adapter.reference("real", {"op": "aten.exp", "args": [{"t": "f32", "v": [0]}, 1]})
        with self.assertRaises(Unsupported):
            self.adapter.reference("unknown", {"op": "aten.exp", "args": [{"t": "f32", "v": [0]}]})


if __name__ == "__main__":
    unittest.main()
