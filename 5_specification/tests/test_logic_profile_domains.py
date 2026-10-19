import importlib.util
import unittest
from pathlib import Path

from kernel_spec.num import Invalid, Unsupported


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "LOGIC.py"
    spec = importlib.util.spec_from_file_location("reviewed_logic_domain_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogicProfileDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def call(self, op, args, kwargs=None, profile="real"):
        inp = {"op": op, "args": args}
        if kwargs is not None:
            inp["kwargs"] = kwargs
        return self.adapter.reference(profile, inp)

    def test_physical_floating_and_boolean_inputs_are_supported(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            with self.subTest(dtype=dtype):
                result = self.call("aten.ge", [{"t": dtype, "v": [-1, 0, "1/2"]}, 0])
                self.assertEqual(result.fields["out"], [False, True, True])
        result = self.call("aten.eq", [
            {"t": "b8", "v": [False, True]}, {"t": "b8", "v": [False, False]},
        ])
        self.assertEqual(result.fields["out"], [True, False])

    def test_unrepresentable_and_nonliteral_boolean_records_are_invalid(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            with self.subTest(dtype=dtype), self.assertRaises(Invalid):
                self.call("aten.eq", [{"t": dtype, "v": ["1/3"]}, 0])
        for value in ([0, 1], ["false", "true"]):
            with self.subTest(value=value), self.assertRaises(Invalid):
                self.call("aten.logical_not", [{"t": "b8", "v": value}])

    def test_unreviewed_types_mixed_dtypes_and_special_values_are_refused(self):
        for value in (
            {"t": "i64", "v": [1]},
            {"t": "f8e4m3", "v": [1]},
            {"t": "complex64", "v": [1]},
            {"t": "f32", "v": [float("inf")]},
            {"t": "f32", "v": [float("nan")]},
        ):
            with self.subTest(value=value), self.assertRaises((Invalid, Unsupported)):
                self.call("aten.eq", [value, 0])
        with self.assertRaises(Unsupported):
            self.call("aten.logical_and", [
                {"t": "f16", "v": [1]}, {"t": "f32", "v": [1]},
            ])

    def test_scalar_comparisons_follow_the_first_tensor_dtype(self):
        with self.assertRaises(Invalid):
            self.call("aten.eq", [{"t": "f32", "v": [0]}, "1/3"])
        with self.assertRaises(Invalid):
            self.call("aten.eq", [{"t": "b8", "v": [True]}, 1])
        with self.assertRaises(Invalid):
            self.call("aten.eq", [1, {"t": "f32", "v": [1]}])

    def test_selected_schemas_reject_other_overloads(self):
        x = {"t": "f32", "v": [0, 1]}
        with self.assertRaises(Invalid):
            self.call("aten.eq", [x])
        with self.assertRaises(Invalid):
            self.call("aten.logical_not", [x, x])
        with self.assertRaises(Unsupported):
            self.call("aten.eq", [x, 0], {"out": {"t": "b8", "v": [True, False]}})
        with self.assertRaises(Unsupported):
            self.call("aten.bitwise_and", [{"t": "i64", "v": [1]}, {"t": "i64", "v": [1]}])

    def test_truth_reduction_dimension_and_keepdim_are_strict(self):
        x = {"t": "f32", "v": [[0, 1], [0, 0]]}
        self.assertEqual(self.call("aten.any", [x, 1]).fields["out"], [True, False])
        self.assertEqual(self.call("aten.all", [x, 1, True]).fields["out"], [[False], [False]])
        with self.assertRaises(Invalid):
            self.call("aten.any", [x, True])
        with self.assertRaises(Unsupported):
            self.call("aten.any", [x, [0, 1]])
        with self.assertRaises(Invalid):
            self.call("aten.all", [x, 1, 1])
        with self.assertRaises(Invalid):
            self.call("aten.any", [x, 2])

    def test_isnan_has_a_separate_ieee_profile(self):
        result = self.call("aten.isnan", [{"t": "f32", "v": [float("nan"), float("inf"), 0]}],
                           profile="ieee_values")
        self.assertEqual(result.fields["out"], [True, False, False])
        with self.assertRaises(Unsupported):
            self.call("aten.isnan", [{"t": "f32", "v": [float("nan")]}])
        with self.assertRaises(Unsupported):
            self.call("aten.eq", [{"t": "f32", "v": [0]}, 0], profile="ieee_values")
        with self.assertRaises(Unsupported):
            self.call("aten.isnan", [{"t": "b8", "v": [False]}], profile="ieee_values")


if __name__ == "__main__":
    unittest.main()
