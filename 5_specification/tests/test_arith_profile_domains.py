import importlib.util
import unittest
from pathlib import Path

from kernel_spec.num import Invalid, Undefined, Unsupported


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "adapters" / "ARITH.py"
    spec = importlib.util.spec_from_file_location("reviewed_arith_domain_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArithmeticProfileDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def call(self, op, args, kwargs=None, profile="real"):
        inp = {"op": op, "args": args}
        if kwargs is not None:
            inp["kwargs"] = kwargs
        return self.adapter.reference(profile, inp)

    def test_all_reviewed_dtypes_and_empty_broadcast_shapes_are_supported(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            with self.subTest(dtype=dtype):
                result = self.call("aten.add", [{"t": dtype, "v": [1, 2]}, "1/2"])
                self.assertEqual(result.fields["out"], [3 / 2, 5 / 2])
        empty = self.call("aten.mul", [
            {"t": "f32", "v": [[], []]},
            {"t": "f32", "v": [], "shape": [1, 0]},
        ])
        self.assertEqual(empty.fields["out"], [[], []])

    def test_physical_tensor_values_must_match_the_declared_dtype(self):
        for dtype in ("f16", "bf16", "f32", "f64"):
            with self.subTest(dtype=dtype), self.assertRaises(Invalid):
                self.call("aten.add", [{"t": dtype, "v": ["1/3"]}, 1])
        with self.assertRaises(Invalid):
            self.call("aten.add", [{"t": "f32", "v": [float("nan")]}, 1])

    def test_unreviewed_types_nonfinite_values_and_mixed_dtypes_are_refused(self):
        values = (
            {"t": "i64", "v": [1]},
            {"t": "b8", "v": [True]},
            {"t": "f8e4m3", "v": [1]},
            {"t": "f32", "v": [float("inf")]},
            {"t": "complex64", "v": [1]},
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises((Invalid, Unsupported)):
                self.call("aten.mul", [value, 2])
        with self.assertRaises(Unsupported):
            self.call("aten.add", [{"t": "f16", "v": [1]}, {"t": "f32", "v": [2]}])

    def test_scalars_bounds_and_internal_fma_stay_inside_the_reviewed_scope(self):
        with self.assertRaises(Invalid):
            self.call("aten.add", [{"t": "f32", "v": [1]}, "1/3"])
        with self.assertRaises(Unsupported):
            self.call("aten.clamp_min", [{"t": "f32", "v": [1]}, {"t": "f32", "v": [0]}])
        with self.assertRaises(Unsupported):
            self.call("prims.fma", [
                {"t": "f32", "v": [[1], [2]]},
                {"t": "f32", "v": [3, 4]},
                {"t": "f32", "v": [[5], [6]]},
            ])

    def test_zero_domains_and_division_modes_are_explicit(self):
        for op, args in (
            ("aten.div", [{"t": "f32", "v": [1]}, 0]),
            ("aten.floor_divide", [{"t": "f32", "v": [1]}, 0]),
            ("aten.reciprocal", [{"t": "f32", "v": [0]}]),
        ):
            with self.subTest(op=op), self.assertRaises(Undefined):
                self.call(op, args)
        with self.assertRaises(Invalid):
            self.call("aten.div", [{"t": "f32", "v": [1]}, 2], {"rounding_mode": "nearest"})

    def test_selected_functional_schemas_reject_other_overloads(self):
        x = {"t": "f32", "v": [1]}
        with self.assertRaises(Invalid):
            self.call("aten.add", [2, x])
        with self.assertRaises(Invalid):
            self.call("aten.add", [x, 2, 3])
        with self.assertRaises(Invalid):
            self.call("aten.div", [x, 2, "floor"])
        with self.assertRaises(Unsupported):
            self.call("aten.add", [x, 2], {"out": x})
        with self.assertRaises(Invalid):
            self.call("aten.clamp", [x])
        with self.assertRaises(Unsupported):
            self.call("aten.pow", [x, 2])
        with self.assertRaises(Unsupported):
            self.call("aten.add", [x, 2], profile="unknown")

    def test_boolean_conditions_are_required_and_broadcast_checked(self):
        x = {"t": "f32", "v": [1, 2]}
        with self.assertRaises(Invalid):
            self.call("aten.where", [{"t": "i64", "v": [1, 0]}, x, x])
        with self.assertRaises(Invalid):
            self.call("aten.where", [{"t": "b8", "v": [1, 0]}, x, x])
        with self.assertRaises(Invalid):
            self.call("aten.masked_fill", [x, {"t": "f32", "v": [0, 1]}, 0])
        with self.assertRaises(Invalid):
            self.call("aten.where", [{"t": "b8", "v": [True, False, True]}, x, x])


if __name__ == "__main__":
    unittest.main()
