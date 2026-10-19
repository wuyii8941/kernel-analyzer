"""Admission boundary for the reviewed finite_rne CAST profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


class CastProfileDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("CAST").adapter

    def outcome(self, op, args, kwargs=None, profile="finite_rne"):
        call = {"op": op, "args": args}
        if kwargs is not None:
            call["kwargs"] = kwargs
        return evaluate(self.adapter.reference, profile, call).outcome

    def test_common_floating_source_and_target_types_are_admitted(self):
        for source in ("f16", "bf16", "f32", "f64"):
            for target in ("f16", "bf16", "f32", "f64"):
                with self.subTest(source=source, target=target):
                    self.assertEqual(self.outcome(
                        "prims.convert_element_type", [{"t": source, "v": [0, 1, "-1/2"]}, target]), "value")

    def test_unreviewed_types_nonfinite_and_overflow_are_refused(self):
        for source in ("i64", "b8", "f8e4m3", "f8e5m2", "complex64"):
            with self.subTest(source=source):
                self.assertEqual(self.outcome(
                    "prims.convert_element_type", [{"t": source, "v": [1]}, "f32"]), "unsupported")
        for target in ("i64", "b8", "f8e4m3", "f8e5m2", "complex64"):
            with self.subTest(target=target):
                self.assertEqual(self.outcome(
                    "prims.convert_element_type", [{"t": "f32", "v": [1]}, target]), "unsupported")
        self.assertEqual(self.outcome(
            "prims.convert_element_type", [{"t": "f32", "v": ["inf"]}, "bf16"]), "unsupported")
        self.assertEqual(self.outcome(
            "prims.convert_element_type", [{"t": "f32", "v": [70000]}, "f16"]), "unsupported")
        self.assertEqual(self.outcome(
            "prims.convert_element_type", [{"t": "f32", "v": [float("nan")]}, "f16"]), "invalid_input")

    def test_source_record_must_be_a_real_value_of_its_declared_dtype(self):
        for source, value in (("f16", "1/3"), ("bf16", "257/256"), ("f32", "1/10")):
            with self.subTest(source=source, value=value):
                self.assertEqual(self.outcome(
                    "prims.convert_element_type", [{"t": source, "v": [value]}, "f64"]), "invalid_input")

    def test_selected_to_copy_schema_is_enforced(self):
        x = {"t": "f32", "v": [1]}
        self.assertEqual(self.outcome("aten._to_copy", [x], {"dtype": "f16"}), "value")
        self.assertEqual(self.outcome("aten._to_copy", [x, "f16"]), "invalid_input")
        self.assertEqual(self.outcome("aten._to_copy", [x]), "unsupported")
        for kwargs in (
            {"dtype": "f16", "device": "cuda"},
            {"dtype": "f16", "layout": "strided"},
            {"dtype": "f16", "pin_memory": True},
            {"dtype": "f16", "memory_format": "contiguous"},
            {"dtype": "f16", "non_blocking": True},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.outcome("aten._to_copy", [x], kwargs), "unsupported")

    def test_profile_choice_is_explicit(self):
        x = {"t": "f32", "v": [1]}
        self.assertEqual(self.outcome("prims.convert_element_type", [x, "f16"], profile="ieee"), "unsupported")


if __name__ == "__main__":
    unittest.main()
