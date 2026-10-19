"""Do not discard task or layout declarations while decoding a reference call."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec.aten_adapter import decode, make_reference
from kernel_spec.num import evaluate


class AtenAdapterInputs(unittest.TestCase):
    def test_missing_extra_and_duplicate_arguments_are_invalid_calls(self):
        ref = make_reference(["aten._softmax"])
        x = {"t": "f32", "v": [0, 0]}
        for args, kwargs in (([], {}), ([x], {}), ([x, 0], {}), ([x, 0, False, 7], {}), ([x, 0], {"dim": 1})):
            with self.subTest(args=args, kwargs=kwargs):
                result = evaluate(ref, "real", {"op": "aten._softmax", "args": args, "kwargs": kwargs})
                self.assertEqual(result.outcome, "invalid_input")
        result = evaluate(ref, "real", {"op": "aten._softmax", "args": [x, 0], "kwargs": {"mask": [True, False]}})
        self.assertEqual(result.outcome, "unsupported")

    def test_type_error_inside_the_evaluator_is_still_an_implementation_error(self):
        def broken(a, dim, half_to_float):
            raise TypeError("internal evaluator failure")
        ref = make_reference(["aten._softmax"], {"aten._softmax": broken})
        result = evaluate(ref, "real", {"op": "aten._softmax", "args": [{"t": "f32", "v": [0]}, 0, False]})
        self.assertEqual(result.outcome, "implementation_error")
        self.assertIn("internal evaluator failure", result.note)

    def test_extra_task_options_are_not_ignored(self):
        ref = make_reference(["aten.add"])
        call = {"op": "aten.add", "args": [{"t": "f32", "v": [1]}, 2]}
        self.assertEqual(evaluate(ref, "real", call).outcome, "value")
        for extra in ({"alpha": 3}, {"out": {"t": "f32", "v": [0]}}):
            with self.subTest(extra=extra):
                self.assertEqual(evaluate(ref, "real", dict(call, **extra)).outcome, "unsupported")

    def test_extra_tensor_layout_is_not_ignored(self):
        ref = make_reference(["aten.add"])
        for extra in ({"stride": [0]}, {"storage_offset": 1}, {"device": "cuda"}):
            tensor = dict(t="f32", v=[1, 2], **extra)
            with self.subTest(extra=extra):
                self.assertEqual(evaluate(ref, "real", {"op": "aten.add", "args": [tensor, 0]}).outcome,
                                 "unsupported")

    def test_malformed_call_records_are_invalid(self):
        ref = make_reference(["aten.add"])
        cases = [None, [], {}, {"op": []}, {"op": "aten.add", "args": "12"},
                 {"op": "aten.add", "kwargs": []}, {"op": "aten.add", "kwargs": {1: 2}}]
        for call in cases:
            with self.subTest(call=call):
                self.assertEqual(evaluate(ref, "real", call).outcome, "invalid_input")

    def test_empty_tensor_shape_and_dtype_decoding_are_preserved(self):
        self.assertEqual(decode({"t": "f32", "v": [], "shape": [0, 3]}).a.shape, (0, 3))
        self.assertEqual(decode({"dtype": "f32"}), "f32")


if __name__ == "__main__":
    unittest.main()
