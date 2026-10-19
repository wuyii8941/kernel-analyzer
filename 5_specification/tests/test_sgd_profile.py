"""Admission boundaries and complete post-state checks for reviewed public SGD."""
from fractions import Fraction
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import registry
from kernel_spec.num import Invalid, Unsupported


def base(**updates):
    value = {
        "state": {"theta": [1, -2], "b": [1, -1]},
        "gradient": [2, 1],
        "lr": "1/8",
        "momentum": "1/2",
        "dampening": 0,
        "weight_decay": "1/4",
        "nesterov": False,
        "maximize": False,
    }
    value.update(updates)
    return value


class SGDProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("SGD").adapter

    def test_public_options_are_explicit(self):
        expected = self.adapter.reference("torch210", base()).fields
        for foreach, fused in ((False, False), (True, False), (None, None), (False, True)):
            inp = base(foreach=foreach, fused=fused, differentiable=False)
            self.assertEqual(self.adapter.reference("torch210", inp).fields, expected)
        with self.assertRaises(Invalid):
            self.adapter.reference("torch210", base(foreach=True, fused=True))
        with self.assertRaises(Invalid):
            self.adapter.reference("torch210", base(foreach=1))
        with self.assertRaises(Invalid):
            self.adapter.reference("torch210", base(differentiable=None))
        with self.assertRaises(Unsupported):
            self.adapter.reference("torch210", base(differentiable=True))

    def test_internal_and_undeclared_fields_are_refused(self):
        for name in ("grad_scale", "found_inf", "closure", "mode"):
            with self.subTest(name=name), self.assertRaises(Unsupported):
                self.adapter.reference("torch210", base(**{name: 1}))
        malformed = base()
        del malformed["gradient"]
        with self.assertRaises(Invalid):
            self.adapter.reference("torch210", malformed)

    def test_complete_output_checks_parameter_and_buffer(self):
        inp = base()
        expected = self.adapter.reference("torch210", inp).fields
        self.assertTrue(self.adapter.check_output("torch210", inp, expected))
        records = {
            "theta": {"t": "f32", "v": expected["theta"], "shape": [2]},
            "b": {"t": "f32", "v": expected["b"], "shape": [2]},
        }
        self.assertTrue(self.adapter.check_output("torch210", inp, records))
        wrong_buffer = {"theta": expected["theta"], "b": [0, 0]}
        self.assertFalse(self.adapter.check_output("torch210", inp, wrong_buffer))
        self.assertFalse(self.adapter.check_output("torch210", inp, {"theta": expected["theta"]}))
        records["b"] = {"t": "i64", "v": expected["b"], "shape": [2]}
        self.assertFalse(self.adapter.check_output("torch210", inp, records))

    def test_first_step_none_zero_and_stale_buffer(self):
        first = base(state={"theta": [1], "b": []}, gradient=[2], momentum="3/4", dampening="1/2",
                     weight_decay=0, lr="1/8")
        self.assertEqual(self.adapter.reference("torch210", first).fields,
                         {"theta": [Fraction(3, 4)], "b": [2]})

        skipped = base(state={"theta": [1], "b": [2]}, gradient=None, weight_decay=1)
        self.assertEqual(self.adapter.reference("torch210", skipped).fields,
                         {"theta": [1], "b": [2]})
        zero = base(state={"theta": [1], "b": [2]}, gradient=[0], momentum="1/2",
                    dampening=0, weight_decay=0, lr="1/8")
        self.assertEqual(self.adapter.reference("torch210", zero).fields["b"], [1])
        stale = base(state={"theta": [1], "b": [3]}, gradient=[1], momentum=0,
                     dampening=0, weight_decay=0, lr="1/2")
        self.assertEqual(self.adapter.reference("torch210", stale).fields, {"theta": [Fraction(1, 2)], "b": [3]})


if __name__ == "__main__":
    unittest.main()
