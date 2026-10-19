import sys
import unittest
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import registry
from kernel_spec.num import evaluate


class OptimState4bitCopy(unittest.TestCase):
    def setUp(self):
        self.contract = registry.load("OPT4BIT")
        self.adapter = self.contract.adapter
        self.profile = "torchao016_state_copy_f32"
        self.signed = {
            "src": [1, "1/2", -1, "1/8"],
            "input_dtype": "f32",
            "signed": True,
            "block_size": 2,
            "qmap": list(registry.load("OPT4BIT").adapter.QMAP_SIGNED),
        }

    def test_packing_scales_and_lookup_table_are_physical_outputs(self):
        out = self.adapter.reference(self.profile, self.signed).fields
        self.assertEqual(out["codes"], [0xFC, 0x0A])
        self.assertEqual(out["scale"], [1, 1])
        self.assertEqual(self.adapter.QMAP_SIGNED[0], Fraction(-14889779, 16777216))
        self.assertEqual(self.adapter.QMAP_SIGNED[7], 0)
        self.assertEqual(self.adapter.QMAP_SIGNED[15], 1)

    def test_unsigned_map_has_no_zero_and_zero_value_uses_lowest_code(self):
        inp = self.signed | {"src": [1, "1/2", 0, "3/32"], "signed": False,
                             "qmap": list(self.adapter.QMAP_UNSIGNED)}
        out = self.adapter.reference(self.profile, inp).fields
        self.assertEqual(out["codes"], [0xF7, 0x0F])
        self.assertEqual(out["scale"], [1, Fraction(3, 32)])
        self.assertEqual(self.adapter.QMAP_UNSIGNED, [Fraction(k, 16) for k in range(1, 17)])

    def test_nearest_code_and_zero_scale_floor(self):
        rounded = self.adapter.reference(
            self.profile, self.signed | {"src": [1, "13/16"]}).fields
        self.assertEqual(rounded["codes"], [0xFE])
        zero = self.adapter.reference(
            self.profile, self.signed | {"src": [0, 0]}).fields
        self.assertEqual(zero["codes"], [0x77])
        self.assertEqual(zero["scale"], [Fraction(2305843, 2305843009213693952)])

    def test_complete_output_and_tensor_metadata_are_required(self):
        expected = self.adapter.reference(self.profile, self.signed).fields
        observed = {
            "codes": {"t": "u8", "shape": [2], "v": expected["codes"]},
            "scale": {"t": "f32", "shape": [2], "v": expected["scale"]},
        }
        self.assertTrue(self.adapter.check_output(self.profile, self.signed, observed))
        for field in expected:
            bad = deepcopy(observed)
            del bad[field]
            self.assertFalse(self.adapter.check_output(self.profile, self.signed, bad))
        bad = deepcopy(observed)
        bad["codes"]["v"][0] ^= 1
        self.assertFalse(self.adapter.check_output(self.profile, self.signed, bad))
        bad = deepcopy(observed)
        bad["scale"]["shape"] = [1, 2]
        self.assertFalse(self.adapter.check_output(self.profile, self.signed, bad))
        bad = deepcopy(observed)
        bad["qmap"] = {"t": "f32", "shape": [16], "v": self.adapter.QMAP_SIGNED}
        self.assertFalse(self.adapter.check_output(self.profile, self.signed, bad))

    def test_unreviewed_domains_are_refused(self):
        cases = [
            (self.signed | {"input_dtype": "f16"}, "unsupported"),
            (self.signed | {"qmap": [0] * 16}, "unsupported"),
            (self.signed | {"signed": False, "src": [-1, 0],
                            "qmap": list(self.adapter.QMAP_UNSIGNED)}, "unsupported"),
            (self.signed | {"block_size": 3, "src": [1, 0, 0, 0, 0, 0]}, "unsupported"),
            (self.signed | {"src": [Fraction(1, 1 << 149), 0]}, "unsupported"),
            (self.signed | {"src": ["1/3", 0]}, "invalid_input"),
            (self.signed | {"src": [1, 0, 0]}, "invalid_input"),
        ]
        for inp, outcome in cases:
            with self.subTest(inp=inp):
                self.assertEqual(evaluate(self.adapter.reference, self.profile, inp).outcome, outcome)

    def test_reviewed_contract_is_runtime_admitted(self):
        expected = self.adapter.reference(self.profile, self.signed).fields
        verdict, note, labels = registry.check_output("OPT4BIT", self.profile, self.signed, expected)
        self.assertEqual(verdict, "consistent", note)
        self.assertEqual(labels["implementation_status"], "validated")


if __name__ == "__main__":
    unittest.main()
