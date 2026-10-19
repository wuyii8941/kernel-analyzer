import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import registry
from kernel_spec.num import evaluate


class F2ContractMigration(unittest.TestCase):
    def setUp(self):
        self.contract = registry.load("F2_MX")
        self.adapter = self.contract.adapter
        self.profile = "torchao016_mxfp8_floor"
        self.inp = {
            "x": [[500, 1, "-3/4"] + [0] * 29],
            "input_dtype": "f32",
            "elem_format": "e4m3",
            "block_size": 32,
            "scaling_mode": "floor",
            "is_swizzled_scales": False,
        }

    def test_versioned_floor_profile_returns_complete_result(self):
        result = self.adapter.reference(self.profile, self.inp).fields
        self.assertEqual(result["scale_codes"], [[127]])
        self.assertEqual(result["elements"][0][:3], [448, 1, -3 / 4])
        self.assertEqual(len(result["elements"][0]), 32)
        self.assertTrue(self.adapter.check_output(self.profile, self.inp, result))

    def test_every_output_coordinate_and_tensor_record_metadata_is_checked(self):
        result = self.adapter.reference(self.profile, self.inp).fields
        observed = {
            "scale_codes": {"t": "u8", "shape": [1, 1], "v": result["scale_codes"]},
            "elements": {"t": "f8e4m3", "shape": [1, 32], "v": result["elements"]},
        }
        self.assertTrue(self.adapter.check_output(self.profile, self.inp, observed))
        for field in ("scale_codes", "elements"):
            bad = deepcopy(observed)
            del bad[field]
            self.assertFalse(self.adapter.check_output(self.profile, self.inp, bad))
        bad = deepcopy(observed)
        bad["elements"]["v"][0].pop()
        self.assertFalse(self.adapter.check_output(self.profile, self.inp, bad))
        bad = deepcopy(observed)
        bad["elements"]["t"] = "f8e5m2"
        self.assertFalse(self.adapter.check_output(self.profile, self.inp, bad))
        bad = deepcopy(observed)
        bad["scale_codes"]["v"][0][0] += 1
        self.assertFalse(self.adapter.check_output(self.profile, self.inp, bad))

    def test_other_legal_modes_and_layouts_are_refused_instead_of_guessed(self):
        for update in (
            {"scaling_mode": "rceil"},
            {"scaling_mode": "even"},
            {"is_swizzled_scales": True},
            {"elem_format": "e2m1"},
            {"block_size": 16},
        ):
            result = evaluate(self.adapter.reference, self.profile, self.inp | update)
            self.assertEqual(result.outcome, "unsupported")

    def test_code_zero_uses_released_torchao_divisor_behavior(self):
        tiny = "1/170141183460469231731687303715884105728"  # 2^-127
        inp = self.inp | {"x": [[tiny] * 32]}
        result = self.adapter.reference(self.profile, inp).fields
        self.assertEqual(result["scale_codes"], [[0]])
        self.assertEqual(result["elements"], [[1 / 2] * 32])

    def test_reviewed_contract_is_admitted_after_retained_review_copy(self):
        verdict, _, labels = registry.check_output(
            "F2_MX", self.profile, self.inp,
            self.adapter.reference(self.profile, self.inp).fields,
        )
        self.assertEqual(verdict, "consistent")
        self.assertEqual(labels["implementation_status"], "validated")


if __name__ == "__main__":
    unittest.main()
