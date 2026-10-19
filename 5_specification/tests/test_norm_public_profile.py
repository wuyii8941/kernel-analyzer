"""The public autograd boundary does not promise internal saved statistics."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry, schema


class NormPublicProfile(unittest.TestCase):
    def setUp(self):
        self.inp = {"op": "rms_norm", "x": [[2, 4]], "normalized_shape": [2],
                    "weight": [1, 3], "eps": 6, "upstream": [[2, -1]]}

    def test_profile_declares_only_public_outputs(self):
        contract = registry.load("NORM")
        self.assertEqual(set(schema.profile_outputs(contract.data, "torch")), {"y", "saved", "grad"})
        self.assertEqual(set(schema.profile_outputs(contract.data, "torch_autograd")), {"y", "grad"})

    def test_reference_filters_internal_saved_state(self):
        contract = registry.load("NORM")
        full = contract.adapter.reference("torch", self.inp).fields
        public = contract.adapter.reference("torch_autograd", self.inp).fields
        self.assertEqual(set(full), {"y", "saved", "grad"})
        self.assertEqual(public, {"y": full["y"], "grad": full["grad"]})

    def test_runtime_query_enforces_selected_interface(self):
        with patch.object(registry, "activation", return_value=(True, [], {})):
            result = registry.query("NORM", "torch_autograd", self.inp)
        self.assertEqual(result.outcome, "value", result.note)
        self.assertEqual(set(result.fields), {"y", "grad"})

    def test_complete_comparison_rejects_internal_or_missing_fields(self):
        contract = registry.load("NORM")
        expected = contract.adapter.reference("torch_autograd", self.inp).fields
        self.assertTrue(contract.adapter.check_output("torch_autograd", self.inp, expected))
        self.assertFalse(contract.adapter.check_output("torch_autograd", self.inp,
                                                       dict(expected, saved={"rstd": [1]})))
        self.assertFalse(contract.adapter.check_output("torch_autograd", self.inp, {"y": expected["y"]}))


if __name__ == "__main__":
    unittest.main()
