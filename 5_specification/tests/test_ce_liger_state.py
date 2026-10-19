"""Liger fused cross entropy has an observable logits-storage post-state."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry, schema
from kernel_spec.num import ends


def contains(value, target):
    lo, hi = ends(value)
    return lo <= target <= hi


class LigerCrossEntropyState(unittest.TestCase):
    def setUp(self):
        self.contract = registry.load("CE_EXT")
        self.base = {"logits": [[0, 0]], "target": [0], "weights": None, "eps": 0,
                     "ignore_index": -100}

    def test_other_profiles_do_not_gain_an_unpromised_post_state(self):
        self.assertEqual(set(schema.profile_outputs(self.contract.data, "probability")), {"loss", "gradient"})
        self.assertEqual(set(schema.profile_outputs(self.contract.data, "liger_inplace_index_unsmoothed")),
                         {"loss", "gradient", "post_logits"})

    def test_scalar_backward_scales_public_storage_in_place(self):
        inp = dict(self.base, reduction="sum", upstream=2)
        fields = self.contract.adapter.reference("liger_inplace_index_unsmoothed", inp).fields
        self.assertEqual(fields["gradient"], fields["post_logits"])
        self.assertTrue(contains(fields["gradient"][0][0], -1))
        self.assertTrue(contains(fields["gradient"][0][1], 1))

    def test_none_backward_leaves_unscaled_derivative_in_public_storage(self):
        inp = dict(self.base, reduction="none", upstream=[2])
        fields = self.contract.adapter.reference("liger_inplace_index_unsmoothed", inp).fields
        self.assertTrue(contains(fields["gradient"][0][0], -1))
        self.assertTrue(contains(fields["gradient"][0][1], 1))
        self.assertTrue(contains(fields["post_logits"][0][0], -1 / 2))
        self.assertTrue(contains(fields["post_logits"][0][1], 1 / 2))

    def test_complete_comparison_cannot_drop_post_state(self):
        inp = dict(self.base, reduction="sum", upstream=2)
        ref = self.contract.adapter.reference("liger_inplace_index_unsmoothed", inp)
        incomplete = {name: value for name, value in ref.fields.items() if name != "post_logits"}
        with patch.object(registry, "query", return_value=ref):
            verdict, note, _ = registry.check_output("CE_EXT", "liger_inplace_index_unsmoothed", inp, incomplete)
        self.assertEqual(verdict, "contradiction")
        self.assertIn("output fields", note)

    def test_runtime_query_keeps_post_state(self):
        inp = dict(self.base, reduction="sum", upstream=2)
        with patch.object(registry, "activation", return_value=(True, [], {})):
            result = registry.query("CE_EXT", "liger_inplace_index_unsmoothed", inp)
        self.assertEqual(result.outcome, "value", result.note)
        self.assertEqual(set(result.fields), {"loss", "gradient", "post_logits"})


if __name__ == "__main__":
    unittest.main()
