import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "theory_oracle/evaluate_qwen3_grpo_bhn_v0_2.py"
SPEC = importlib.util.spec_from_file_location("qwen_bhn", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenBHNTest(unittest.TestCase):
    def rows(self, n=512):
        return [
            {
                "state_id": "s0",
                "flat_index": index,
                "logp_ref_first": -1.0,
                "logp_ref_second": -1.0,
                "logp_alt_first": -1.0 + index * 1e-6,
                "logp_alt_second": -1.0 + index * 1e-6,
            }
            for index in range(n)
        ]

    def test_state_field_alignment(self):
        states = MODULE.state_arrays(list(reversed(self.rows())))
        self.assertEqual(set(states), {"s0"})
        ref1, ref2, cand1, cand2 = states["s0"]
        self.assertEqual(ref1.shape, (512,))
        self.assertAlmostEqual(float(cand1[-1] - ref1[-1]), 511e-6, places=15)
        self.assertTrue((ref1 == ref2).all())
        self.assertTrue((cand1 == cand2).all())

    def test_missing_token_invalidates_state(self):
        with self.assertRaises(ValueError):
            MODULE.state_arrays(self.rows(511))


if __name__ == "__main__":
    unittest.main()
