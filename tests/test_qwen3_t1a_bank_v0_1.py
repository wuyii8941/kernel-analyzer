from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


THEORY_DIR = Path(__file__).resolve().parents[1] / "theory_oracle"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from generate_qwen3_t1a_bank_v0_1 import (  # noqa: E402
    arithmetic_fields,
    derive_seed,
    group_advantages,
    numeric_reward,
)


class T1aBankTests(unittest.TestCase):
    def test_frozen_seed_rule(self) -> None:
        seed, digest = derive_seed(
            "qwen3-bias-oracle-t1a-v0.1/Q-R/legacy-B/step29-natural-transition"
        )
        self.assertEqual(seed, 1207123700)
        self.assertEqual(
            digest,
            "47f33ef41c77a935dfd90b7dde455afa5deacd477644383889222afded321072",
        )

    def test_arithmetic_and_reward_match_subject_rule(self) -> None:
        fields = arithmetic_fields(9000)
        reward, predicted, exact = numeric_reward(
            f"work; final {fields['result']}", fields["result"]
        )
        self.assertEqual(predicted, float(fields["result"]))
        self.assertTrue(exact)
        self.assertEqual(reward, 2.0)

    def test_advantages_center_and_zero_for_tied_group(self) -> None:
        values = group_advantages(torch, [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(sum(values), 0.0, places=6)
        self.assertEqual(group_advantages(torch, [1.0, 1.0, 1.0, 1.0]), [0.0] * 4)


if __name__ == "__main__":
    unittest.main()
