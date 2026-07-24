from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


THEORY_DIR = Path(__file__).resolve().parents[1] / "theory_oracle"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from evaluate_qwen3_common_t1a_smoke_v0_1 import grpo_components  # noqa: E402


class CommonT1aSmokeTests(unittest.TestCase):
    def test_identity_ratio_reduces_to_negative_advantage(self) -> None:
        old = torch.zeros((4, 3), dtype=torch.float32)
        new = old.clone()
        advantages = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float32)
        mask = torch.ones_like(old)
        loss, per_completion, clipped = grpo_components(
            torch, new, old, advantages, mask, 0.2
        )
        self.assertTrue(torch.allclose(per_completion, -advantages))
        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertFalse(bool(clipped.any()))

    def test_directional_clipping_uses_advantage_sign(self) -> None:
        old = torch.zeros((2, 1), dtype=torch.float32)
        new = torch.log(torch.tensor([[1.3], [0.7]], dtype=torch.float32))
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
        mask = torch.ones_like(old)
        _, _, clipped = grpo_components(torch, new, old, advantages, mask, 0.2)
        self.assertEqual(clipped.tolist(), [[True], [True]])


if __name__ == "__main__":
    unittest.main()
