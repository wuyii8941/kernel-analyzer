from __future__ import annotations

import unittest

from forkcert.twin_train import trajectory_summary


class TwinTrainTest(unittest.TestCase):
    def test_summarizes_fork_and_no_fork_intervals(self) -> None:
        rows = [
            {"optimizer_step": 0, "fork_count": 0, "weight_divergence": 0.0, "relative_weight_divergence": 0.0},
            {"optimizer_step": 1, "fork_count": 2, "weight_divergence": None},
            {
                "optimizer_step": 5,
                "fork_count": 0,
                "weight_divergence": 0.5,
                "relative_weight_divergence": 0.01,
                "interval_had_fork": True,
            },
            {
                "optimizer_step": 10,
                "fork_count": 0,
                "weight_divergence": 0.6,
                "relative_weight_divergence": 0.012,
                "interval_had_fork": False,
            },
        ]

        summary = trajectory_summary(rows)

        self.assertEqual(summary["total_fork_events"], 2)
        self.assertEqual(summary["fork_steps"], 1)
        self.assertEqual(summary["fork_intervals"], 1)
        self.assertEqual(summary["no_fork_intervals"], 1)
        self.assertAlmostEqual(summary["mean_divergence_jump_fork_intervals"], 0.5)
        self.assertAlmostEqual(summary["mean_divergence_jump_no_fork_intervals"], 0.1)


if __name__ == "__main__":
    unittest.main()
