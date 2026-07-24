from __future__ import annotations

import unittest

from scripts.phase1_logprob_pipeline import summarize_by_position


class Phase1StatsTest(unittest.TestCase):
    def test_position_buckets(self) -> None:
        rows = [
            {"token_index": 0, "logprob_delta": 0.1},
            {"token_index": 31, "logprob_delta": 0.2},
            {"token_index": 32, "logprob_delta": 0.3},
        ]
        buckets = summarize_by_position(rows)
        self.assertEqual([row["token_positions"] for row in buckets], ["0-31", "32-63"])
        self.assertEqual([row["n"] for row in buckets], [2, 1])


if __name__ == "__main__":
    unittest.main()
