from __future__ import annotations

import unittest

from forkcert.stats import mean, percentile


class StatsTest(unittest.TestCase):
    def test_mean_empty(self) -> None:
        self.assertEqual(mean([]), 0.0)

    def test_percentile_empty(self) -> None:
        self.assertEqual(percentile([], 99), 0.0)

    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(percentile([0.0, 10.0], 50), 5.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 10), 1.0)

    def test_percentile_unsorted(self) -> None:
        self.assertEqual(percentile([3.0, 1.0, 2.0], 50), 2.0)


if __name__ == "__main__":
    unittest.main()
