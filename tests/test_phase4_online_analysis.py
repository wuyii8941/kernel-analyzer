from __future__ import annotations

import unittest

from scripts.phase4_online_analysis import pearson, predicted_rate


class Phase4OnlineAnalysisTest(unittest.TestCase):
    def test_predicted_rate_uses_independent_empirical_convolution(self) -> None:
        self.assertAlmostEqual(predicted_rate([0.1, 0.2], [0.15, 0.25]), 0.75)

    def test_pearson(self) -> None:
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
        self.assertIsNone(pearson([1.0], [2.0]))


if __name__ == "__main__":
    unittest.main()
