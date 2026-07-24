from __future__ import annotations

import unittest

from scripts.phase3_calibration import empirical_predicted_fork_rate


class Phase3CalibrationTest(unittest.TestCase):
    def test_empirical_convolution(self) -> None:
        # For delta 0.15, one of two margins is below it; for 0.25, both are.
        self.assertAlmostEqual(empirical_predicted_fork_rate([0.1, 0.2], [0.15, 0.25]), 0.75)

    def test_empty_distribution(self) -> None:
        self.assertEqual(empirical_predicted_fork_rate([], [0.1]), 0.0)


if __name__ == "__main__":
    unittest.main()
