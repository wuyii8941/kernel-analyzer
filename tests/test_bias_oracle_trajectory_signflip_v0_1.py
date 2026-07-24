from __future__ import annotations

import unittest

from theory_oracle.bias_oracle_trajectory_signflip_v0_1 import (
    trajectory_signflip_test,
)


class TrajectorySignFlipTests(unittest.TestCase):
    def test_symmetric_zero_mean_pattern_is_not_promoted(self) -> None:
        result = trajectory_signflip_test(
            [-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5],
            null_center=0.0,
        )
        self.assertGreater(result["two_sided_p_value"], 0.05)
        self.assertEqual(result["trajectories"], 8)

    def test_fixed_shift_is_detected_at_exact_eight_trajectory_resolution(self) -> None:
        result = trajectory_signflip_test([1.0] * 8, null_center=0.0)
        self.assertAlmostEqual(result["two_sided_p_value"], 2 / 256)
        self.assertEqual(result["method"], "EXACT_RADEMACHER_SIGN_FLIP_STUDENTIZED")

    def test_measurement_floor_is_the_tested_null_center(self) -> None:
        result = trajectory_signflip_test([1.1] * 8, null_center=1.0)
        self.assertAlmostEqual(result["two_sided_p_value"], 2 / 256)
        self.assertEqual(result["null_center"], 1.0)

    def test_monte_carlo_path_is_reproducible(self) -> None:
        effects = [0.1 + index / 100.0 for index in range(17)]
        first = trajectory_signflip_test(
            effects, null_center=0.0, monte_carlo_draws=1999, monte_carlo_seed=7
        )
        second = trajectory_signflip_test(
            effects, null_center=0.0, monte_carlo_draws=1999, monte_carlo_seed=7
        )
        self.assertEqual(first["two_sided_p_value"], second["two_sided_p_value"])
        self.assertEqual(
            first["method"], "MONTE_CARLO_RADEMACHER_SIGN_FLIP_STUDENTIZED"
        )


if __name__ == "__main__":
    unittest.main()
