from __future__ import annotations

import unittest

from theory_oracle.analyze_qwen3_u2_direction_stability_v0_1 import (
    direction_diagnostics_from_gram,
)


def gram(vectors: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a * b for a, b in zip(left, right)) for right in vectors]
        for left in vectors
    ]


class U2DirectionStabilityTests(unittest.TestCase):
    def test_identical_trajectory_fields_have_stable_crossfit_direction(self) -> None:
        result = direction_diagnostics_from_gram(
            gram([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0], [2.0, -1.0]])
        )
        self.assertAlmostEqual(result["minimum_full_vs_leave_one_out_cosine"], 1.0)
        self.assertAlmostEqual(result["crossfit_projection_sample_variance"], 0.0)
        self.assertGreater(result["crossfit_projection_mean"], 0.0)

    def test_zero_full_mean_does_not_create_a_direction_from_norms(self) -> None:
        result = direction_diagnostics_from_gram(
            gram([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        )
        self.assertEqual(result["full_calibration_mean_norm"], 0.0)
        self.assertFalse(result["direction_defined_algebraically"])
        # Cross-fit planning values can exist even though there is no final
        # full-calibration direction; they must not instantiate one.
        self.assertIsNotNone(result["crossfit_projection_sample_variance"])

    def test_in_sample_and_crossfit_projection_are_not_interchangeable(self) -> None:
        result = direction_diagnostics_from_gram(
            gram([[3.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
        )
        rows = result["leave_one_out_rows"]
        self.assertTrue(
            any(
                row["held_out_projection_on_leave_one_out_direction"]
                != row["held_out_projection_on_full_in_sample_direction"]
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
