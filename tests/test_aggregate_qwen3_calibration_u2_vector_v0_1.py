from __future__ import annotations

import unittest

import torch

from theory_oracle.aggregate_qwen3_calibration_u2_vector_v0_1 import (
    aggregate_one_tensor,
)


def rows_and_values(
    state_delta_repeats: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[dict, dict]:
    rows = {}
    values = {}
    for phase in ("early", "late"):
        rows[phase] = []
        for state_index, repeats in enumerate(state_delta_repeats):
            repeat_rows = []
            for repeat_id, delta in enumerate(repeats, start=1):
                delta_source = f"{phase}-s{state_index}-d{repeat_id}"
                reference_source = f"{phase}-s{state_index}-r{repeat_id}"
                values[(delta_source, "weight")] = torch.tensor([delta], dtype=torch.float64)
                values[(reference_source, "weight")] = torch.tensor(
                    [2.0], dtype=torch.float64
                )
                repeat_rows.append(
                    {
                        "repeat_id": repeat_id,
                        "delta_source": delta_source,
                        "reference_source": reference_source,
                    }
                )
            rows[phase].append({"state_id": f"{phase}-s{state_index}", "repeats": repeat_rows})
    return rows, values


class AggregateU2VectorTests(unittest.TestCase):
    def evaluate(self, patterns):
        rows, values = rows_and_values(patterns)
        return aggregate_one_tensor(
            rows,
            ("early", "late"),
            "weight",
            lambda source, key: values[(source, key)],
        )

    def test_fixed_signed_shift_is_preserved_as_mean_vector(self) -> None:
        mean, profile = self.evaluate(((1.0, 1.0), (1.0, 1.0)))
        self.assertTrue(torch.equal(mean, torch.tensor([1.0], dtype=torch.float64)))
        self.assertAlmostEqual(profile["trajectory_mean_delta_l2"], 1.0)
        self.assertAlmostEqual(
            profile["mean_same_state_paired_effect_variance_trace"], 0.0
        )
        self.assertAlmostEqual(
            profile["mean_within_phase_state_variance_trace_repeat_corrected"],
            0.0,
        )

    def test_state_heterogeneity_does_not_become_runtime_noise(self) -> None:
        mean, profile = self.evaluate(((-1.0, -1.0), (1.0, 1.0)))
        self.assertAlmostEqual(float(mean.item()), 0.0)
        self.assertAlmostEqual(
            profile["mean_within_phase_state_variance_trace_repeat_corrected"],
            2.0,
        )
        self.assertAlmostEqual(
            profile["mean_same_state_paired_effect_variance_trace"], 0.0
        )

    def test_repeat_variability_is_removed_from_state_H(self) -> None:
        mean, profile = self.evaluate(((-1.0, 1.0), (-1.0, 1.0)))
        self.assertAlmostEqual(float(mean.item()), 0.0)
        self.assertAlmostEqual(
            profile["mean_same_state_paired_effect_variance_trace"], 2.0
        )
        self.assertAlmostEqual(
            profile["mean_within_phase_state_variance_trace_repeat_corrected"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
