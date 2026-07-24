from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "theory_oracle" / "bias_oracle_population_v0_2.py"
SPEC = importlib.util.spec_from_file_location("bias_population", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EffectRecord = MODULE.EffectRecord


def balanced_records(
    trajectory_effects: list[float],
    *,
    state_pattern: tuple[float, float] = (0.0, 0.0),
    repeat_pattern: tuple[float, float] = (0.0, 0.0),
) -> list[EffectRecord]:
    rows = []
    for trajectory_index, trajectory_effect in enumerate(trajectory_effects):
        for phase in ("early", "middle", "late"):
            for state_index, state_effect in enumerate(state_pattern):
                for repeat_id, repeat_effect in enumerate(repeat_pattern):
                    rows.append(
                        EffectRecord(
                            trajectory_id=f"t{trajectory_index}",
                            phase=phase,
                            state_id=f"{phase}-s{state_index}",
                            repeat_id=repeat_id,
                            effect=trajectory_effect + state_effect + repeat_effect,
                        )
                    )
    return rows


class BiasOraclePopulationTest(unittest.TestCase):
    def estimate(self, records: list[EffectRecord], **kwargs):
        return MODULE.estimate_scalar_population(
            records,
            required_phases=("early", "middle", "late"),
            **kwargs,
        )

    def test_fixed_shift_is_B_not_H_or_N(self) -> None:
        result = self.estimate(balanced_records([0.01] * 8))
        self.assertAlmostEqual(result["B"]["estimate"], 0.01)
        self.assertAlmostEqual(result["H"]["between_trajectory_variance"], 0.0)
        self.assertAlmostEqual(result["H"]["mean_within_phase_state_variance_repeat_corrected"], 0.0)
        self.assertAlmostEqual(result["N"]["mean_same_state_paired_effect_variance"], 0.0)
        self.assertEqual(result["verdicts"]["shift_existence"], "REPRODUCIBLE_AVERAGE_SHIFT")
        self.assertIn(
            "retains reference/candidate covariance",
            result["identification_assumptions"]["paired_noise_covariance"],
        )

    def test_state_heterogeneity_is_not_runtime_variability(self) -> None:
        result = self.estimate(
            balanced_records([0.0] * 8, state_pattern=(-1.0, 1.0))
        )
        self.assertAlmostEqual(result["B"]["estimate"], 0.0)
        self.assertGreater(result["H"]["mean_within_phase_state_variance_repeat_corrected"], 0.0)
        self.assertAlmostEqual(result["N"]["mean_same_state_paired_effect_variance"], 0.0)
        self.assertEqual(
            result["verdicts"]["shift_existence"], "NO_STABLE_AVERAGE_DETECTED"
        )
        self.assertGreater(
            result["H"][
                "mean_within_phase_state_variance_repeat_corrected_unconstrained"
            ],
            0.0,
        )

    def test_repeat_noise_is_N_and_is_removed_from_state_H(self) -> None:
        result = self.estimate(
            balanced_records([0.0] * 8, repeat_pattern=(-1.0, 1.0))
        )
        self.assertAlmostEqual(result["B"]["estimate"], 0.0)
        self.assertAlmostEqual(result["H"]["mean_within_phase_state_variance_repeat_corrected"], 0.0)
        self.assertLess(
            result["H"][
                "mean_within_phase_state_variance_repeat_corrected_unconstrained"
            ],
            0.0,
        )
        self.assertGreater(result["N"]["mean_same_state_paired_effect_variance"], 0.0)

    def test_fewer_than_eight_trajectories_fails_closed(self) -> None:
        result = self.estimate(balanced_records([0.01] * 4))
        self.assertEqual(
            result["verdicts"]["shift_existence"], "INDETERMINATE_TOO_FEW_TRAJECTORIES"
        )

    def test_one_trajectory_does_not_turn_unidentified_variance_into_zero(self) -> None:
        result = self.estimate(balanced_records([0.01]))
        self.assertEqual(result["B"]["estimate"], 0.01)
        self.assertIsNone(result["B"]["standard_error"])
        self.assertEqual(result["B"]["trajectory_t_interval_95"], [None, None])
        self.assertIsNone(result["H"]["between_trajectory_variance"])
        self.assertEqual(
            result["verdicts"]["shift_existence"], "INDETERMINATE_TOO_FEW_TRAJECTORIES"
        )

    def test_states_do_not_inflate_top_level_degrees_of_freedom(self) -> None:
        effects = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        result = self.estimate(balanced_records(effects))
        self.assertEqual(result["B"]["degrees_of_freedom"], 7)
        expected_se = math.sqrt(MODULE._sample_variance(effects) / len(effects))
        self.assertAlmostEqual(result["B"]["standard_error"], expected_se)

    def test_trajectory_weighting_is_not_changed_by_state_count(self) -> None:
        rows = []
        for trajectory, effect, states in (("short", -1.0, 1), ("long", 1.0, 20)):
            for phase in ("early", "middle", "late"):
                for state_index in range(states):
                    for repeat_id in (0, 1):
                        rows.append(
                            EffectRecord(
                                trajectory_id=trajectory,
                                phase=phase,
                                state_id=f"{trajectory}-{phase}-{state_index}",
                                repeat_id=repeat_id,
                                effect=effect,
                            )
                        )
        result = self.estimate(rows)
        self.assertAlmostEqual(result["B"]["estimate"], 0.0)
        self.assertEqual(result["B"]["degrees_of_freedom"], 1)

    def test_phase_weighting_is_not_changed_by_state_count(self) -> None:
        rows = []
        for trajectory_index in range(8):
            for phase, effect, states in (
                ("early", -1.0, 1),
                ("middle", 0.0, 2),
                ("late", 1.0, 20),
            ):
                for state_index in range(states):
                    for repeat_id in (0, 1):
                        rows.append(
                            EffectRecord(
                                trajectory_id=f"t{trajectory_index}",
                                phase=phase,
                                state_id=f"{phase}-{state_index}",
                                repeat_id=repeat_id,
                                effect=effect,
                            )
                        )
        result = self.estimate(rows)
        self.assertAlmostEqual(result["B"]["estimate"], 0.0)

    def test_runtime_N_uses_equal_phase_not_raw_state_count_weighting(self) -> None:
        rows = []
        for trajectory_index in range(8):
            for phase, states, repeat_effects in (
                ("early", 1, (-1.0, 1.0)),
                ("middle", 10, (0.0, 0.0)),
                ("late", 20, (0.0, 0.0)),
            ):
                for state_index in range(states):
                    for repeat_id, effect in enumerate(repeat_effects):
                        rows.append(
                            EffectRecord(
                                trajectory_id=f"t{trajectory_index}",
                                phase=phase,
                                state_id=f"{phase}-{state_index}",
                                repeat_id=repeat_id,
                                effect=effect,
                            )
                        )
        result = self.estimate(rows)
        # sample variance of (-1, +1) is 2; phases are equally weighted.
        self.assertAlmostEqual(
            result["N"]["mean_same_state_paired_effect_variance"], 2.0 / 3.0
        )

    def test_global_cancellation_retains_directional_phase_profile(self) -> None:
        rows = []
        for trajectory_index in range(8):
            for phase, effect in (("early", 1.0), ("middle", 0.0), ("late", -1.0)):
                for state_index in range(2):
                    for repeat_id in (0, 1):
                        rows.append(
                            EffectRecord(
                                trajectory_id=f"t{trajectory_index}",
                                phase=phase,
                                state_id=f"{phase}-{state_index}",
                                repeat_id=repeat_id,
                                effect=effect,
                            )
                        )
        result = self.estimate(rows)
        self.assertAlmostEqual(result["B"]["estimate"], 0.0)
        conditional = {
            row["phase"]: row for row in result["conditional_B"]["predeclared_phase_rows"]
        }
        self.assertAlmostEqual(conditional["early"]["estimate"], 1.0)
        self.assertAlmostEqual(conditional["late"]["estimate"], -1.0)
        self.assertEqual(
            result["conditional_B"]["status"],
            "DESCRIPTIVE_ONLY_PHASE_NOT_IN_FROZEN_MULTIPLICITY_FAMILY",
        )
        self.assertFalse(result["conditional_B"]["operator_attribution_allowed"])

    def test_rare_deterministic_state_effect_is_H_not_runtime_N(self) -> None:
        rows = []
        for trajectory_index in range(8):
            for phase in ("early", "middle", "late"):
                for state_index in range(20):
                    effect = 1.0 if state_index == 0 else 0.0
                    for repeat_id in (0, 1):
                        rows.append(
                            EffectRecord(
                                trajectory_id=f"t{trajectory_index}",
                                phase=phase,
                                state_id=f"{phase}-{state_index}",
                                repeat_id=repeat_id,
                                effect=effect,
                            )
                        )
        result = self.estimate(rows)
        self.assertAlmostEqual(result["B"]["estimate"], 0.05)
        self.assertGreater(
            result["H"]["mean_within_phase_state_variance_repeat_corrected"], 0.0
        )
        self.assertAlmostEqual(result["N"]["mean_same_state_paired_effect_variance"], 0.0)

    def test_practical_equivalence_is_separate_from_shift_existence(self) -> None:
        result = self.estimate(
            balanced_records([0.001] * 8), practical_tolerance=0.01
        )
        self.assertEqual(result["verdicts"]["shift_existence"], "REPRODUCIBLE_AVERAGE_SHIFT")
        self.assertEqual(
            result["verdicts"]["materiality"],
            "PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT",
        )

    def test_five_percent_tail_coverage_requires_fifty_nine_trajectories(self) -> None:
        insufficient = self.estimate(
            balanced_records([0.0] * 58), tail_prevalence=0.05, tail_alpha=0.05
        )
        sufficient = self.estimate(
            balanced_records([0.0] * 59), tail_prevalence=0.05, tail_alpha=0.05
        )
        self.assertEqual(
            insufficient["verdicts"]["tail_coverage"], "TAIL_COVERAGE_INSUFFICIENT"
        )
        self.assertEqual(sufficient["verdicts"]["tail_coverage"], "TAIL_COVERAGE_SUFFICIENT")
        self.assertEqual(
            sufficient["tail_coverage"]["required_independent_trajectories_for_at_least_one_observation"],
            59,
        )

    def test_multiplicity_adjusted_interval_is_wider_and_not_mislabeled_95(self) -> None:
        records = balanced_records([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
        ordinary = self.estimate(records, interval_alpha=0.05)
        adjusted = self.estimate(records, interval_alpha=0.05 / 3.0)
        ordinary_width = ordinary["B"]["trajectory_t_interval"][1] - ordinary["B"]["trajectory_t_interval"][0]
        adjusted_width = adjusted["B"]["trajectory_t_interval"][1] - adjusted["B"]["trajectory_t_interval"][0]
        self.assertGreater(adjusted_width, ordinary_width)
        self.assertIsNone(adjusted["B"]["trajectory_t_interval_95"])
        self.assertAlmostEqual(adjusted["B"]["interval_alpha"], 0.05 / 3.0)


if __name__ == "__main__":
    unittest.main()
