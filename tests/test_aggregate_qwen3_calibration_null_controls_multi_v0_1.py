import copy
import unittest


from theory_oracle.aggregate_qwen3_calibration_null_controls_multi_v0_1 import (
    aggregate_controls,
)
from theory_oracle.aggregate_qwen3_calibration_null_controls_v0_1 import (
    ARMS,
    SCALAR_PATHS,
    TASK_ENDPOINT_VALUE_FIELDS,
)


def metric(value: float, *, state_count: int = 24) -> dict:
    return {
        "state_count": state_count,
        "nonzero_state_count": 0 if value == 0.0 else 1,
        "max_absolute_contrast": abs(value),
        "trajectory_weighted_signed_contrast": value,
        "trajectory_weighted_absolute_contrast": abs(value),
    }


def trajectory_summary() -> dict:
    return {
        "controls": {
            "within_implementation": {
                arm: {
                    "scalar_controls": {
                        endpoint: metric(0.0) for endpoint in SCALAR_PATHS
                    },
                    "exact_artifact_event_controls": {
                        field: {
                            "equal_states": 24,
                            "state_count": 24,
                            "all_equal": True,
                        }
                        for field in (
                            "parameter_update_artifact_sha_equal",
                            "next_state_digests_equal",
                            "semantic_events_equal",
                        )
                    },
                }
                for arm in ARMS
            },
            "within_evaluator": {
                endpoint: {arm: metric(0.0, state_count=48) for arm in ARMS}
                for endpoint in TASK_ENDPOINT_VALUE_FIELDS
            },
        }
    }


class MultiTrajectoryNullControlTests(unittest.TestCase):
    def test_all_zero_controls_remain_measurement_description(self) -> None:
        result = aggregate_controls([trajectory_summary() for _ in range(4)])
        row = result["within_implementation"]["candidate"]["scalar_controls"][
            "training_loss"
        ]
        self.assertTrue(row["all_observed_contrasts_zero"])
        self.assertEqual(row["observed_state_count"], 96)
        self.assertEqual(row["trajectory_signed_contrast_sd"], 0.0)

    def test_one_nonzero_trajectory_is_not_averaged_away(self) -> None:
        rows = [trajectory_summary() for _ in range(4)]
        changed = copy.deepcopy(rows[-1])
        changed["controls"]["within_implementation"]["candidate"][
            "scalar_controls"
        ]["training_loss"] = metric(-0.25)
        rows[-1] = changed
        result = aggregate_controls(rows)
        row = result["within_implementation"]["candidate"]["scalar_controls"][
            "training_loss"
        ]
        self.assertFalse(row["all_observed_contrasts_zero"])
        self.assertEqual(row["maximum_observed_state_absolute_contrast"], 0.25)
        self.assertEqual(row["nonzero_state_count"], 1)


if __name__ == "__main__":
    unittest.main()
