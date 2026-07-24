from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


THEORY_DIR = Path(__file__).resolve().parents[1] / "theory_oracle"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from bias_oracle_record_v0_2 import (  # noqa: E402
    SCHEMA_VERSION,
    attach_record_digest,
    endpoint,
    validate_record_bundle,
)


def identity(arm: str | None = None) -> dict:
    result = {
        "query_id": "Q-R",
        "trajectory_id": "t0",
        "trajectory_anchor": "EAGER_TRAJECTORY",
        "trajectory_seed": 1,
        "data_slice_id": "slice0",
        "phase": "early",
        "eligible_step_population": "1:100",
        "state_selection_prng_seed": 2,
        "state_id": "s1",
        "optimizer_step": 10,
        "repeat_id": 1,
    }
    if arm is not None:
        result["arm"] = arm
    return result


def arm_record(arm: str, t1a: float, t1b: float) -> dict:
    return attach_record_digest(
        {
            "identity": identity(arm),
            "pre_state": {
                "model_digest": "m",
                "buffer_digest": "b",
                "optimizer_digest": "o",
                "scheduler_digest": "s",
                "scaler_digest": "c",
                "rng_digest": "r",
                "minibatch_digest": "x",
            },
            "realization": {
                "compiler_config_digest": "cc",
                "graph_family_digest": "g" if arm == "candidate" else "NOT_APPLICABLE_EAGER",
            },
            "outcomes": {
                "parameter_update_artifact": {"path": "unused", "sha256": "unused"},
                "T1a_arm_loss": endpoint("MEASURED", t1a),
                "T1b_arm_nll": endpoint("MEASURED", t1b),
                "propagation_ledgers": {},
                "semantic_events": {
                    "clip_count": 1,
                    "clip_decisions": [[False, True]],
                    "gradient_clip_triggered": True,
                    "optimizer_step_skipped": False,
                },
                "next_state_digests": {},
            },
        }
    )


def valid_bundle() -> dict:
    reference = arm_record("reference", 1.0, 2.0)
    candidate = arm_record("candidate", 1.1, 1.8)
    pair = attach_record_digest(
        {
            "identity": identity(),
            "links": {
                "reference_arm_record_digest": reference["record_digest"],
                "candidate_arm_record_digest": candidate["record_digest"],
                "coupling_protocol_digest": "coupling",
            },
            "effects": {
                "U1": endpoint("MEASURED", 0.01),
                "U2_delta": endpoint("UNINSTANTIATED"),
                "T1a_shift": endpoint("MEASURED", 0.1),
                "T1b_shift": endpoint("MEASURED", -0.2),
                "paired_semantic_events": {
                    "reference": copy.deepcopy(reference["outcomes"]["semantic_events"]),
                    "candidate": copy.deepcopy(candidate["outcomes"]["semantic_events"]),
                    "clip_count_difference": 0,
                    "gradient_clip_trigger_difference": 0,
                    "optimizer_skip_difference": 0,
                },
                "paired_next_state_digests": {},
            },
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {"population_eligible": False},
        "arm_records": [reference, candidate],
        "paired_effect_records": [pair],
    }


class RecordBundleTests(unittest.TestCase):
    def test_paired_semantic_events_must_equal_linked_arms(self) -> None:
        bundle = valid_bundle()
        bundle["paired_effect_records"][0]["effects"]["paired_semantic_events"][
            "candidate"
        ]["clip_decisions"] = [[True, True]]
        bundle["paired_effect_records"][0] = attach_record_digest(
            bundle["paired_effect_records"][0]
        )
        report = validate_record_bundle(bundle, verify_artifacts=False)
        self.assertFalse(report["valid"])
        self.assertTrue(
            any("does not equal linked candidate arm" in error for error in report["errors"])
        )

    def test_valid_selected_state_bundle(self) -> None:
        report = validate_record_bundle(valid_bundle(), verify_artifacts=False)
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["population_eligible"])

    def test_paired_field_is_forbidden_in_arm_record(self) -> None:
        bundle = valid_bundle()
        bundle["arm_records"][0]["outcomes"]["U1"] = endpoint("MEASURED", 0.01)
        bundle["arm_records"][0] = attach_record_digest(bundle["arm_records"][0])
        report = validate_record_bundle(bundle, verify_artifacts=False)
        self.assertFalse(report["valid"])
        self.assertTrue(any("paired fields" in error for error in report["errors"]))

    def test_prestate_mismatch_fails(self) -> None:
        bundle = valid_bundle()
        bundle["arm_records"][1]["pre_state"]["rng_digest"] = "different"
        bundle["arm_records"][1] = attach_record_digest(bundle["arm_records"][1])
        bundle["paired_effect_records"][0]["links"]["candidate_arm_record_digest"] = bundle[
            "arm_records"
        ][1]["record_digest"]
        bundle["paired_effect_records"][0] = attach_record_digest(bundle["paired_effect_records"][0])
        report = validate_record_bundle(bundle, verify_artifacts=False)
        self.assertTrue(any("pre-state mismatch" in error for error in report["errors"]))

    def test_wrong_candidate_minus_reference_effect_fails(self) -> None:
        bundle = valid_bundle()
        bundle["paired_effect_records"][0]["effects"]["T1a_shift"] = endpoint(
            "MEASURED", -0.1
        )
        bundle["paired_effect_records"][0] = attach_record_digest(bundle["paired_effect_records"][0])
        report = validate_record_bundle(bundle, verify_artifacts=False)
        self.assertTrue(any("T1a_shift" in error for error in report["errors"]))

    def test_duplicate_arm_key_fails(self) -> None:
        bundle = valid_bundle()
        bundle["arm_records"].append(copy.deepcopy(bundle["arm_records"][0]))
        report = validate_record_bundle(bundle, verify_artifacts=False)
        self.assertTrue(any("duplicate arm keys" in error for error in report["errors"]))

    def test_population_bundle_requires_u2_artifact(self) -> None:
        bundle = valid_bundle()
        bundle["scope"]["population_eligible"] = True
        report = validate_record_bundle(bundle, verify_artifacts=False)
        self.assertTrue(any("U2_delta must be measured" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
