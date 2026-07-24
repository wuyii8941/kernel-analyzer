from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (
    clip_decision_metric,
    collect_endpoint,
    endpoint_extractors,
    linked_arms,
    load_complete_state_bundles,
    measured_endpoint,
    reference_aligned_dot,
)
from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    EXPECTED_ENDPOINT_ROLE_CATALOG,
    SIGNED_B_ENDPOINT_CLASSES,
    U2_ENDPOINT,
)
from theory_oracle.bias_oracle_record_v0_2 import (
    SCHEMA_VERSION,
    attach_record_digest,
    endpoint,
    validate_record_bundle,
)


def _identity(repeat: int, arm: str | None = None) -> dict:
    value = {
        "query_id": "Q-R",
        "trajectory_id": "calibration-test",
        "trajectory_anchor": "EAGER_TRAJECTORY",
        "trajectory_seed": 7,
        "data_slice_id": "slice",
        "phase": "early",
        "eligible_step_population": "1:100",
        "state_selection_prng_seed": "rank-seed",
        "state_id": "calibration-test-early-step010",
        "optimizer_step": 10,
        "repeat_id": repeat,
    }
    if arm is not None:
        value["arm"] = arm
    return value


def _arm(repeat: int, arm: str, task_value: float) -> dict:
    return attach_record_digest(
        {
            "identity": _identity(repeat, arm),
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
                "compiler_config_digest": "compiler",
                "graph_family_digest": "graph",
            },
            "outcomes": {
                "parameter_update_artifact": {"path": "not-checked", "sha256": "x"},
                "T1a_arm_loss": endpoint("MEASURED", task_value),
                "T1b_arm_nll": endpoint("MEASURED", task_value),
                "propagation_ledgers": {"training_loss": task_value},
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


def _bundle(second_t1a_status: str = "MEASURED") -> dict:
    arms = []
    pairs = []
    for repeat in (1, 2):
        reference = _arm(repeat, "reference", 1.0)
        candidate = _arm(repeat, "candidate", 1.1)
        arms.extend((reference, candidate))
        t1a = (
            endpoint("MEASURED", 0.1)
            if repeat == 1 or second_t1a_status == "MEASURED"
            else endpoint(second_t1a_status, reason="test")
        )
        pairs.append(
            attach_record_digest(
                {
                    "identity": _identity(repeat),
                    "links": {
                        "reference_arm_record_digest": reference["record_digest"],
                        "candidate_arm_record_digest": candidate["record_digest"],
                        "coupling_protocol_digest": "coupling",
                    },
                    "effects": {
                        "U1": endpoint("MEASURED", 0.01),
                        "U2_delta": endpoint(
                            "MEASURED",
                            0.02,
                            artifact={"path": "not-checked", "sha256": "x"},
                        ),
                        "T1a_shift": t1a,
                        "T1b_shift": endpoint("MEASURED", 0.1),
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
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {"population_eligible": True},
        "arm_records": arms,
        "paired_effect_records": pairs,
    }


def _plan() -> dict:
    return {
        "identity": {
            "query_id": "Q-R",
            "trajectory_id": "calibration-test",
            "trajectory_anchor": "EAGER_TRAJECTORY",
            "trajectory_seed": 7,
            "data_slice_id": "slice",
            "state_selection_prng_seed": "rank-seed",
        },
        "targets": [
            {
                "state_id": "calibration-test-early-step010",
                "optimizer_step": 10,
                "phase": "early",
                "eligible_step_population": "1:100",
            }
        ],
    }


def _materialize(root: Path, bundle: dict) -> None:
    state_root = root / "step010"
    state_root.mkdir(parents=True)
    bundle_path = state_root / "record_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    report = validate_record_bundle(bundle, verify_artifacts=False)
    report["bundle"] = {
        "path": str(bundle_path),
        "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    }
    (state_root / "record_validation.json").write_text(json.dumps(report), encoding="utf-8")


class AggregateCalibrationRecordsTests(unittest.TestCase):
    def test_same_state_repeat_realization_drift_is_not_runtime_N(self) -> None:
        bundle = _bundle()
        candidate = next(
            row
            for row in bundle["arm_records"]
            if row["identity"]["arm"] == "candidate"
            and row["identity"]["repeat_id"] == 2
        )
        old_digest = candidate["record_digest"]
        candidate["realization"]["graph_family_digest"] = "different-variant"
        replacement = attach_record_digest(candidate)
        bundle["arm_records"][bundle["arm_records"].index(candidate)] = replacement
        pair = next(
            row
            for row in bundle["paired_effect_records"]
            if row["identity"]["repeat_id"] == 2
        )
        self.assertEqual(pair["links"]["candidate_arm_record_digest"], old_digest)
        pair["links"]["candidate_arm_record_digest"] = replacement["record_digest"]
        replacement_pair = attach_record_digest(pair)
        bundle["paired_effect_records"][
            bundle["paired_effect_records"].index(pair)
        ] = replacement_pair
        report = validate_record_bundle(bundle, verify_artifacts=False)
        self.assertFalse(report["valid"])
        self.assertTrue(
            any("repeat" in error and "realization" in error for error in report["errors"])
        )

    def test_endpoint_role_catalog_matches_actual_extractor_semantics(self) -> None:
        extractors = endpoint_extractors()
        signed_names = {
            name
            for role in (
                "core_signed_bias_candidates",
                "optional_signed_numerical_bias_candidates",
                "optional_signed_semantic_bias_candidates",
            )
            for name in EXPECTED_ENDPOINT_ROLE_CATALOG[role]
            if name != U2_ENDPOINT
        }
        descriptive_names = set(
            EXPECTED_ENDPOINT_ROLE_CATALOG[
                "descriptive_nonnegative_profiles_not_bias"
            ]
        )
        self.assertFalse(signed_names - set(extractors))
        self.assertFalse(descriptive_names - set(extractors))
        self.assertTrue(
            all(extractors[name][0] in SIGNED_B_ENDPOINT_CLASSES for name in signed_names)
        )
        self.assertTrue(
            all(
                extractors[name][0] not in SIGNED_B_ENDPOINT_CLASSES
                for name in descriptive_names
            )
        )

    def test_clip_directional_shift_does_not_hide_balanced_disagreement(self) -> None:
        pair = {
            "effects": {
                "paired_semantic_events": {
                    "reference": {"clip_decisions": [[False, True, True, False]]},
                    "candidate": {"clip_decisions": [[True, False, True, False]]},
                }
            }
        }
        self.assertEqual(
            clip_decision_metric(pair, "directional_rate_shift"), 0.0
        )
        self.assertEqual(clip_decision_metric(pair, "disagreement_rate"), 0.5)
        self.assertEqual(clip_decision_metric(pair, "off_to_on_rate"), 0.25)
        self.assertEqual(clip_decision_metric(pair, "on_to_off_rate"), 0.25)
        self.assertEqual(clip_decision_metric(pair, "exposure_count"), 4.0)

    def test_aligned_dot_avoids_the_relative_update_denominator(self) -> None:
        pair = {"effects": {"U1": {"status": "MEASURED", "value": 0.25}}}
        reference = {
            "outcomes": {"propagation_ledgers": {"parameter_update_l2": 2.0}}
        }
        self.assertEqual(reference_aligned_dot(pair, reference, {}), 1.0)
        zero_reference = {
            "outcomes": {"propagation_ledgers": {"parameter_update_l2": 0.0}}
        }
        undefined = {"effects": {"U1": {"status": "UNDEFINED"}}}
        self.assertEqual(reference_aligned_dot(undefined, zero_reference, {}), 0.0)

    def test_complete_loader_requires_exact_frozen_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _bundle()
            _materialize(root, bundle)
            rows, evidence, errors = load_complete_state_bundles(_plan(), root)
            self.assertFalse(errors)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(evidence), 1)

            changed_plan = copy.deepcopy(_plan())
            changed_plan["identity"]["trajectory_seed"] = 8
            rows, _, errors = load_complete_state_bundles(changed_plan, root)
            self.assertFalse(rows)
            self.assertEqual(
                errors,
                ["invalid record evidence for calibration-test-early-step010"],
            )

    def test_unavailable_endpoint_is_counted_not_dropped(self) -> None:
        bundle = _bundle(second_t1a_status="UNINSTANTIATED")
        records, availability, unavailable = collect_endpoint(
            [(_plan()["targets"][0], bundle)],
            lambda pair, ref, cand: measured_endpoint(pair["effects"]["T1a_shift"]),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(availability, {"MEASURED": 1, "UNAVAILABLE": 1})
        self.assertEqual(
            unavailable[0]["state_id"], "calibration-test-early-step010"
        )
        self.assertEqual(unavailable[0]["repeat_id"], 2)
        reference, candidate = linked_arms(bundle, bundle["paired_effect_records"][0])
        self.assertEqual(reference["identity"]["arm"], "reference")
        self.assertEqual(candidate["identity"]["arm"], "candidate")


if __name__ == "__main__":
    unittest.main()
