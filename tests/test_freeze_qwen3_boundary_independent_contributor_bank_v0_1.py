from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)
from theory_oracle.freeze_qwen3_boundary_attribution_bridge_v0_1 import (
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
)
from theory_oracle.freeze_qwen3_boundary_independent_contributor_bank_v0_1 import (
    INPUT_SCHEMA_VERSION,
    freeze_independent_bank,
    sha256_file,
)
from theory_oracle.validate_qwen3_boundary_independent_contributor_records_v0_1 import (
    SCHEMA_VERSION as RECORD_SCHEMA_VERSION,
    validate_and_collect_independent,
)


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root: Path) -> tuple[dict, Path, dict, Path]:
    manifest = {
        "trajectory_inputs": [
            {
                "trajectory_id": "endpoint-confirmation-0",
                "trajectory_seed": 100,
                "data_slice_id": "endpoint-slice-0",
            }
        ],
        "calibration_exclusion": {
            "trajectory_ids": ["calibration-0"],
            "trajectory_seeds": [10],
            "data_slice_ids": ["calibration-slice-0"],
        },
    }
    manifest_path = root / "confirmation-manifest.json"
    write(manifest_path, manifest)
    confirmation = {
        "confirmation_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    }
    confirmation_path = root / "boundary-confirmation.json"
    write(confirmation_path, confirmation)
    bridge = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "valid": True,
        "status": "FROZEN_BEFORE_BOUNDARY_ATTRIBUTION_PILOT",
        "state_bank_role": "ENDPOINT_CONFIRMATION_BANK_REUSED_FOR_MECHANISM_PILOT",
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "population_operator_contribution_claim_allowed": False,
        "endpoint_confirmation": {
            "path": str(confirmation_path),
            "sha256": sha256_file(confirmation_path),
        },
        "target_endpoint": "boundary_margin_shift::tau=0.01",
        "tau": 0.01,
        "frozen_effect_direction": 1,
        "target_effect_role": "SIGNED_BOUNDARY_CONDITIONAL_EFFECT",
        "attribution_objective": "INTERVENTION_CHANGES_EFFECT_IN_FROZEN_SIGNED_DIRECTION",
    }
    bridge_path = root / "bridge.json"
    write(bridge_path, bridge)
    trajectories = [
        {
            "trajectory_id": f"contributor-{index}",
            "trajectory_seed": 1000 + index,
            "data_slice_id": f"contributor-slice-{index}",
        }
        for index in range(8)
    ]
    rows = []
    for trajectory in trajectories:
        for phase in ("early", "middle", "late"):
            for state in range(8):
                state_id = f"{trajectory['trajectory_id']}-{phase}-{state}"
                rows.append(
                    {
                        "trajectory_id": trajectory["trajectory_id"],
                        "phase": phase,
                        "state_id": state_id,
                        "reference_exposures": 2 if state < 2 else 0,
                        "reference_condition_mask_sha256": hashlib.sha256(
                            f"mask::{state_id}".encode()
                        ).hexdigest(),
                        "reference_scorer_logps_exact_across_repeats": True,
                        "mask_frozen_before_operator_interventions": True,
                    }
                )
    bank_input = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "REFERENCE_MASKS_FROZEN_BEFORE_OPERATOR_INTERVENTIONS",
        "operator_intervention_outcomes_exist": False,
        "endpoint_definition_bridge": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
        },
        "target_endpoint": bridge["target_endpoint"],
        "target_effect_role": bridge["target_effect_role"],
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "trajectory_inputs": trajectories,
        "state_rows": rows,
    }
    input_path = root / "bank-input.json"
    write(input_path, bank_input)
    return bridge, bridge_path, bank_input, input_path


class IndependentBoundaryContributorBankTests(unittest.TestCase):
    def test_disjoint_reference_masks_can_freeze_before_interventions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bridge_path, bank_input, input_path = fixture(Path(temporary))
            result = freeze_independent_bank(
                bank_input, input_path, bridge, bridge_path
            )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["state_bank_role"], "INDEPENDENT_CONTRIBUTOR_CONFIRMATION_BANK")
        self.assertEqual(result["full_state_count"], 8 * 24)
        self.assertFalse(result["population_operator_contribution_claim_allowed"])
        self.assertTrue(result["eligible_for_independent_contributor_measurement"])

    def test_reusing_endpoint_confirmation_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bridge_path, bank_input, input_path = fixture(Path(temporary))
            bank_input["trajectory_inputs"][0]["trajectory_id"] = "endpoint-confirmation-0"
            input_path.write_text(json.dumps(bank_input), encoding="utf-8")
            result = freeze_independent_bank(bank_input, input_path, bridge, bridge_path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("forbidden trajectory_id" in error for error in result["errors"]))

    def test_post_outcome_or_insufficient_phase_support_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bridge_path, bank_input, input_path = fixture(Path(temporary))
            broken = copy.deepcopy(bank_input)
            broken["operator_intervention_outcomes_exist"] = True
            for row in broken["state_rows"]:
                if row["trajectory_id"] == "contributor-0" and row["phase"] == "late":
                    row["reference_exposures"] = 0
            input_path.write_text(json.dumps(broken), encoding="utf-8")
            result = freeze_independent_bank(broken, input_path, bridge, bridge_path)
        self.assertFalse(result["valid"])
        self.assertTrue(any("outcomes already exist" in error for error in result["errors"]))
        self.assertTrue(any("contributor-0/late" in error for error in result["errors"]))

    def test_independent_repair_records_are_estimator_ready_not_a_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge, bridge_path, bank_input, input_path = fixture(root)
            independent = freeze_independent_bank(
                bank_input, input_path, bridge, bridge_path
            )
            independent_path = root / "independent-bank.json"
            independent_path.write_text(json.dumps(independent), encoding="utf-8")
            rows = []
            masks = independent[
                "reference_anchor_condition_mask_sha256_by_state"
            ]
            for group, states in independent[
                "exposed_state_ids_by_trajectory_phase"
            ].items():
                trajectory, phase = group.split("::")
                for state_id in states:
                    for repeat_id in (1, 2):
                        rows.append(
                            {
                                "trajectory_id": trajectory,
                                "phase": phase,
                                "state_id": state_id,
                                "repeat_id": repeat_id,
                                "arm_condition_mask_sha256": {
                                    arm: masks[state_id]
                                    for arm in ("reference", "candidate", "intervention")
                                },
                                "arm_condition_cardinality": {
                                    arm: 2
                                    for arm in ("reference", "candidate", "intervention")
                                },
                                "reference_value": 0.0,
                                "candidate_value": 1.0,
                                "intervention_value": 0.25,
                                "attribution_effect": 0.75,
                            }
                        )
            records_bank = {
                "schema_version": RECORD_SCHEMA_VERSION,
                "status": "COMPLETE_VALID_AFTER_OPERATOR_INTERVENTIONS",
                "independent_contributor_bank": {
                    "path": str(independent_path),
                    "sha256": sha256_file(independent_path),
                },
                "target_endpoint": independent["target_endpoint"],
                "target_effect_role": independent["target_effect_role"],
                "candidate_id": "kernel-7",
                "intervention_kind": "REPAIR_REMOVAL",
                "condition_membership_recomputed_after_intervention": False,
                "operator_intervention_plan": {
                    "status": "FROZEN_BEFORE_OPERATOR_OUTCOMES",
                    "candidate_id": "kernel-7",
                    "intervention_kind": "REPAIR_REMOVAL",
                    "family_id": "repair-family-v0.1",
                    "family_member_id": "kernel-7::repair",
                    "candidate_selection_uses_independent_bank_outcomes": False,
                },
                "rows": rows,
            }
            records, construction, errors = validate_and_collect_independent(
                records_bank, independent, independent_path
            )
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 8 * 3 * 2 * 2)
        self.assertTrue(construction["eligible_for_trajectory_level_evaluation"])
        self.assertFalse(construction["population_operator_contribution_claim_allowed"])

    def test_post_outcome_candidate_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge, bridge_path, bank_input, input_path = fixture(root)
            independent = freeze_independent_bank(bank_input, input_path, bridge, bridge_path)
            independent_path = root / "independent-bank.json"
            independent_path.write_text(json.dumps(independent), encoding="utf-8")
            records_bank = {
                "schema_version": RECORD_SCHEMA_VERSION,
                "status": "COMPLETE_VALID_AFTER_OPERATOR_INTERVENTIONS",
                "independent_contributor_bank": {
                    "path": str(independent_path),
                    "sha256": sha256_file(independent_path),
                },
                "target_endpoint": independent["target_endpoint"],
                "target_effect_role": independent["target_effect_role"],
                "candidate_id": "kernel-picked-after-results",
                "intervention_kind": "REPAIR_REMOVAL",
                "condition_membership_recomputed_after_intervention": False,
                "operator_intervention_plan": {
                    "status": "FROZEN_BEFORE_OPERATOR_OUTCOMES",
                    "candidate_id": "kernel-picked-after-results",
                    "intervention_kind": "REPAIR_REMOVAL",
                    "family_id": "repair-family-v0.1",
                    "family_member_id": "bad",
                    "candidate_selection_uses_independent_bank_outcomes": True,
                },
                "rows": [],
            }
            records, _, errors = validate_and_collect_independent(
                records_bank, independent, independent_path
            )
        self.assertEqual(records, [])
        self.assertTrue(any("prospectively frozen" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
