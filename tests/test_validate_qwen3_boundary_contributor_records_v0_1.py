from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.freeze_qwen3_boundary_attribution_bridge_v0_1 import (
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
)
from theory_oracle.validate_qwen3_boundary_contributor_records_v0_1 import (
    SCHEMA_VERSION,
    sha256_file,
    validate_and_collect,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


def fixture(root: Path, intervention_kind: str = "REPAIR_REMOVAL") -> tuple[dict, dict, Path]:
    state_groups = {
        f"confirmation-{trajectory}::{phase}": [
            f"confirmation-{trajectory}-{phase}-{state}" for state in range(2)
        ]
        for trajectory in range(2)
        for phase in ("early", "middle", "late")
    }
    state_ids = [state for group in state_groups.values() for state in group]
    masks = {
        state: hashlib.sha256(f"mask::{state}".encode()).hexdigest()
        for state in state_ids
    }
    bridge = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "valid": True,
        "status": "FROZEN_BEFORE_BOUNDARY_ATTRIBUTION_PILOT",
        "target_endpoint": "boundary_margin_shift::tau=0.01",
        "target_effect_role": "SIGNED_BOUNDARY_CONDITIONAL_EFFECT",
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "state_bank_role": "ENDPOINT_CONFIRMATION_BANK_REUSED_FOR_MECHANISM_PILOT",
        "population_operator_contribution_claim_allowed": False,
        "exposed_state_ids_by_trajectory_phase": state_groups,
        "reference_anchor_condition_mask_sha256_by_state": masks,
    }
    bridge_path = root / "bridge.json"
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    rows = []
    for group, group_states in state_groups.items():
        trajectory, phase = group.split("::")
        for state in group_states:
            for repeat in (1, 2):
                reference, candidate, intervention = 0.0, 1.0, 0.25
                effect = (
                    candidate - intervention
                    if intervention_kind == "REPAIR_REMOVAL"
                    else intervention - reference
                )
                rows.append(
                    {
                        "trajectory_id": trajectory,
                        "phase": phase,
                        "state_id": state,
                        "repeat_id": repeat,
                        "arm_condition_mask_sha256": {
                            arm: masks[state]
                            for arm in ("reference", "candidate", "intervention")
                        },
                        "arm_condition_cardinality": {
                            arm: 4
                            for arm in ("reference", "candidate", "intervention")
                        },
                        "reference_value": reference,
                        "candidate_value": candidate,
                        "intervention_value": intervention,
                        "attribution_effect": effect,
                    }
                )
    bank = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_VALID_AFTER_INTERVENTION",
        "boundary_attribution_bridge": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
        },
        "target_endpoint": bridge["target_endpoint"],
        "target_effect_role": bridge["target_effect_role"],
        "candidate_id": "kernel-7",
        "intervention_kind": intervention_kind,
        "condition_membership_recomputed_after_intervention": False,
        "rows": rows,
    }
    return bridge, bank, bridge_path


class BoundaryContributorRecordTests(unittest.TestCase):
    def test_valid_repair_bank_produces_balanced_effect_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary))
            records, construction, errors = validate_and_collect(bank, bridge, path)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 2 * 3 * 2 * 2)
        self.assertTrue(all(record.effect == 0.75 for record in records))
        self.assertIn("global-B", construction["claim_scope"])
        self.assertFalse(construction["population_operator_contribution_claim_allowed"])

    def test_valid_injection_uses_injection_minus_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary), "INJECTION_CREATION")
            records, _, errors = validate_and_collect(bank, bridge, path)
        self.assertEqual(errors, [])
        self.assertTrue(all(record.effect == 0.25 for record in records))

    def test_semantic_disagreement_uses_explicit_relational_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary))
            bridge["target_endpoint"] = "boundary_semantic_disagreement::tau=0.01"
            bridge["target_effect_role"] = "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
            path.write_text(json.dumps(bridge), encoding="utf-8")
            bank["boundary_attribution_bridge"]["sha256"] = sha256_file(path)
            bank["target_endpoint"] = bridge["target_endpoint"]
            bank["target_effect_role"] = bridge["target_effect_role"]
            for row in bank["rows"]:
                row["value_semantics"] = (
                    "REFERENCE_ZERO_CANDIDATE_AND_INTERVENTION_ARE_"
                    "REFERENCE_PAIRED_DISAGREEMENT_RATES"
                )
            records, construction, errors = validate_and_collect(
                bank, bridge, path
            )
        self.assertEqual(errors, [])
        self.assertTrue(all(record.effect == 0.75 for record in records))
        self.assertEqual(
            construction["value_semantics"], "RELATIONAL_DISAGREEMENT_RATES"
        )

    def test_semantic_disagreement_cannot_masquerade_as_arm_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary))
            bridge["target_endpoint"] = "boundary_semantic_disagreement::tau=0.01"
            bridge["target_effect_role"] = "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
            path.write_text(json.dumps(bridge), encoding="utf-8")
            bank["boundary_attribution_bridge"]["sha256"] = sha256_file(path)
            bank["target_endpoint"] = bridge["target_endpoint"]
            bank["target_effect_role"] = bridge["target_effect_role"]
            records, _, errors = validate_and_collect(bank, bridge, path)
        self.assertEqual(records, [])
        self.assertTrue(any("relational value semantics" in error for error in errors))

    def test_mask_reselection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary))
            bank["rows"][0]["arm_condition_mask_sha256"]["intervention"] = "0" * 64
            records, _, errors = validate_and_collect(bank, bridge, path)
        self.assertEqual(records, [])
        self.assertTrue(any("arm masks" in error for error in errors))

    def test_out_of_bank_state_and_missing_repeat_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary))
            broken = copy.deepcopy(bank)
            broken["rows"][0]["state_id"] = "post-hoc-selected-state"
            broken["rows"].pop()
            records, _, errors = validate_and_collect(broken, bridge, path)
        self.assertEqual(records, [])
        self.assertTrue(any("outside frozen exposed bank" in error for error in errors))
        self.assertTrue(any("exactly two repeats" in error for error in errors))

    def test_wrong_repair_formula_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary))
            bank["rows"][0]["attribution_effect"] = 1.0
            records, _, errors = validate_and_collect(bank, bridge, path)
        self.assertEqual(records, [])
        self.assertTrue(any("formula mismatch" in error for error in errors))

    def test_bridge_weighting_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge, bank, path = fixture(Path(temporary))
            bridge["weighting_contract_id"] = "EXPOSURE_POOLED"
            path.write_text(json.dumps(bridge), encoding="utf-8")
            bank["boundary_attribution_bridge"]["sha256"] = sha256_file(path)
            records, _, errors = validate_and_collect(bank, bridge, path)
        self.assertEqual(records, [])
        self.assertTrue(any("weighting contract" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
