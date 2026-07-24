from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)
from theory_oracle.evaluate_qwen3_bias_oracle_confirmation_v0_1 import (
    EXPECTED_SENSITIVITY,
)
from theory_oracle.evaluate_qwen3_boundary_independent_contributor_v0_1 import (
    PLAN_SCHEMA_VERSION,
    evaluate,
)
from theory_oracle.freeze_qwen3_boundary_independent_contributor_bank_v0_1 import (
    SCHEMA_VERSION as INDEPENDENT_BANK_SCHEMA_VERSION,
    sha256_file,
)
from theory_oracle.validate_qwen3_boundary_independent_contributor_records_v0_1 import (
    SCHEMA_VERSION as RECORD_SCHEMA_VERSION,
)


def make_fixture(
    root: Path,
    *,
    baseline_cancels: bool = False,
    repair_overshoots: bool = False,
    semantic_disagreement: bool = False,
) -> tuple[dict, Path, dict, dict]:
    trajectories = [
        {
            "trajectory_id": f"contributor-{index}",
            "trajectory_seed": 1000 + index,
            "data_slice_id": f"slice-{index}",
        }
        for index in range(8)
    ]
    groups = {}
    masks = {}
    rows = []
    for trajectory_index, trajectory in enumerate(trajectories):
        baseline = -1.0 if baseline_cancels and trajectory_index % 2 else 1.0
        for phase in ("early", "middle", "late"):
            key = f"{trajectory['trajectory_id']}::{phase}"
            groups[key] = []
            for state in range(2):
                state_id = f"{trajectory['trajectory_id']}-{phase}-{state}"
                groups[key].append(state_id)
                masks[state_id] = hashlib.sha256(f"mask::{state_id}".encode()).hexdigest()
                for repeat in (1, 2):
                    reference = 0.0
                    candidate = baseline
                    intervention = -candidate if repair_overshoots else candidate - 0.75
                    attribution_effect = candidate - intervention
                    rows.append(
                        {
                            "trajectory_id": trajectory["trajectory_id"],
                            "phase": phase,
                            "state_id": state_id,
                            "repeat_id": repeat,
                            "arm_condition_mask_sha256": {
                                arm: masks[state_id]
                                for arm in ("reference", "candidate", "intervention")
                            },
                            "arm_condition_cardinality": {
                                arm: 3
                                for arm in ("reference", "candidate", "intervention")
                            },
                            "reference_value": reference,
                            "candidate_value": candidate,
                            "intervention_value": intervention,
                            "attribution_effect": attribution_effect,
                            **(
                                {
                                    "value_semantics": (
                                        "REFERENCE_ZERO_CANDIDATE_AND_INTERVENTION_"
                                        "ARE_REFERENCE_PAIRED_DISAGREEMENT_RATES"
                                    )
                                }
                                if semantic_disagreement
                                else {}
                            ),
                        }
                    )
    target_effect_role = (
        "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B"
        if semantic_disagreement
        else "SIGNED_BOUNDARY_CONDITIONAL_EFFECT"
    )
    independent = {
        "schema_version": INDEPENDENT_BANK_SCHEMA_VERSION,
        "valid": True,
        "status": "FROZEN_BEFORE_BOUNDARY_OPERATOR_INTERVENTIONS",
        "state_bank_role": "INDEPENDENT_CONTRIBUTOR_CONFIRMATION_BANK",
        "eligible_for_independent_contributor_measurement": True,
        "operator_intervention_outcomes_used_for_bank_freeze": False,
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "target_endpoint": (
            "boundary_semantic_disagreement::tau=0.01"
            if semantic_disagreement
            else "boundary_margin_shift::tau=0.01"
        ),
        "target_effect_role": target_effect_role,
        "frozen_effect_direction": 1,
        "trajectory_inputs": trajectories,
        "exposed_state_ids_by_trajectory_phase": groups,
        "reference_anchor_condition_mask_sha256_by_state": masks,
    }
    independent_path = root / "independent-bank.json"
    independent_path.write_text(json.dumps(independent), encoding="utf-8")
    intervention_plan = {
        "status": "FROZEN_BEFORE_OPERATOR_OUTCOMES",
        "candidate_id": "kernel-7",
        "intervention_kind": "REPAIR_REMOVAL",
        "family_id": "repair-family-v0.1",
        "family_member_id": "kernel-7::repair",
        "candidate_selection_uses_independent_bank_outcomes": False,
    }
    records = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "status": "COMPLETE_VALID_AFTER_OPERATOR_INTERVENTIONS",
        "independent_contributor_bank": {
            "path": str(independent_path),
            "sha256": sha256_file(independent_path),
        },
        "target_endpoint": independent["target_endpoint"],
        "target_effect_role": target_effect_role,
        "candidate_id": "kernel-7",
        "intervention_kind": "REPAIR_REMOVAL",
        "condition_membership_recomputed_after_intervention": False,
        "operator_intervention_plan": intervention_plan,
        "rows": rows,
    }
    source = {
        "kind": "EXACT_ZERO_NULL",
        "description": "unit-test exact directional null",
        "selection_rule": "zero fixed before operator outcomes",
        "uses_operator_outcomes": False,
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_OPERATOR_OUTCOMES",
        "planning_mode": "FIXED_RESOURCE_EXISTENCE",
        "independent_contributor_bank": {
            "path": str(independent_path),
            "sha256": sha256_file(independent_path),
        },
        "target_endpoint": independent["target_endpoint"],
        "target_effect_role": target_effect_role,
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "frozen_effect_direction": 1,
        "candidate_id": "kernel-7",
        "intervention_kind": "REPAIR_REMOVAL",
        "family_id": intervention_plan["family_id"],
        "family_member_id": intervention_plan["family_member_id"],
        "family_members": [intervention_plan["family_member_id"]],
        "family_alpha": 0.05,
        "per_member_interval_alpha": 0.05,
        "planned_trajectories": 8,
        "baseline_transport_floor": 0.0,
        "baseline_transport_floor_source": source,
        "directional_contribution_floor": 0.0,
        "directional_contribution_floor_source": source,
        "operator_outcomes_used_for_planning": False,
        "sensitivity": EXPECTED_SENSITIVITY,
    }
    return independent, independent_path, records, plan


class IndependentBoundaryContributorEvaluationTests(unittest.TestCase):
    def test_independent_transport_and_contribution_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            independent, path, records, plan = make_fixture(Path(temporary))
            result = evaluate(plan, independent, path, records)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(
            result["final_verdict"],
            "REPRODUCIBLE_DIRECTIONAL_INTERVENTION_CONTRIBUTION",
        )
        self.assertTrue(result["population_operator_contribution_claim_allowed"])
        self.assertEqual(result["construction"]["top_level_df"], 7)

    def test_stable_intervention_response_is_not_contribution_if_baseline_does_not_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            independent, path, records, plan = make_fixture(
                Path(temporary), baseline_cancels=True
            )
            result = evaluate(plan, independent, path, records)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(
            result["final_verdict"], "INDETERMINATE_BASELINE_DID_NOT_TRANSPORT"
        )
        self.assertFalse(result["population_operator_contribution_claim_allowed"])

    def test_post_outcome_inference_plan_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            independent, path, records, plan = make_fixture(Path(temporary))
            plan["operator_outcomes_used_for_planning"] = True
            result = evaluate(plan, independent, path, records)
        self.assertFalse(result["valid"])
        self.assertFalse(result["population_operator_contribution_claim_allowed"])
        self.assertTrue(any("used operator outcomes" in error for error in result["errors"]))

    def test_semantic_disagreement_contribution_keeps_non_bias_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            independent, path, records, plan = make_fixture(
                Path(temporary), semantic_disagreement=True
            )
            result = evaluate(plan, independent, path, records)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(
            result["final_verdict"],
            "REPRODUCIBLE_SEMANTIC_IMPACT_INTERVENTION_CONTRIBUTION",
        )
        self.assertNotIn(
            "baseline_candidate_minus_reference_aligned", result["profiles"]
        )
        baseline = result["profiles"][
            "baseline_reference_candidate_disagreement"
        ]
        self.assertTrue(baseline["semantic_impact_is_not_B"])
        self.assertNotIn("B", baseline)

    def test_directional_contribution_does_not_imply_absolute_reduction_or_explained_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            independent, path, records, plan = make_fixture(
                Path(temporary), repair_overshoots=True
            )
            result = evaluate(plan, independent, path, records)
        self.assertTrue(result["population_operator_contribution_claim_allowed"])
        diagnostic = result["profiles"]["repair_diagnostics"]
        self.assertEqual(
            diagnostic["absolute_discrepancy_reduction"]["B"]["estimate"], 0.0
        )
        self.assertFalse(diagnostic["absolute_reduction_claim_allowed"])
        self.assertEqual(
            diagnostic["explained_fraction"]["status"],
            "UNINSTANTIATED_RATIO_REQUIRES_SEPARATE_ESTIMAND_AND_INFERENCE",
        )


if __name__ == "__main__":
    unittest.main()
