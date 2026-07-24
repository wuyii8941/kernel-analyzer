from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from forkcert.training_step_decision import decide_training_step


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class TrainingStepDecisionRecordTests(unittest.TestCase):
    def assert_record(
        self,
        query_path: str,
        evidence_path: str,
        result_path: str,
        correctness: str,
        completion: str,
    ) -> dict:
        actual = decide_training_step(load(query_path), load(evidence_path))
        self.assertEqual(actual, load(result_path))
        self.assertEqual(actual["correctness_claim"]["verdict"], correctness)
        self.assertEqual(actual["subject_instantiation"]["status"], completion)
        return actual

    def test_qwen_impact_reject_with_correctness_refusal(self) -> None:
        result = self.assert_record(
            "theory_oracle/QWEN3_GRPO_UNIFIED_ORACLE_QUERY_V0_3.json",
            "theory_oracle/QWEN3_GRPO_UNIFIED_ORACLE_EVIDENCE_V0_3.json",
            "results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/unified_oracle_result_v0_3.json",
            "UNINSTANTIATED",
            "INCOMPLETE",
        )
        self.assertEqual(
            result["impacts"]["negative_advantage_grpo_clipping"]["status"], "REJECT"
        )
        self.assertEqual(
            result["impacts"]["negative_advantage_grpo_clipping"]["evidence"]["details"][
                "directional_shift"
            ],
            0.0,
        )
        self.assertFalse(result["sampling_uncertainty"]["population_claim_licensed"])

    def test_qwen_branch_repair_query_is_frozen_and_mechanically_valid(self) -> None:
        query_path = ROOT / "theory_oracle/QWEN3_GRPO_BRANCH_REPAIR_UNIFIED_QUERY_V0_3.json"
        query = json.loads(query_path.read_text(encoding="utf-8"))
        manifest = load(
            "theory_oracle/QWEN3_GRPO_BRANCH_REPAIR_UNIFIED_QUERY_MANIFEST_V0_3.json"
        )
        self.assertEqual(
            hashlib.sha256(query_path.read_bytes()).hexdigest(),
            manifest["query"]["sha256"],
        )
        evidence = {
            "schema_version": "forkcert.training-step-oracle-evidence.v0.3",
            "subject_id": query["subject_id"],
            "validity_gates": [
                {
                    "id": "pre_execution_query_audit",
                    "passed": True,
                    "evidence": "query schema audit only; GPU execution deliberately not substituted",
                }
            ],
            "candidate_identity": {
                "status": "INAPPLICABLE",
                "reason": "pre-execution query validation",
            },
            "ledgers": {},
        }
        result = decide_training_step(query, evidence)
        self.assertEqual(result["validity"]["status"], "INAPPLICABLE")
        stratum = result["declared_context"]["population_contract"]["strata"][0]
        self.assertEqual(stratum["trajectory_anchor"], "REFERENCE_TRAJECTORY")
        self.assertEqual(stratum["enrichment"], "EVENT_CONDITIONED")

    def test_qwen_branch_repair_v03_refuses_mismatched_endpoint(self) -> None:
        result = self.assert_record(
            "theory_oracle/QWEN3_GRPO_BRANCH_REPAIR_UNIFIED_QUERY_V0_3.json",
            "theory_oracle/QWEN3_GRPO_BRANCH_REPAIR_UNIFIED_EVIDENCE_V0_3_INVALID.json",
            "results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/branch_repair_unified_result_v0_3_invalid.json",
            "INVALID",
            "INCOMPLETE",
        )
        self.assertEqual(result["validity"]["status"], "INVALID")
        self.assertIn(
            "failed validity gates: A_reference_endpoint_realization, complete_three_arm_execution",
            result["validity"]["failed_gates"],
        )
        self.assertEqual(
            result["impacts"]["selected_branch_update_effect"]["status"], "INVALID"
        )
        self.assertEqual(result["attribution"]["status"], "INVALID")

    def test_qwen_grad_event_bank_completes_impact_without_correctness_promotion(self) -> None:
        result = self.assert_record(
            "theory_oracle/QWEN3_GRPO_GRAD_EVENT_BANK_UNIFIED_QUERY_V0_4.json",
            "theory_oracle/QWEN3_GRPO_GRAD_EVENT_BANK_UNIFIED_EVIDENCE_V0_4.json",
            "results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/unified_oracle_result_v0_4.json",
            "UNINSTANTIATED",
            "COMPLETE",
        )
        self.assertEqual(result["validity"]["status"], "VALID")
        self.assertEqual(result["discrepancy"]["status"], "ESTIMATED")
        self.assertEqual(
            result["impacts"]["grad_context_negative_advantage_grpo_clipping"]["status"],
            "REJECT",
        )
        self.assertFalse(result["sampling_uncertainty"]["population_claim_licensed"])

    def test_qwen_grad_branch_repair_is_intervention_dependent_not_correctness(self) -> None:
        result = self.assert_record(
            "theory_oracle/QWEN3_GRPO_GRAD_BRANCH_REPAIR_UNIFIED_QUERY_V0_9.json",
            "theory_oracle/QWEN3_GRPO_GRAD_BRANCH_REPAIR_UNIFIED_EVIDENCE_V0_9.json",
            "results/training_step_oracle/qwen3_grpo_grad_branch_repair_v0_9/unified_oracle_result_v0_9.json",
            "UNINSTANTIATED",
            "COMPLETE",
        )
        self.assertEqual(result["validity"]["status"], "VALID")
        self.assertEqual(
            result["impacts"]["selected_branch_controlled_update_effect"]["status"],
            "REJECT",
        )
        self.assertEqual(result["attribution"]["status"], "EFFECT_DETECTED")
        self.assertEqual(
            result["attribution"]["eligible_claim_level"], "INTERVENTION_DEPENDENT"
        )
        self.assertIn("injection_reported", result["attribution"]["causal_gate_failures"])

    def test_analytic_correct_accepts(self) -> None:
        self.assert_record(
            "theory_oracle/ANALYTIC_LINEAR_CORRECT_UNIFIED_QUERY_V0_3.json",
            "theory_oracle/ANALYTIC_LINEAR_CORRECT_UNIFIED_EVIDENCE_V0_3.json",
            "results/training_step_oracle/analytic_linear_correct_v0_1/unified_oracle_result_v0_3.json",
            "ACCEPT",
            "COMPLETE",
        )

    def test_analytic_drop_last_rejects(self) -> None:
        result = self.assert_record(
            "theory_oracle/ANALYTIC_LINEAR_DROP_LAST_UNIFIED_QUERY_V0_3.json",
            "theory_oracle/ANALYTIC_LINEAR_DROP_LAST_UNIFIED_EVIDENCE_V0_3.json",
            "results/training_step_oracle/analytic_linear_drop_last_v0_1/unified_oracle_result_v0_3.json",
            "REJECT",
            "COMPLETE",
        )
        self.assertEqual(result["transition"]["exact_transition"]["status"], "ACCEPT")
        self.assertEqual(result["transition"]["numerical_transition"]["status"], "REJECT")

    def test_sampling_law_separates_rng_from_model_execution(self) -> None:
        result = self.assert_record(
            "theory_oracle/QWEN3_SAMPLING_LAW_UNIFIED_QUERY_V0_3.json",
            "theory_oracle/QWEN3_SAMPLING_LAW_UNIFIED_EVIDENCE_V0_3.json",
            "results/oracle_sampling/confirmation/unified_oracle_result_v0_3.json",
            "UNINSTANTIATED",
            "INCOMPLETE",
        )
        self.assertEqual(result["discrepancy"]["status"], "ESTIMATED")
        self.assertEqual(
            result["impacts"]["categorical_selection_law_compatibility"]["status"],
            "UNINSTANTIATED",
        )
        sources = result["variability_profile"]["evidence"]["sources"]
        self.assertEqual(
            sources["categorical_draw_rng"]["classification"], "WITHIN_STATE_RANDOMNESS"
        )
        self.assertEqual(sources["model_execution"]["classification"], "FIXED_BY_PROTOCOL")
        self.assertEqual(
            sources["implementation_path"]["classification"],
            "FIXED_IMPLEMENTATION_EFFECT",
        )

    def test_greedy_categorical_impact_has_no_fabricated_direction(self) -> None:
        result = self.assert_record(
            "theory_oracle/QWEN3_GREEDY_UNIFIED_QUERY_V0_3.json",
            "theory_oracle/QWEN3_GREEDY_UNIFIED_EVIDENCE_V0_3.json",
            "results/training_step_oracle/qwen3_impact_confirmation_v0_1/unified_oracle_result_v0_3.json",
            "UNINSTANTIATED",
            "INCOMPLETE",
        )
        endpoint = result["impacts"]["greedy_next_token"]
        self.assertEqual(endpoint["status"], "REJECT")
        details = endpoint["evidence"]["details"]
        self.assertNotIn("directional_shift", details)
        self.assertEqual(details["distance_or_cost"]["token_transition_counts"], {"19->422": 1})

    def test_bert_region_record_is_complete_only_for_segmented_intervention(self) -> None:
        result = self.assert_record(
            "theory_oracle/BERT_LAYER0_ATTRIBUTION_UNIFIED_QUERY_V0_3.json",
            "theory_oracle/BERT_LAYER0_ATTRIBUTION_UNIFIED_EVIDENCE_V0_3.json",
            "results/oracle_region/layer0_confirmation/unified_oracle_result_v0_3.json",
            "UNINSTANTIATED",
            "COMPLETE",
        )
        attribution = result["attribution"]
        self.assertEqual(attribution["status"], "EFFECT_DETECTED")
        self.assertEqual(attribution["eligible_claim_level"], "INTERVENTION_DEPENDENT")
        self.assertIn("anchor_parity", attribution["causal_gate_failures"])
        self.assertEqual(
            attribution["contrast_interpretation"]["unique_root_cause"], "NOT_LICENSED"
        )

    def test_bert_operator_coverage_refuses_r1_family_promotion(self) -> None:
        result = self.assert_record(
            "theory_oracle/BERT_OPERATOR_COVERAGE_UNIFIED_QUERY_V0_3.json",
            "theory_oracle/BERT_OPERATOR_COVERAGE_UNIFIED_EVIDENCE_V0_3.json",
            "results/training_step_oracle/bert_operator_coverage_v0_1/unified_oracle_result_v0_3.json",
            "UNINSTANTIATED",
            "INCOMPLETE",
        )
        ledger = result["operator_conformance"]
        self.assertEqual(ledger["status"], "PARTIAL")
        self.assertEqual(ledger["evidence"]["coverage"]["contract_covered"], 0)
        self.assertTrue(
            all(entry["identity_level"] == "R1" for entry in ledger["evidence"]["entries"])
        )


if __name__ == "__main__":
    unittest.main()
