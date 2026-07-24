from __future__ import annotations

import copy
import unittest

from forkcert.training_step_decision import (
    EVIDENCE_SCHEMA,
    QUERY_SCHEMA,
    decide_training_step,
)


def base_query() -> dict:
    return {
        "schema_version": QUERY_SCHEMA,
        "subject_id": "subject-1",
        "state_contract": {
            "declared": True,
            "fields": ["parameters", "optimizer", "batch", "rng"],
            "outcome_relevant_omissions": [],
        },
        "implementation_contract": {
            "declared": True,
            "reference": "reference-realization",
            "candidate": "candidate-realization",
        },
        "observable_contract": {
            "declared": True,
            "fields": ["loss", "gradients", "next_parameters", "counters"],
            "outcome_relevant_omissions": [],
        },
        "population_contract": {
            "declared": True,
            "population_id": "finite-bank-A",
            "claim_kind": "FINITE_BANK",
            "population_kind": "MATCHED_STATE",
            "sampling_unit": "trajectory_cluster",
            "selection_design": "frozen deterministic trajectory-cluster bank",
            "aggregation_rule": "equal cluster weighting; no external prevalence claim",
            "strata": [
                {
                    "id": "reference-bank",
                    "trajectory_anchor": "REFERENCE_TRAJECTORY",
                    "enrichment": "NONE",
                    "provenance": "frozen reference trajectory",
                    "inclusion_rule": "all predeclared clusters in finite-bank-A",
                }
            ],
        },
        "randomness_contract": {
            "declared": True,
            "mode": "DETERMINISTIC",
            "coupling": "identical frozen RNG state",
        },
        "ledgers": {
            "exact_transition": {
                "scope": "REQUIRED",
                "authority": {
                    "kind": "DOCUMENTED_SEMANTICS",
                    "source": "documented exact update relation",
                    "scope": "covered exact fields",
                    "independent_of_candidate_measurements": True,
                    "acceptance_rule_frozen": True,
                },
            },
            "numerical_transition": {
                "scope": "REQUIRED",
                "authority": {
                    "kind": "HIGH_PRECISION_REFERENCE",
                    "source": "independent high-precision envelope",
                    "scope": "covered numerical fields",
                    "independent_of_candidate_measurements": True,
                    "acceptance_rule_frozen": True,
                },
            },
            "stochastic_transition": {"scope": "NOT_IN_SCOPE"},
            "impacts": {
                "clip": {
                    "scope": "REQUIRED",
                    "authority": "predeclared application compatibility rule",
                    "acceptance_rule": "no paired clipping disagreement",
                    "event_type": "BINARY",
                    "boundary_geometry": "NATURAL",
                }
            },
            "discrepancy": {
                "scope": "REQUIRED",
                "estimands": [
                    {
                        "id": "average_implementation_relative_shift",
                        "observable": "loss",
                        "comparison": "candidate minus reference",
                        "aggregation_unit": "trajectory cluster",
                        "geometry": "signed scalar mean",
                    },
                    {
                        "id": "state_conditioned_heterogeneity",
                        "observable": "per-state loss difference",
                        "comparison": "candidate minus reference across states",
                        "aggregation_unit": "trajectory cluster",
                        "geometry": "between-state distribution",
                    },
                    {
                        "id": "within_state_runtime_variability",
                        "observable": "loss repeat difference",
                        "comparison": "repeat-to-repeat within implementation and state",
                        "aggregation_unit": "implementation by frozen state",
                        "geometry": "within-cell dispersion",
                    },
                ],
            },
            "sampling_uncertainty": {
                "scope": "REQUIRED",
                "sampling_unit": "trajectory_cluster",
                "method": "cluster-aware finite-bank interval",
            },
            "variability_profile": {
                "scope": "REQUIRED",
                "sources": [
                    {"id": "state_sampling", "kind": "STATE_SAMPLING"},
                    {"id": "gpu_execution", "kind": "EXECUTION_NONDETERMINISM"},
                    {
                        "id": "implementation_path",
                        "kind": "FIXED_IMPLEMENTATION_DIFFERENCE",
                    },
                ],
            },
            "operator_conformance": {"scope": "NOT_IN_SCOPE"},
            "attribution": {"scope": "NOT_IN_SCOPE"},
        },
        "claim_scope": {
            "covered_scope": "one deterministic finite bank",
            "explicit_non_claims": ["deployment prevalence", "long-run convergence"],
        },
    }


def base_evidence() -> dict:
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "subject_id": "subject-1",
        "validity_gates": [
            {"id": "state_identity", "passed": True, "evidence": "hashes"},
            {"id": "candidate_execution", "passed": True, "evidence": "tracked invocation"},
        ],
        "candidate_identity": {"status": "VALID", "evidence": "graph hash"},
        "ledgers": {
            "exact_transition": {"status": "ACCEPT", "evidence": "exact fields"},
            "numerical_transition": {"status": "ACCEPT", "evidence": "certified bound"},
            "impacts": {
                "clip": {
                    "status": "ACCEPT",
                    "details": {
                        "directional_shift": 0.0,
                        "disagreement": 0.0,
                        "boundary_exposure": {"near_boundary": 3},
                    },
                }
            },
            "discrepancy": {
                "status": "ESTIMATED",
                "average_implementation_relative_shift": {
                    "estimate": 0.0,
                    "scope_or_uncertainty": "finite trajectory-cluster bank",
                },
                "state_conditioned_heterogeneity": {
                    "estimate": 0.1,
                    "scope_or_uncertainty": "finite trajectory-cluster bank",
                },
                "within_state_runtime_variability": {
                    "estimate": 0.0,
                    "scope_or_uncertainty": "two repeats per implementation and state",
                },
            },
            "sampling_uncertainty": {
                "status": "ESTIMATED",
                "sampling_unit": "trajectory_cluster",
                "intervals": {},
            },
            "variability_profile": {
                "status": "CHARACTERIZED",
                "sources": {
                    "state_sampling": {
                        "classification": "BETWEEN_STATE_DESIGN",
                        "observation": "finite trajectory-cluster bank",
                    },
                    "gpu_execution": {
                        "classification": "FIXED_BY_PROTOCOL",
                        "observation": "exact repeated signatures under deterministic protocol",
                    },
                    "implementation_path": {
                        "classification": "FIXED_IMPLEMENTATION_EFFECT",
                        "observation": "fixed eager versus compiled realizations",
                    },
                },
            },
        },
    }


def attribution_contract() -> dict:
    return {
        "scope": "REQUIRED",
        "intervention": "repair and injection",
        "requested_claim_level": "OPERATOR_CAUSAL",
        "endpoints": [{"id": "next_update", "geometry": "L2"}],
        "required_contrasts": ["TOTAL", "REPAIR", "INJECTION", "INTERACTION"],
    }


def attribution_evidence(*, full_integrity: bool, operator_identity: bool = True) -> dict:
    return {
        "status": "EFFECT_DETECTED",
        "treatment": {
            "claimed_subject": "SEMANTIC_OPERATOR" if operator_identity else "REGION",
            "correspondence_level": "R4" if operator_identity else "R3",
            "intervention_level": "I3" if operator_identity else "I1",
        },
        "effects": {
            "next_update": {
                "contrasts": {
                    "TOTAL": {
                        "definition": "T_B - T_A",
                        "estimate": {"l2": 1.0},
                        "scope_or_uncertainty": "held-out state bank",
                    },
                    "REPAIR": {
                        "definition": "T_R - T_B",
                        "estimate": {"l2": 0.4},
                        "scope_or_uncertainty": "held-out state bank",
                    },
                    "INJECTION": {
                        "definition": "T_I - T_A",
                        "estimate": {"l2": 0.3},
                        "scope_or_uncertainty": "held-out state bank",
                    },
                    "INTERACTION": {
                        "definition": "T_B - T_I - T_R + T_A",
                        "estimate": {"l2": 0.5},
                        "scope_or_uncertainty": "held-out state bank",
                    },
                }
            }
        },
        "integrity": {
            "treatment_identity": True,
            "anchor_parity": True,
            "non_target_preserved": full_integrity,
            "repair_reported": True,
            "injection_reported": full_integrity,
            "interactions_reported": full_integrity,
            "heldout_replication": full_integrity,
        },
        "roles": [{"role": "SOURCE", "evidence": "identified boundary discrepancy"}],
    }


def operator_contract() -> dict:
    return {
        "scope": "REQUIRED",
        "claim_unit": "SEMANTIC_OPERATOR",
        "coverage_unit": "INSTANCE",
        "coverage_rule": "record every covered operator instance encountered in the step",
    }


def operator_evidence(*, verdict: str = "ACCEPT") -> dict:
    return {
        "status": "RECORDED",
        "coverage": {
            "encountered": 1,
            "contract_covered": 1,
            "unidentified_or_uncontracted": 0,
        },
        "entries": [
            {
                "unit_id": "sum-1",
                "subject_kind": "SEMANTIC_OPERATOR",
                "semantic_operator": "sum",
                "identity_level": "R4",
                "contract_authority": {
                    "kind": "FORMAL_RELATION",
                    "source": "independent reduction contract",
                    "scope": "covered sum operands and output",
                    "independent_of_candidate_measurements": True,
                    "acceptance_rule_frozen": True,
                },
                "verdict": verdict,
                "evidence": "covered operands and result witness",
            }
        ],
    }


class TrainingStepDecisionTests(unittest.TestCase):
    def test_complete_accept(self) -> None:
        result = decide_training_step(base_query(), base_evidence())
        self.assertEqual(result["validity"]["status"], "VALID")
        self.assertEqual(result["correctness_claim"]["verdict"], "ACCEPT")
        self.assertEqual(result["subject_instantiation"]["status"], "COMPLETE")
        self.assertEqual(
            result["declared_context"]["population_contract"]["strata"][0][
                "trajectory_anchor"
            ],
            "REFERENCE_TRAJECTORY",
        )

    def test_exact_reject_survives_numerical_uninstantiated(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["numerical_transition"].pop("authority")
        evidence["ledgers"]["exact_transition"]["status"] = "REJECT"
        evidence["ledgers"].pop("numerical_transition")
        result = decide_training_step(query, evidence)
        self.assertEqual(result["transition"]["numerical_transition"]["status"], "UNINSTANTIATED")
        self.assertEqual(result["correctness_claim"]["verdict"], "REJECT")
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_impact_reject_does_not_become_correctness_reject(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["impacts"]["clip"]["status"] = "REJECT"
        evidence["ledgers"]["impacts"]["clip"]["details"].update(
            directional_shift=0.0, disagreement=0.02
        )
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["impacts"]["clip"]["status"], "REJECT")
        self.assertEqual(result["correctness_claim"]["verdict"], "ACCEPT")

    def test_local_operator_reject_does_not_become_step_reject(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["operator_conformance"] = operator_contract()
        evidence["ledgers"]["operator_conformance"] = operator_evidence(verdict="REJECT")
        result = decide_training_step(query, evidence)
        self.assertEqual(result["operator_conformance"]["status"], "RECORDED")
        self.assertEqual(result["correctness_claim"]["verdict"], "ACCEPT")

    def test_region_cannot_be_submitted_as_operator_conformance(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["operator_conformance"] = operator_contract()
        operator = operator_evidence()
        operator["entries"][0]["subject_kind"] = "REGION"
        evidence["ledgers"]["operator_conformance"] = operator
        result = decide_training_step(query, evidence)
        self.assertEqual(result["operator_conformance"]["status"], "INVALID")

    def test_terminal_operator_verdict_requires_r4_identity(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["operator_conformance"] = operator_contract()
        operator = operator_evidence()
        operator["entries"][0]["identity_level"] = "R3"
        evidence["ledgers"]["operator_conformance"] = operator
        result = decide_training_step(query, evidence)
        self.assertEqual(result["operator_conformance"]["status"], "INVALID")

    def test_partial_operator_coverage_keeps_required_subject_incomplete(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["operator_conformance"] = operator_contract()
        evidence["ledgers"]["operator_conformance"] = {
            "status": "PARTIAL",
            "coverage": {
                "encountered": 1,
                "contract_covered": 0,
                "unidentified_or_uncontracted": 1,
            },
            "entries": [
                {
                    "unit_id": "fused-1",
                    "subject_kind": "SEMANTIC_OPERATOR",
                    "semantic_operator": "unresolved fused source",
                    "identity_level": "R3",
                    "verdict": "UNINSTANTIATED",
                    "reason": "no R4 correspondence or independent constituent contract",
                }
            ],
        }
        result = decide_training_step(query, evidence)
        self.assertEqual(result["operator_conformance"]["status"], "PARTIAL")
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_indeterminate_local_operator_cannot_be_recorded_as_complete(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["operator_conformance"] = operator_contract()
        operator = operator_evidence(verdict="INDETERMINATE")
        evidence["ledgers"]["operator_conformance"] = operator
        result = decide_training_step(query, evidence)
        self.assertEqual(result["operator_conformance"]["status"], "INVALID")

        operator["status"] = "INDETERMINATE"
        evidence["ledgers"]["operator_conformance"] = operator
        result = decide_training_step(query, evidence)
        self.assertEqual(result["operator_conformance"]["status"], "INDETERMINATE")
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_missing_state_contract_is_invalid(self) -> None:
        query = base_query()
        query["state_contract"]["declared"] = False
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["validity"]["status"], "INVALID")
        self.assertEqual(result["correctness_claim"]["verdict"], "INVALID")

    def test_population_stratum_requires_provenance_and_anchor(self) -> None:
        query = base_query()
        query["population_contract"]["strata"][0].pop("provenance")
        query["population_contract"]["strata"][0]["trajectory_anchor"] = "UNKNOWN"
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["validity"]["status"], "INVALID")

    def test_population_origin_and_enrichment_are_separate(self) -> None:
        query = base_query()
        stratum = query["population_contract"]["strata"][0]
        stratum["trajectory_anchor"] = "REFERENCE_TRAJECTORY"
        stratum["enrichment"] = "EVENT_CONDITIONED"
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["validity"]["status"], "VALID")
        stratum.pop("enrichment")
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["validity"]["status"], "INVALID")

    def test_population_aggregation_rule_is_mandatory(self) -> None:
        query = base_query()
        query["population_contract"].pop("aggregation_rule")
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["validity"]["status"], "INVALID")

    def test_missing_numerical_authority_refuses_correctness(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["numerical_transition"].pop("authority")
        evidence["ledgers"].pop("numerical_transition")
        result = decide_training_step(query, evidence)
        self.assertEqual(result["transition"]["numerical_transition"]["status"], "UNINSTANTIATED")
        self.assertEqual(result["correctness_claim"]["verdict"], "UNINSTANTIATED")

    def test_candidate_calibrated_authority_is_uninstantiated(self) -> None:
        query = base_query()
        query["ledgers"]["numerical_transition"]["authority"][
            "independent_of_candidate_measurements"
        ] = False
        result = decide_training_step(query, base_evidence())
        self.assertEqual(
            result["transition"]["numerical_transition"]["status"], "UNINSTANTIATED"
        )
        self.assertEqual(result["correctness_claim"]["verdict"], "UNINSTANTIATED")

    def test_unregistered_impact_endpoint_invalidates_query(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["impacts"]["posthoc_accuracy"] = {
            "status": "REJECT",
            "details": {},
        }
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["validity"]["status"], "INVALID")
        self.assertIn("unregistered impact endpoints", result["validity"]["failed_gates"][-1])

    def test_missing_ledger_declaration_invalidates_query(self) -> None:
        query = base_query()
        query["ledgers"].pop("sampling_uncertainty")
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["validity"]["status"], "INVALID")
        self.assertTrue(
            any("missing ledger declarations" in issue for issue in result["validity"]["failed_gates"])
        )

    def test_invalid_ledger_scope_invalidates_query(self) -> None:
        query = base_query()
        query["ledgers"]["numerical_transition"]["scope"] = "MAYBE"
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["validity"]["status"], "INVALID")

    def test_terminal_conformance_requires_evidence_payload(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["numerical_transition"] = {"status": "ACCEPT"}
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["transition"]["numerical_transition"]["status"], "INVALID")
        self.assertEqual(result["correctness_claim"]["verdict"], "INVALID")

    def test_binary_impact_requires_both_directions_and_boundary(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["impacts"]["clip"]["details"] = {"disagreement": 0.1}
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["impacts"]["clip"]["status"], "INVALID")
        self.assertEqual(result["correctness_claim"]["verdict"], "ACCEPT")
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_binary_impact_counts_must_match_reported_probabilities(self) -> None:
        evidence = base_evidence()
        details = evidence["ledgers"]["impacts"]["clip"]["details"]
        details.update(
            direction_0_to_1_count=2,
            direction_1_to_0_count=2,
            disagreement_count=4,
            denominator=100,
            directional_shift=0.02,
            disagreement=0.01,
        )
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["impacts"]["clip"]["status"], "INVALID")

    def test_validity_gates_require_evidence_not_only_boolean(self) -> None:
        evidence = base_evidence()
        evidence["validity_gates"][0].pop("evidence")
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["validity"]["status"], "INVALID")

    def test_unqualified_bias_variance_keys_are_rejected(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["discrepancy"]["bias"] = 1.0
        evidence["ledgers"]["discrepancy"]["variance"] = 2.0
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["discrepancy"]["status"], "INVALID")

    def test_discrepancy_estimand_requires_operational_definition(self) -> None:
        query = base_query()
        query["ledgers"]["discrepancy"]["estimands"][0].pop("aggregation_unit")
        result = decide_training_step(query, base_evidence())
        self.assertEqual(result["discrepancy"]["status"], "INVALID")

    def test_discrepancy_evidence_requires_estimate_and_scope(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["discrepancy"]["average_implementation_relative_shift"] = {
            "estimate": 0.0
        }
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["discrepancy"]["status"], "INVALID")

    def test_unregistered_discrepancy_component_is_rejected(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["discrepancy"]["posthoc_metric"] = {
            "estimate": 3.0,
            "scope_or_uncertainty": "selected after outcome inspection",
        }
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["discrepancy"]["status"], "INVALID")

    def test_invalid_supplied_optional_ledger_prevents_completion(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["attribution"] = attribution_contract()
        query["ledgers"]["attribution"]["scope"] = "OPTIONAL"
        attribution = attribution_evidence(full_integrity=True)
        attribution["effects"]["next_update"]["contrasts"].pop("INTERACTION")
        evidence["ledgers"]["attribution"] = attribution
        result = decide_training_step(query, evidence)
        self.assertEqual(result["attribution"]["status"], "INVALID")
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_sampling_unit_must_match_population_contract(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["sampling_uncertainty"]["sampling_unit"] = "token"
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["sampling_uncertainty"]["status"], "INVALID")

    def test_variability_profile_requires_every_predeclared_source(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["variability_profile"]["sources"].pop("gpu_execution")
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["variability_profile"]["status"], "INVALID")

    def test_state_sampling_cannot_be_called_runtime_randomness(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["variability_profile"]["sources"]["state_sampling"][
            "classification"
        ] = "WITHIN_STATE_RANDOMNESS"
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["variability_profile"]["status"], "INVALID")

    def test_fixed_implementation_difference_cannot_be_called_variance(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["variability_profile"]["sources"]["implementation_path"][
            "classification"
        ] = "WITHIN_STATE_RANDOMNESS"
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["variability_profile"]["status"], "INVALID")

    def test_unknown_variability_source_keeps_subject_incomplete(self) -> None:
        evidence = base_evidence()
        evidence["ledgers"]["variability_profile"]["sources"]["gpu_execution"] = {
            "classification": "UNKNOWN",
            "observation": "the protocol did not repeat this source enough to characterize it",
        }
        result = decide_training_step(base_query(), evidence)
        self.assertEqual(result["variability_profile"]["status"], "INDETERMINATE")
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_operator_causal_claim_downgrades_when_integrity_incomplete(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["attribution"] = attribution_contract()
        evidence["ledgers"]["attribution"] = attribution_evidence(full_integrity=False)
        result = decide_training_step(query, evidence)
        self.assertEqual(
            result["attribution"]["eligible_claim_level"], "INTERVENTION_DEPENDENT"
        )
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_operator_causal_eligibility_requires_every_gate(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["attribution"] = attribution_contract()
        evidence["ledgers"]["attribution"] = attribution_evidence(full_integrity=True)
        result = decide_training_step(query, evidence)
        self.assertEqual(
            result["attribution"]["eligible_claim_level"], "OPERATOR_CAUSAL_ELIGIBLE"
        )
        self.assertEqual(result["subject_instantiation"]["status"], "COMPLETE")

    def test_region_r3_i1_cannot_be_promoted_to_operator_causal(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["attribution"] = attribution_contract()
        evidence["ledgers"]["attribution"] = attribution_evidence(
            full_integrity=True, operator_identity=False
        )
        result = decide_training_step(query, evidence)
        self.assertEqual(
            result["attribution"]["eligible_claim_level"], "INTERVENTION_DEPENDENT"
        )
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_failed_monolithic_anchor_allows_only_intervention_dependent_claim(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["attribution"] = attribution_contract()
        attribution = attribution_evidence(full_integrity=True, operator_identity=False)
        attribution["integrity"]["anchor_parity"] = False
        evidence["ledgers"]["attribution"] = attribution
        result = decide_training_step(query, evidence)
        self.assertEqual(result["attribution"]["status"], "EFFECT_DETECTED")
        self.assertEqual(
            result["attribution"]["eligible_claim_level"], "INTERVENTION_DEPENDENT"
        )
        self.assertIn("anchor_parity", result["attribution"]["causal_gate_failures"])
        self.assertEqual(result["subject_instantiation"]["status"], "INCOMPLETE")

    def test_unidentified_intervention_treatment_is_invalid(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["attribution"] = attribution_contract()
        attribution = attribution_evidence(full_integrity=True)
        attribution["integrity"]["treatment_identity"] = False
        evidence["ledgers"]["attribution"] = attribution
        result = decide_training_step(query, evidence)
        self.assertEqual(result["attribution"]["status"], "INVALID")

    def test_attribution_effect_requires_every_predeclared_contrast(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["attribution"] = attribution_contract()
        attribution = attribution_evidence(full_integrity=True)
        attribution["effects"]["next_update"]["contrasts"].pop("INJECTION")
        evidence["ledgers"]["attribution"] = attribution
        result = decide_training_step(query, evidence)
        self.assertEqual(result["attribution"]["status"], "INVALID")

    def test_unknown_attribution_claim_level_is_invalid(self) -> None:
        query, evidence = base_query(), base_evidence()
        contract = attribution_contract()
        contract["requested_claim_level"] = "ROOT_CAUSE"
        query["ledgers"]["attribution"] = contract
        evidence["ledgers"]["attribution"] = attribution_evidence(full_integrity=True)
        result = decide_training_step(query, evidence)
        self.assertEqual(result["attribution"]["status"], "INVALID")

    def test_single_sampled_token_cannot_validate_sampling_law(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["impacts"]["clip"] = {
            "scope": "REQUIRED",
            "authority": "predeclared distributional test",
            "acceptance_rule": "selection-law equality region",
            "event_type": "STOCHASTIC_SELECTION_LAW",
            "boundary_geometry": "NONE",
        }
        evidence["ledgers"]["impacts"]["clip"] = {
            "status": "ACCEPT",
            "details": {"law_comparison": {}, "single_draw_only": True},
        }
        result = decide_training_step(query, evidence)
        self.assertEqual(result["impacts"]["clip"]["status"], "INVALID")

    def test_set_valued_endpoint_uses_distance_not_scalar_direction(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["impacts"]["clip"] = {
            "scope": "REQUIRED",
            "authority": "predeclared top-k contract",
            "acceptance_rule": "no set change",
            "event_type": "SET_VALUED",
            "boundary_geometry": "NATURAL",
        }
        evidence["ledgers"]["impacts"]["clip"] = {
            "status": "REJECT",
            "details": {
                "disagreement": 0.1,
                "distance_or_cost": {"metric": "Jaccard", "mean": 0.02},
                "boundary_exposure": {"k_vs_k_plus_1_margin": 1e-4},
            },
        }
        result = decide_training_step(query, evidence)
        self.assertEqual(result["impacts"]["clip"]["status"], "REJECT")
        self.assertNotIn(
            "directional_shift", result["impacts"]["clip"]["evidence"]["details"]
        )

    def test_continuous_impact_requires_signed_effect_and_cost(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["impacts"] = {
            "loss_change": {
                "scope": "REQUIRED",
                "authority": "predeclared application loss contract",
                "acceptance_rule": "absolute paired loss effect below frozen bound",
                "event_type": "CONTINUOUS",
                "boundary_geometry": "NONE",
            }
        }
        evidence["ledgers"]["impacts"] = {
            "loss_change": {
                "status": "REJECT",
                "details": {
                    "signed_effect": 0.2,
                    "distance_or_cost": {"absolute": 0.2},
                },
            }
        }
        result = decide_training_step(query, evidence)
        self.assertEqual(result["impacts"]["loss_change"]["status"], "REJECT")
        evidence["ledgers"]["impacts"]["loss_change"]["details"].pop("signed_effect")
        result = decide_training_step(query, evidence)
        self.assertEqual(result["impacts"]["loss_change"]["status"], "INVALID")

    def test_vector_impact_requires_geometry_without_fabricated_direction(self) -> None:
        query, evidence = base_query(), base_evidence()
        query["ledgers"]["impacts"] = {
            "parameter_update": {
                "scope": "REQUIRED",
                "authority": "predeclared update compatibility contract",
                "acceptance_rule": "paired update distance below frozen bound",
                "event_type": "VECTOR",
                "boundary_geometry": "NONE",
            }
        }
        evidence["ledgers"]["impacts"] = {
            "parameter_update": {
                "status": "REJECT",
                "details": {
                    "geometry": "parameter-vector L2",
                    "distance_or_cost": {"l2": 0.3},
                },
            }
        }
        result = decide_training_step(query, evidence)
        endpoint = result["impacts"]["parameter_update"]
        self.assertEqual(endpoint["status"], "REJECT")
        self.assertNotIn("directional_shift", endpoint["evidence"]["details"])

    def test_inputs_are_not_mutated(self) -> None:
        query, evidence = base_query(), base_evidence()
        query_before, evidence_before = copy.deepcopy(query), copy.deepcopy(evidence)
        decide_training_step(query, evidence)
        self.assertEqual(query, query_before)
        self.assertEqual(evidence, evidence_before)


if __name__ == "__main__":
    unittest.main()
