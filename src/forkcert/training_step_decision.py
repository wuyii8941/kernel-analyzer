"""Fail-closed decision core for structured Training-Step Oracle records.

This module does not manufacture numerical truth or statistical evidence.  It
checks that a predeclared query and its evidence satisfy the Oracle record
contract, preserves separate ledgers, and derives only the claims licensed by
those ledgers.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


QUERY_SCHEMA = "forkcert.training-step-oracle-query.v0.3"
EVIDENCE_SCHEMA = "forkcert.training-step-oracle-evidence.v0.3"
RESULT_SCHEMA = "forkcert.training-step-oracle-result.v0.3"

SCOPES = {"REQUIRED", "OPTIONAL", "NOT_IN_SCOPE"}
CONFORMANCE_STATUSES = {
    "ACCEPT",
    "REJECT",
    "INDETERMINATE",
    "UNINSTANTIATED",
    "NOT_IN_SCOPE",
    "INVALID",
    "INAPPLICABLE",
}
MEASUREMENT_STATUSES = {
    "ESTIMATED",
    "INDETERMINATE",
    "UNINSTANTIATED",
    "NOT_IN_SCOPE",
    "INVALID",
    "INAPPLICABLE",
}
VARIABILITY_STATUSES = {
    "CHARACTERIZED",
    "INDETERMINATE",
    "UNINSTANTIATED",
    "NOT_IN_SCOPE",
    "INVALID",
    "INAPPLICABLE",
}
VARIABILITY_SOURCE_KINDS = {
    "STATE_SAMPLING",
    "BATCH_TOKEN_SAMPLING",
    "ALGORITHMIC_RNG",
    "EXECUTION_NONDETERMINISM",
    "AUTOTUNING_SELECTION",
    "STOCHASTIC_ROUNDING",
    "FIXED_IMPLEMENTATION_DIFFERENCE",
}
SOURCE_CLASSIFICATIONS = {
    "STATE_SAMPLING": {"BETWEEN_STATE_DESIGN", "ABSENT_OR_DISABLED", "UNKNOWN"},
    "BATCH_TOKEN_SAMPLING": {"BETWEEN_STATE_DESIGN", "FIXED_BY_PROTOCOL", "UNKNOWN"},
    "ALGORITHMIC_RNG": {
        "WITHIN_STATE_RANDOMNESS",
        "FIXED_BY_PROTOCOL",
        "ABSENT_OR_DISABLED",
        "UNKNOWN",
    },
    "EXECUTION_NONDETERMINISM": {
        "WITHIN_STATE_RANDOMNESS",
        "FIXED_BY_PROTOCOL",
        "ABSENT_OR_DISABLED",
        "UNKNOWN",
    },
    "AUTOTUNING_SELECTION": {
        "WITHIN_STATE_RANDOMNESS",
        "FIXED_BY_PROTOCOL",
        "ABSENT_OR_DISABLED",
        "UNKNOWN",
    },
    "STOCHASTIC_ROUNDING": {
        "WITHIN_STATE_RANDOMNESS",
        "FIXED_BY_PROTOCOL",
        "ABSENT_OR_DISABLED",
        "UNKNOWN",
    },
    "FIXED_IMPLEMENTATION_DIFFERENCE": {"FIXED_IMPLEMENTATION_EFFECT"},
}
ATTRIBUTION_STATUSES = {
    "EFFECT_DETECTED",
    "NO_EFFECT_DETECTED",
    "INDETERMINATE",
    "UNINSTANTIATED",
    "NOT_IN_SCOPE",
    "INVALID",
    "INAPPLICABLE",
}
OPERATOR_LEDGER_STATUSES = {
    "RECORDED",
    "PARTIAL",
    "INDETERMINATE",
    "UNINSTANTIATED",
    "NOT_IN_SCOPE",
    "INVALID",
    "INAPPLICABLE",
}
LOCAL_OPERATOR_VERDICTS = {
    "ACCEPT",
    "REJECT",
    "INDETERMINATE",
    "UNINSTANTIATED",
    "INAPPLICABLE",
}
FORBIDDEN_DISCREPANCY_KEYS = {
    "bias",
    "variance",
    "compiler_bias",
    "floating_point_variance",
    "total_error",
}
CORE_DISCREPANCY_ESTIMANDS = {
    "average_implementation_relative_shift",
    "state_conditioned_heterogeneity",
    "within_state_runtime_variability",
}
DISCREPANCY_DEFINITION_FIELDS = {
    "observable",
    "comparison",
    "aggregation_unit",
    "geometry",
}
ATTRIBUTION_INTEGRITY_GATES = {
    "treatment_identity",
    "non_target_preserved",
    "anchor_parity",
    "repair_reported",
    "injection_reported",
    "interactions_reported",
    "heldout_replication",
}
ATTRIBUTION_CONTRASTS = {"TOTAL", "REPAIR", "INJECTION", "INTERACTION"}
ATTRIBUTION_SUBJECTS = {
    "SEMANTIC_OPERATOR",
    "OPERATOR_BOUNDARY",
    "REGION",
    "KERNEL",
    "BRANCH_FUNCTION",
}
ATTRIBUTION_CLAIM_LEVELS = {"INTERVENTION_DEPENDENT", "OPERATOR_CAUSAL"}
CORRESPONDENCE_LEVELS = {"R0", "R1", "R2", "R3", "R4"}
INTERVENTION_LEVELS = {"I0", "I1", "I2", "I3"}
POPULATION_KINDS = {"MATCHED_STATE", "OPERATOR_COVERAGE"}
TRAJECTORY_ANCHORS = {
    "REFERENCE_TRAJECTORY",
    "CANDIDATE_TRAJECTORY",
    "EXTERNAL_FROZEN",
    "SYNTHETIC",
    "NOT_APPLICABLE",
}
ENRICHMENT_TYPES = {
    "NONE",
    "EVENT_CONDITIONED",
    "BOUNDARY_STRESS",
    "SYNTHETIC_CONTROL",
    "NOT_APPLICABLE",
}
AUTHORITY_KINDS = {
    "DOCUMENTED_SEMANTICS",
    "FORMAL_RELATION",
    "HIGH_PRECISION_REFERENCE",
    "CERTIFIED_ERROR_ENVELOPE",
    "CONFIRMED_WRONG_CODE_RELATION",
    "APPLICATION_CONTRACT",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _scope(contract: Mapping[str, Any]) -> str:
    value = str(contract.get("scope", "REQUIRED"))
    return value if value in SCOPES else "REQUIRED"


def _independent_authority(value: Any) -> bool:
    authority = _mapping(value)
    return bool(
        authority.get("kind") in AUTHORITY_KINDS
        and authority.get("source")
        and authority.get("scope")
        and authority.get("independent_of_candidate_measurements") is True
        and authority.get("acceptance_rule_frozen") is True
    )


def _contract_authority(contract: Mapping[str, Any], kind: str) -> bool:
    if kind in {"exact_transition", "numerical_transition", "stochastic_transition"}:
        return _independent_authority(contract.get("authority"))
    if kind == "impact":
        return bool(contract.get("acceptance_rule")) and bool(contract.get("authority"))
    if kind == "discrepancy":
        return bool(contract.get("estimands"))
    if kind == "sampling_uncertainty":
        return bool(contract.get("sampling_unit")) and bool(contract.get("method"))
    if kind == "variability_profile":
        return bool(contract.get("sources"))
    if kind == "attribution":
        return bool(contract.get("intervention"))
    if kind == "operator_conformance":
        return bool(contract.get("coverage_rule"))
    raise ValueError(f"unknown ledger kind: {kind}")


def _default_status(contract: Mapping[str, Any], evidence: Any, kind: str) -> str:
    if _scope(contract) == "NOT_IN_SCOPE":
        return "NOT_IN_SCOPE"
    if not _contract_authority(contract, kind):
        return "UNINSTANTIATED"
    if not isinstance(evidence, Mapping):
        return "INDETERMINATE"
    return str(evidence.get("status", "INDETERMINATE"))


def _record(
    contract: Mapping[str, Any], evidence: Any, kind: str, allowed: set[str]
) -> dict[str, Any]:
    status = _default_status(contract, evidence, kind)
    record = {
        "scope": _scope(contract),
        "status": status if status in allowed else "INVALID",
        "authority": contract.get("authority"),
        "evidence": deepcopy(dict(evidence)) if isinstance(evidence, Mapping) else None,
    }
    if status not in allowed:
        record["failure"] = f"unsupported {kind} status: {status}"
    if record["scope"] == "NOT_IN_SCOPE" and status != "NOT_IN_SCOPE":
        record["status"] = "INVALID"
        record["failure"] = "evidence supplied a verdict for a NOT_IN_SCOPE ledger"
    if (
        kind in {"exact_transition", "numerical_transition", "stochastic_transition"}
        and record["status"] in {"ACCEPT", "REJECT"}
        and isinstance(evidence, Mapping)
        and not any(key in evidence for key in ("evidence", "details", "test", "witnesses"))
    ):
        record["status"] = "INVALID"
        record["failure"] = "terminal conformance verdict has no evidence payload"
    return record


def _required_declared(
    query: Mapping[str, Any], name: str, issues: list[str], *, omissions: bool = False
) -> Mapping[str, Any]:
    item = _mapping(query.get(name))
    if not item or item.get("declared") is not True:
        issues.append(f"{name} is not declared")
    if omissions and item.get("outcome_relevant_omissions", []) != []:
        issues.append(f"{name} has outcome-relevant omissions")
    return item


def _validity(query: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    state = _required_declared(query, "state_contract", issues, omissions=True)
    implementation = _required_declared(query, "implementation_contract", issues)
    observable = _required_declared(query, "observable_contract", issues, omissions=True)
    population = _required_declared(query, "population_contract", issues)
    randomness = _required_declared(query, "randomness_contract", issues)

    if not implementation.get("reference") or not implementation.get("candidate"):
        issues.append("reference and candidate identities are required")
    if not state.get("fields"):
        issues.append("state_contract fields are required")
    if not observable.get("fields"):
        issues.append("observable_contract fields are required")
    if not population.get("population_id") or not population.get("sampling_unit"):
        issues.append("population_id and actual sampling_unit are required")
    if population.get("claim_kind") not in {"FINITE_BANK", "POPULATION"}:
        issues.append("population claim_kind must be FINITE_BANK or POPULATION")
    population_kind = str(population.get("population_kind", ""))
    if population_kind not in POPULATION_KINDS:
        issues.append("population_kind must be MATCHED_STATE or OPERATOR_COVERAGE")
    if not population.get("selection_design"):
        issues.append("population selection_design is required")
    if not population.get("aggregation_rule"):
        issues.append("population aggregation_rule is required")
    strata = population.get("strata")
    if not isinstance(strata, list) or not strata:
        issues.append("population strata must be a nonempty list")
    else:
        stratum_ids: set[str] = set()
        for raw in strata:
            item = _mapping(raw)
            stratum_id = str(item.get("id", ""))
            label = stratum_id or "<unnamed>"
            if not stratum_id or stratum_id in stratum_ids:
                issues.append("population stratum ids must be unique and nonempty")
            else:
                stratum_ids.add(stratum_id)
            if not item.get("provenance") or not item.get("inclusion_rule"):
                issues.append(f"population stratum {label} lacks provenance/inclusion rule")
            anchor = str(item.get("trajectory_anchor", ""))
            enrichment = str(item.get("enrichment", ""))
            if anchor not in TRAJECTORY_ANCHORS:
                issues.append(f"population stratum {label} has invalid trajectory anchor")
            elif population_kind == "MATCHED_STATE" and anchor == "NOT_APPLICABLE":
                issues.append("matched-state strata require an explicit trajectory/data anchor")
            elif population_kind == "OPERATOR_COVERAGE" and anchor != "NOT_APPLICABLE":
                issues.append("operator-coverage strata must use NOT_APPLICABLE trajectory anchor")
            if enrichment not in ENRICHMENT_TYPES:
                issues.append(f"population stratum {label} has invalid enrichment type")
            elif population_kind == "MATCHED_STATE" and enrichment == "NOT_APPLICABLE":
                issues.append("matched-state strata require an explicit enrichment status")
            elif population_kind == "OPERATOR_COVERAGE" and enrichment != "NOT_APPLICABLE":
                issues.append("operator-coverage strata must use NOT_APPLICABLE enrichment")
            elif anchor == "SYNTHETIC" and enrichment != "SYNTHETIC_CONTROL":
                issues.append("synthetic strata must be labelled SYNTHETIC_CONTROL")
    if not randomness.get("mode") or not randomness.get("coupling"):
        issues.append("randomness mode and coupling protocol are required")

    ledgers = query.get("ledgers")
    required_ledger_keys = {
        "exact_transition",
        "numerical_transition",
        "stochastic_transition",
        "impacts",
        "discrepancy",
        "sampling_uncertainty",
        "variability_profile",
        "operator_conformance",
        "attribution",
    }
    if not isinstance(ledgers, Mapping):
        issues.append("ledgers are not declared")
    else:
        missing_ledgers = sorted(required_ledger_keys.difference(ledgers))
        if missing_ledgers:
            issues.append("missing ledger declarations: " + ", ".join(missing_ledgers))
        for name in required_ledger_keys.difference({"impacts"}):
            contract = _mapping(ledgers.get(name))
            if contract.get("scope") not in SCOPES:
                issues.append(f"{name} has missing or invalid scope")
        impacts = ledgers.get("impacts")
        if not isinstance(impacts, Mapping):
            issues.append("impacts must be a mapping, even when empty")
        else:
            for endpoint_id, contract in impacts.items():
                if not endpoint_id or _mapping(contract).get("scope") not in SCOPES:
                    issues.append(f"impact endpoint {endpoint_id!r} has missing or invalid scope")

    gates = evidence.get("validity_gates")
    if not isinstance(gates, list) or not gates:
        issues.append("at least one evidence-backed validity gate is required")
        gates = []
    failed_gates = [
        str(gate.get("id", "unnamed"))
        for gate in gates
        if gate.get("passed") is not True or not gate.get("evidence")
    ]
    if failed_gates:
        issues.append("failed validity gates: " + ", ".join(failed_gates))

    identity = _mapping(evidence.get("candidate_identity"))
    identity_status = str(identity.get("status", "INVALID"))
    if identity_status == "INAPPLICABLE" and not issues:
        return {
            "status": "INAPPLICABLE",
            "failed_gates": [],
            "candidate_identity": deepcopy(dict(identity)),
        }
    if identity_status == "VALID" and not identity.get("evidence"):
        issues.append("candidate identity has no evidence")
    elif identity_status != "VALID":
        issues.append(f"candidate identity is {identity_status}")
    return {
        "status": "INVALID" if issues else "VALID",
        "failed_gates": issues,
        "candidate_identity": deepcopy(dict(identity)) if identity else None,
    }


def _impact_record(
    endpoint_id: str, contract: Mapping[str, Any], evidence: Any
) -> dict[str, Any]:
    record = _record(contract, evidence, "impact", CONFORMANCE_STATUSES)
    record["endpoint_id"] = endpoint_id
    if record["status"] not in {"ACCEPT", "REJECT"} or not isinstance(evidence, Mapping):
        return record
    details = _mapping(evidence.get("details"))
    event_type = contract.get("event_type")
    missing: list[str] = []
    if event_type == "BINARY":
        for field in ("directional_shift", "disagreement"):
            if field not in details:
                missing.append(field)
        if not missing:
            directional = float(details["directional_shift"])
            disagreement = float(details["disagreement"])
            if not (-1.0 <= directional <= 1.0):
                missing.append("directional_shift in [-1,1]")
            if not (0.0 <= disagreement <= 1.0):
                missing.append("disagreement in [0,1]")
            if abs(directional) > disagreement + 1e-15:
                missing.append("abs(directional_shift) <= disagreement")
            count_fields = {
                "direction_0_to_1_count",
                "direction_1_to_0_count",
                "disagreement_count",
                "denominator",
            }
            if count_fields.intersection(details):
                if not count_fields.issubset(details):
                    missing.append("complete directional counts and denominator")
                else:
                    up = int(details["direction_0_to_1_count"])
                    down = int(details["direction_1_to_0_count"])
                    total = int(details["disagreement_count"])
                    denominator = int(details["denominator"])
                    if min(up, down, total) < 0 or denominator <= 0:
                        missing.append("nonnegative counts and positive denominator")
                    elif (
                        total != up + down
                        or abs(directional - (up - down) / denominator) > 1e-15
                        or abs(disagreement - total / denominator) > 1e-15
                    ):
                        missing.append("direction/disagreement values consistent with counts")
    elif event_type in {"CATEGORICAL", "SET_VALUED"}:
        if "disagreement" not in details:
            missing.append("disagreement")
        if "distance_or_cost" not in details:
            missing.append("distance_or_cost")
    elif event_type == "CONTINUOUS":
        if "signed_effect" not in details:
            missing.append("signed_effect")
        if "distance_or_cost" not in details:
            missing.append("distance_or_cost")
    elif event_type == "VECTOR":
        if "distance_or_cost" not in details:
            missing.append("distance_or_cost")
        if not details.get("geometry"):
            missing.append("geometry")
    elif event_type == "STOCHASTIC_SELECTION_LAW":
        if not details.get("law_comparison"):
            missing.append("law_comparison")
        if details.get("single_draw_only") is True:
            missing.append("law-level evidence (single draw is insufficient)")
    else:
        missing.append("recognized event_type")
    if contract.get("boundary_geometry") == "NATURAL" and "boundary_exposure" not in details:
        missing.append("boundary_exposure")
    if missing:
        record["status"] = "INVALID"
        record["failure"] = "missing endpoint evidence: " + ", ".join(missing)
    return record


def _discrepancy_record(contract: Mapping[str, Any], evidence: Any) -> dict[str, Any]:
    record = _record(contract, evidence, "discrepancy", MEASUREMENT_STATUSES)
    if record["status"] != "ESTIMATED" or not isinstance(evidence, Mapping):
        return record
    failures: list[str] = []
    estimand_rows = contract.get("estimands")
    declared: dict[str, Mapping[str, Any]] = {}
    if not isinstance(estimand_rows, list) or not estimand_rows:
        failures.append("discrepancy estimands must be a nonempty list")
    else:
        for row in estimand_rows:
            item = _mapping(row)
            estimand_id = str(item.get("id", ""))
            if not estimand_id or estimand_id in declared:
                failures.append("discrepancy estimand ids must be unique and nonempty")
                continue
            missing_definition = sorted(
                field for field in DISCREPANCY_DEFINITION_FIELDS if not item.get(field)
            )
            if missing_definition:
                failures.append(
                    f"{estimand_id} missing definition fields: "
                    + ", ".join(missing_definition)
                )
            declared[estimand_id] = item
    missing_core = sorted(CORE_DISCREPANCY_ESTIMANDS.difference(declared))
    if missing_core:
        failures.append("missing core discrepancy estimands: " + ", ".join(missing_core))

    forbidden = sorted(FORBIDDEN_DISCREPANCY_KEYS.intersection(evidence))
    if forbidden:
        failures.append("forbidden unqualified discrepancy keys: " + ", ".join(forbidden))
    evidence_ids = set(evidence).difference({"status"})
    missing_evidence = sorted(set(declared).difference(evidence_ids))
    extra_evidence = sorted(evidence_ids.difference(declared))
    if missing_evidence:
        failures.append("missing discrepancy evidence: " + ", ".join(missing_evidence))
    if extra_evidence:
        failures.append("unregistered discrepancy evidence: " + ", ".join(extra_evidence))
    for estimand_id in set(declared).intersection(evidence_ids):
        item = _mapping(evidence.get(estimand_id))
        if "estimate" not in item:
            failures.append(f"{estimand_id} has no estimate")
        if not item.get("scope_or_uncertainty"):
            failures.append(f"{estimand_id} has no scope or uncertainty record")
    if failures:
        record["status"] = "INVALID"
        record["failure"] = failures
    else:
        record["estimand_definitions"] = deepcopy(
            {estimand_id: dict(item) for estimand_id, item in declared.items()}
        )
    return record


def _sampling_record(
    contract: Mapping[str, Any], evidence: Any, population: Mapping[str, Any]
) -> dict[str, Any]:
    record = _record(contract, evidence, "sampling_uncertainty", MEASUREMENT_STATUSES)
    if record["status"] == "ESTIMATED" and isinstance(evidence, Mapping):
        if evidence.get("sampling_unit") != population.get("sampling_unit"):
            record["status"] = "INVALID"
            record["failure"] = "uncertainty sampling unit differs from population contract"
    if population.get("claim_kind") == "POPULATION" and record["status"] != "ESTIMATED":
        record["population_claim_licensed"] = False
    else:
        record["population_claim_licensed"] = population.get("claim_kind") == "POPULATION"
    return record


def _variability_record(contract: Mapping[str, Any], evidence: Any) -> dict[str, Any]:
    record = _record(contract, evidence, "variability_profile", VARIABILITY_STATUSES)
    if record["status"] != "CHARACTERIZED" or not isinstance(evidence, Mapping):
        return record
    declared_rows = contract.get("sources")
    if not isinstance(declared_rows, list) or not declared_rows:
        record["status"] = "INVALID"
        record["failure"] = "variability sources are not declared"
        return record
    declared: dict[str, str] = {}
    failures: list[str] = []
    for row in declared_rows:
        item = _mapping(row)
        source_id, kind = str(item.get("id", "")), str(item.get("kind", ""))
        if not source_id or source_id in declared:
            failures.append("variability source ids must be unique and nonempty")
            continue
        if kind not in VARIABILITY_SOURCE_KINDS:
            failures.append(f"unknown variability source kind for {source_id}: {kind}")
            continue
        declared[source_id] = kind
    measured = evidence.get("sources")
    if not isinstance(measured, Mapping):
        failures.append("variability evidence sources must be a mapping")
        measured = {}
    missing = sorted(set(declared).difference(measured))
    extra = sorted(set(measured).difference(declared))
    if missing:
        failures.append("missing variability sources: " + ", ".join(missing))
    if extra:
        failures.append("unregistered variability sources: " + ", ".join(extra))
    for source_id, kind in declared.items():
        item = _mapping(measured.get(source_id))
        classification = str(item.get("classification", ""))
        if classification not in SOURCE_CLASSIFICATIONS[kind]:
            failures.append(
                f"{source_id} classification {classification!r} is invalid for {kind}"
            )
        if not item.get("observation"):
            failures.append(f"{source_id} has no evidence-backed observation")
    if failures:
        record["status"] = "INVALID"
        record["failure"] = failures
    else:
        unknown = sorted(
            source_id
            for source_id in declared
            if _mapping(measured.get(source_id)).get("classification") == "UNKNOWN"
        )
        if unknown:
            record["status"] = "INDETERMINATE"
            record["unknown_sources"] = unknown
    return record


def _attribution_record(contract: Mapping[str, Any], evidence: Any) -> dict[str, Any]:
    record = _record(contract, evidence, "attribution", ATTRIBUTION_STATUSES)
    record["requested_claim_level"] = contract.get(
        "requested_claim_level", "INTERVENTION_DEPENDENT"
    )
    record["eligible_claim_level"] = "NONE"
    if record["status"] in {
        "NOT_IN_SCOPE",
        "UNINSTANTIATED",
        "INDETERMINATE",
        "INVALID",
        "INAPPLICABLE",
    }:
        return record
    endpoints = contract.get("endpoints")
    required_contrasts = contract.get("required_contrasts")
    failures: list[str] = []
    if record["requested_claim_level"] not in ATTRIBUTION_CLAIM_LEVELS:
        failures.append("unknown attribution requested_claim_level")
    endpoint_ids: list[str] = []
    if not isinstance(endpoints, list) or not endpoints:
        failures.append("attribution endpoints are not declared")
    else:
        for endpoint in endpoints:
            endpoint_id = str(_mapping(endpoint).get("id", ""))
            if not endpoint_id or endpoint_id in endpoint_ids:
                failures.append("attribution endpoint ids must be unique and nonempty")
            else:
                endpoint_ids.append(endpoint_id)
    if (
        not isinstance(required_contrasts, list)
        or not required_contrasts
        or set(required_contrasts).difference(ATTRIBUTION_CONTRASTS)
    ):
        failures.append("required_contrasts must be a nonempty subset of TOTAL/REPAIR/INJECTION/INTERACTION")
        required_contrasts = []

    treatment = _mapping(evidence.get("treatment")) if isinstance(evidence, Mapping) else {}
    subject = str(treatment.get("claimed_subject", ""))
    correspondence = str(treatment.get("correspondence_level", ""))
    intervention = str(treatment.get("intervention_level", ""))
    if subject not in ATTRIBUTION_SUBJECTS:
        failures.append("unknown or missing attribution claimed_subject")
    if correspondence not in CORRESPONDENCE_LEVELS:
        failures.append("unknown or missing correspondence_level")
    if intervention not in INTERVENTION_LEVELS:
        failures.append("unknown or missing intervention_level")

    effects = evidence.get("effects") if isinstance(evidence, Mapping) else None
    if not isinstance(effects, Mapping):
        failures.append("attribution effects must be a mapping")
        effects = {}
    missing_endpoints = sorted(set(endpoint_ids).difference(effects))
    extra_endpoints = sorted(set(effects).difference(endpoint_ids))
    if missing_endpoints:
        failures.append("missing attribution endpoints: " + ", ".join(missing_endpoints))
    if extra_endpoints:
        failures.append("unregistered attribution endpoints: " + ", ".join(extra_endpoints))
    for endpoint_id in endpoint_ids:
        contrasts = _mapping(_mapping(effects.get(endpoint_id)).get("contrasts"))
        missing_contrasts = sorted(set(required_contrasts).difference(contrasts))
        extra_contrasts = sorted(set(contrasts).difference(ATTRIBUTION_CONTRASTS))
        if missing_contrasts:
            failures.append(
                f"{endpoint_id} missing contrasts: " + ", ".join(missing_contrasts)
            )
        if extra_contrasts:
            failures.append(
                f"{endpoint_id} has unknown contrasts: " + ", ".join(extra_contrasts)
            )
        for contrast_name in set(required_contrasts).intersection(contrasts):
            contrast = _mapping(contrasts.get(contrast_name))
            if not contrast.get("definition"):
                failures.append(f"{endpoint_id}/{contrast_name} has no estimand definition")
            if "estimate" not in contrast:
                failures.append(f"{endpoint_id}/{contrast_name} has no estimate")
            if not contrast.get("scope_or_uncertainty"):
                failures.append(
                    f"{endpoint_id}/{contrast_name} has no scope or uncertainty record"
                )

    role_rows = evidence.get("roles") if isinstance(evidence, Mapping) else None
    roles: set[str] = set()
    if not isinstance(role_rows, list) or not role_rows:
        failures.append("at least one evidence-backed attribution role is required")
    else:
        for row in role_rows:
            item = _mapping(row)
            role = str(item.get("role", ""))
            if role not in {"SOURCE", "PROPAGATION", "BOUNDARY_CONVERSION"}:
                failures.append(f"unknown attribution role: {role}")
            elif not item.get("evidence"):
                failures.append(f"attribution role {role} has no evidence")
            else:
                roles.add(role)
    if failures:
        record["status"] = "INVALID"
        record["failure"] = failures
        return record

    integrity = _mapping(evidence.get("integrity")) if isinstance(evidence, Mapping) else {}
    missing = sorted(gate for gate in ATTRIBUTION_INTEGRITY_GATES if integrity.get(gate) is not True)
    if not integrity.get("treatment_identity"):
        record["status"] = "INVALID"
        record["failure"] = "the intervention treatment itself has not been identified"
        return record
    if missing:
        record["eligible_claim_level"] = "INTERVENTION_DEPENDENT"
        record["causal_gate_failures"] = missing
    else:
        operator_identity = (
            subject == "SEMANTIC_OPERATOR"
            and correspondence == "R4"
            and intervention == "I3"
        )
        record["eligible_claim_level"] = (
            "OPERATOR_CAUSAL_ELIGIBLE" if operator_identity else "INTERVENTION_DEPENDENT"
        )
        if not operator_identity:
            record["causal_gate_failures"] = [
                "operator causal language requires SEMANTIC_OPERATOR + R4 + I3"
            ]
    record["contrast_interpretation"] = {
        "repair": (
            "CONTEXT_SPECIFIC_REPAIR_EFFECT_ONLY"
            if "REPAIR" in required_contrasts
            else "NOT_REPORTED"
        ),
        "injection": (
            "CONTEXT_SPECIFIC_INJECTION_EFFECT_ONLY"
            if "INJECTION" in required_contrasts
            else "NOT_REPORTED"
        ),
        "unique_necessity_or_sufficiency": "NOT_LICENSED",
        "unique_root_cause": "NOT_LICENSED",
    }
    record["roles_reported"] = sorted(roles)
    return record


def _operator_conformance_record(
    contract: Mapping[str, Any], evidence: Any
) -> dict[str, Any]:
    record = _record(
        contract,
        evidence,
        "operator_conformance",
        OPERATOR_LEDGER_STATUSES,
    )
    if record["status"] in {
        "NOT_IN_SCOPE",
        "UNINSTANTIATED",
        "INVALID",
        "INAPPLICABLE",
    }:
        return record
    failures: list[str] = []
    if contract.get("claim_unit") != "SEMANTIC_OPERATOR":
        failures.append("operator conformance claim_unit must be SEMANTIC_OPERATOR")
    coverage_unit = str(contract.get("coverage_unit", ""))
    if coverage_unit not in {"INSTANCE", "FAMILY"}:
        failures.append("operator coverage_unit must be INSTANCE or FAMILY")
    entries = evidence.get("entries") if isinstance(evidence, Mapping) else None
    if not isinstance(entries, list) or not entries:
        failures.append("operator conformance entries must be a nonempty list")
        entries = []
    unit_ids: set[str] = set()
    covered_from_entries = 0
    indeterminate_entries = 0
    for raw in entries:
        item = _mapping(raw)
        unit_id = str(item.get("unit_id", ""))
        label = unit_id or "<unnamed>"
        if not unit_id or unit_id in unit_ids:
            failures.append("operator coverage-unit ids must be unique and nonempty")
        else:
            unit_ids.add(unit_id)
        if item.get("subject_kind") != "SEMANTIC_OPERATOR":
            failures.append(f"{label} is not a semantic operator")
        if not item.get("semantic_operator"):
            failures.append(f"{label} has no semantic operator identity")
        identity_level = str(item.get("identity_level", ""))
        if identity_level not in CORRESPONDENCE_LEVELS:
            failures.append(f"{label} has invalid identity level")
        verdict = str(item.get("verdict", ""))
        if verdict not in LOCAL_OPERATOR_VERDICTS:
            failures.append(f"{label} has invalid local verdict")
            continue
        terminal = verdict in {"ACCEPT", "REJECT", "INDETERMINATE"}
        if verdict == "INDETERMINATE":
            indeterminate_entries += 1
        if terminal:
            if identity_level != "R4":
                failures.append(f"{label} terminal operator verdict requires R4 identity")
            if not _independent_authority(item.get("contract_authority")):
                failures.append(
                    f"{label} terminal operator verdict has no independent contract authority"
                )
            if not item.get("evidence"):
                failures.append(f"{label} terminal operator verdict has no evidence")
            if (
                identity_level == "R4"
                and _independent_authority(item.get("contract_authority"))
                and item.get("evidence")
            ):
                covered_from_entries += 1
        elif verdict in {"UNINSTANTIATED", "INAPPLICABLE"} and not item.get("reason"):
            failures.append(f"{label} non-terminal operator record has no reason")

    coverage = _mapping(evidence.get("coverage")) if isinstance(evidence, Mapping) else {}
    try:
        encountered = int(coverage.get("encountered"))
        covered = int(coverage.get("contract_covered"))
        uncovered = int(coverage.get("unidentified_or_uncontracted"))
    except (TypeError, ValueError):
        failures.append("operator coverage counts must be integers")
        encountered = covered = uncovered = -1
    if min(encountered, covered, uncovered) < 0:
        failures.append("operator coverage counts must be nonnegative")
    elif encountered != len(entries) or encountered != covered + uncovered:
        failures.append("operator coverage counts are inconsistent with entries")
    elif covered != covered_from_entries:
        failures.append("contract_covered count is inconsistent with R4 terminal entries")
    if record["status"] == "RECORDED" and (uncovered != 0 or indeterminate_entries != 0):
        failures.append(
            "RECORDED operator ledger requires full coverage and no indeterminate entries"
        )
    if record["status"] == "PARTIAL" and uncovered <= 0:
        failures.append("PARTIAL operator ledger requires at least one uncovered instance")
    if record["status"] == "INDETERMINATE" and (
        uncovered != 0 or indeterminate_entries <= 0
    ):
        failures.append(
            "INDETERMINATE operator ledger requires full coverage and at least one indeterminate entry"
        )
    if failures:
        record["status"] = "INVALID"
        record["failure"] = failures
    else:
        record["aggregation_rule"] = (
            "local operator verdicts are diagnostic and are never OR'ed into the "
            "whole-step correctness verdict"
        )
    return record


def _correctness(validity: str, ledgers: Mapping[str, Mapping[str, Any]]) -> str:
    if validity in {"INVALID", "INAPPLICABLE"}:
        return validity
    statuses = [
        record["status"]
        for record in ledgers.values()
        if record["scope"] == "REQUIRED"
    ]
    if "REJECT" in statuses:
        return "REJECT"
    if "INVALID" in statuses:
        return "INVALID"
    if "INAPPLICABLE" in statuses:
        return "INAPPLICABLE"
    if "INDETERMINATE" in statuses:
        return "INDETERMINATE"
    if "UNINSTANTIATED" in statuses or not statuses:
        return "UNINSTANTIATED"
    return "ACCEPT" if all(status == "ACCEPT" for status in statuses) else "UNINSTANTIATED"


def decide_training_step(query: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one query/evidence pair and return a structured, fail-closed result."""

    query = deepcopy(dict(query))
    evidence = deepcopy(dict(evidence))
    schema_issues = []
    if query.get("schema_version") != QUERY_SCHEMA:
        schema_issues.append("unsupported query schema")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        schema_issues.append("unsupported evidence schema")
    if not query.get("subject_id") or query.get("subject_id") != evidence.get("subject_id"):
        schema_issues.append("subject identity is missing or mismatched")

    validity = _validity(query, evidence)
    if schema_issues:
        validity["status"] = "INVALID"
        validity["failed_gates"] = schema_issues + validity["failed_gates"]

    contracts = _mapping(query.get("ledgers"))
    evidence_ledgers = _mapping(evidence.get("ledgers"))
    transition_records = {
        name: _record(
            _mapping(contracts.get(name)),
            evidence_ledgers.get(name),
            name,
            CONFORMANCE_STATUSES,
        )
        for name in ("exact_transition", "numerical_transition", "stochastic_transition")
    }

    impact_contracts = _mapping(contracts.get("impacts"))
    impact_evidence = _mapping(evidence_ledgers.get("impacts"))
    unregistered = sorted(set(impact_evidence).difference(impact_contracts))
    if unregistered:
        validity["status"] = "INVALID"
        validity["failed_gates"].append(
            "unregistered impact endpoints: " + ", ".join(unregistered)
        )
    impacts = {
        endpoint_id: _impact_record(
            endpoint_id, _mapping(contract), impact_evidence.get(endpoint_id)
        )
        for endpoint_id, contract in impact_contracts.items()
    }

    population = _mapping(query.get("population_contract"))
    discrepancy = _discrepancy_record(
        _mapping(contracts.get("discrepancy")), evidence_ledgers.get("discrepancy")
    )
    sampling = _sampling_record(
        _mapping(contracts.get("sampling_uncertainty")),
        evidence_ledgers.get("sampling_uncertainty"),
        population,
    )
    variability = _variability_record(
        _mapping(contracts.get("variability_profile")),
        evidence_ledgers.get("variability_profile"),
    )
    attribution = _attribution_record(
        _mapping(contracts.get("attribution")), evidence_ledgers.get("attribution")
    )
    operator_conformance = _operator_conformance_record(
        _mapping(contracts.get("operator_conformance")),
        evidence_ledgers.get("operator_conformance"),
    )

    required_records: list[tuple[str, Mapping[str, Any]]] = list(transition_records.items())
    required_records.extend((f"impact:{name}", record) for name, record in impacts.items())
    required_records.extend(
        [
            ("discrepancy", discrepancy),
            ("sampling_uncertainty", sampling),
            ("variability_profile", variability),
            ("operator_conformance", operator_conformance),
            ("attribution", attribution),
        ]
    )
    incomplete = [
        name
        for name, record in required_records
        if record["scope"] == "REQUIRED"
        and record["status"]
        not in {
            "ACCEPT",
            "REJECT",
            "ESTIMATED",
            "CHARACTERIZED",
            "EFFECT_DETECTED",
            "NO_EFFECT_DETECTED",
            "RECORDED",
        }
    ]
    incomplete.extend(
        name
        for name, record in required_records
        if record["scope"] == "OPTIONAL" and record["status"] == "INVALID"
    )
    if (
        attribution["scope"] == "REQUIRED"
        and attribution.get("requested_claim_level") == "OPERATOR_CAUSAL"
        and attribution.get("eligible_claim_level") != "OPERATOR_CAUSAL_ELIGIBLE"
    ):
        incomplete.append("attribution:operator_causal_gate")

    correctness = _correctness(validity["status"], transition_records)
    correctness_authority = {
        name: record.get("authority")
        for name, record in transition_records.items()
        if record["scope"] == "REQUIRED"
    }
    non_claims = list(_mapping(query.get("claim_scope")).get("explicit_non_claims", []))
    return {
        "schema_version": RESULT_SCHEMA,
        "subject_id": query.get("subject_id"),
        "declared_context": {
            name: deepcopy(dict(_mapping(query.get(name))))
            for name in (
                "state_contract",
                "implementation_contract",
                "observable_contract",
                "population_contract",
                "randomness_contract",
                "claim_scope",
            )
        },
        "validity": validity,
        "transition": transition_records,
        "impacts": impacts,
        "discrepancy": discrepancy,
        "sampling_uncertainty": sampling,
        "variability_profile": variability,
        "operator_conformance": operator_conformance,
        "attribution": attribution,
        "correctness_claim": {
            "verdict": correctness,
            "authorities": correctness_authority,
            "covered_scope": _mapping(query.get("claim_scope")).get("covered_scope"),
            "explicit_non_claims": non_claims,
        },
        "subject_instantiation": {
            "status": "COMPLETE" if validity["status"] == "VALID" and not incomplete else "INCOMPLETE",
            "incomplete_required_ledgers": sorted(set(incomplete)),
        },
        "aggregation_note": (
            "Correctness is derived only from required transition-conformance ledgers; "
            "local operator records, impact, discrepancy and attribution are not "
            "whole-step correctness proxies."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    query = json.loads(Path(args.query).read_text(encoding="utf-8"))
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    result = decide_training_step(query, evidence)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
