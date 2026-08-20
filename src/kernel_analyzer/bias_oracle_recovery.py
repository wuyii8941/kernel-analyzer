"""Label-blind recovery checks for the low-cost bias-oracle cascade.

This module deliberately separates two operations:

* ``predict_*`` functions read measurement artifacts but never historical
  T1--T4, SEUP, mechanism, or case verdicts;
* ``compare_recovery`` receives frozen targets only after predictions exist.

The result is a development-set recovery audit, not held-out validation.  A
direct risk screen can prioritize a case, while an escalation/abstention keeps
the denominator fail-closed without pretending that the cheap screen found the
mechanism.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import random
from typing import Any, Mapping, Sequence

from .bias_formation_v22 import (
    BiasV22Status,
    ConditionalPolicy,
    summarize_conditional_gram,
)
from .bias_oracle_feasibility import subset_square_matrix
from .reference_relative_oracle import (
    ReferenceRelativeObservation,
    certify_reference_relative,
)


class RecoveryDisposition(str, Enum):
    DIRECT_RISK_SOURCE = "DIRECT_RISK_SOURCE"
    DIRECT_RISK_RESPONSE_RECTIFICATION = "DIRECT_RISK_RESPONSE_RECTIFICATION"
    DIRECT_RISK_POPULATION_COHERENCE = "DIRECT_RISK_POPULATION_COHERENCE"
    DIRECT_RISK_REFERENCE_RELATIVE = "DIRECT_RISK_REFERENCE_RELATIVE"
    DIRECT_RISK_CROSSFIT_PROJECTION = "DIRECT_RISK_CROSSFIT_PROJECTION"
    DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE = (
        "DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE"
    )
    ESCALATE_MISSING_EVENT_MOMENT = "ESCALATE_MISSING_EVENT_MOMENT"
    ESCALATE_MISSING_TRANSPORT_JOINT_MOMENT = (
        "ESCALATE_MISSING_TRANSPORT_JOINT_MOMENT"
    )
    ABSTAIN_SOURCE_FIDELITY_FAILED = "ABSTAIN_SOURCE_FIDELITY_FAILED"
    NO_RISK_DETECTED = "NO_RISK_DETECTED"
    UNRESOLVED = "UNRESOLVED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class RecoveryPrediction:
    case_id: str
    disposition: str
    direct_risk: bool
    routed_for_exact_followup: bool
    safe_release: bool
    evidence_kind: str
    measurements: Mapping[str, Any]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BiasRiskCertificate:
    case_id: str
    verdict: str
    hit_witnesses: tuple[str, ...]
    evaluated_witnesses: tuple[str, ...]
    safe_release: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine_risk_witnesses(
    case_id: str, predictions: Sequence[RecoveryPrediction],
) -> BiasRiskCertificate:
    """Combine sufficient witnesses without converting misses into safety."""

    rows = list(predictions)
    if not rows:
        return BiasRiskCertificate(
            case_id=case_id,
            verdict="ABSTAIN",
            hit_witnesses=(),
            evaluated_witnesses=(),
            safe_release=False,
            reason="no supported risk witness was evaluated",
        )
    if any(row.case_id != case_id for row in rows):
        raise ValueError("all witness predictions must belong to the same case")
    hits = tuple(row.evidence_kind for row in rows if row.direct_risk)
    evaluated = tuple(row.evidence_kind for row in rows)
    return BiasRiskCertificate(
        case_id=case_id,
        verdict="DIRECTIONAL_RISK" if hits else "ABSTAIN",
        hit_witnesses=hits,
        evaluated_witnesses=evaluated,
        safe_release=False,
        reason=(
            "at least one independently defined sufficient risk witness fired"
            if hits else
            "no sufficient witness fired; absence of evidence is not safety"
        ),
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fixed_prefix_certificate(
    certificate: Mapping[str, Any], *, condition_id: str, repeats: int,
) -> Mapping[str, Any]:
    gram = certificate.get("complete_gram")
    ids = certificate.get("state_ids")
    digests = certificate.get("vector_digests")
    if not isinstance(gram, list) or not isinstance(ids, list) or not isinstance(digests, list):
        raise ValueError("conditional certificate lacks its complete Gram provenance")
    if len(gram) < repeats or len(ids) < repeats or len(digests) < repeats:
        raise ValueError("conditional certificate has fewer than the requested repeats")
    indices = tuple(range(repeats))
    return summarize_conditional_gram(
        subset_square_matrix(gram, indices),
        condition_id=condition_id,
        coordinate_count=int(certificate["coordinate_count"]),
        replicate_ids=[ids[index] for index in indices],
        vector_digests=[digests[index] for index in indices],
        estimand=str(certificate["estimand"]),
        reference=str(certificate["reference"]),
        policy=ConditionalPolicy(),
    )


def predict_conditional_source_risk(
    case_id: str,
    artifact: Mapping[str, Any],
    *,
    repeats: int = 4,
    condition_limit: int | None = None,
) -> RecoveryPrediction:
    """Screen a fixed-condition source experiment from its first repeats.

    The screen uses only the candidate-local and repair-local Grams.  It does
    not read aggregate/full-repeat verdicts.  Detecting a candidate-local bias
    is enough to route the case as risky; a non-centered repair or missing
    downstream result prevents a stronger source-removal claim.
    """

    try:
        states = artifact["states"]
        if not isinstance(states, list) or not states:
            raise ValueError("no fixed conditions")
        if condition_limit is not None:
            if condition_limit < 1:
                raise ValueError("condition_limit must be positive")
            states = states[:condition_limit]
        candidate_statuses: list[str] = []
        repair_statuses: list[str] = []
        downstream_statuses: dict[str, list[str]] = {}
        for state in states:
            arms = state["arms"]
            if len(arms) != 1:
                raise ValueError("recovery screen requires exactly one frozen repair arm")
            layers = next(iter(arms.values()))["conditional_debias"]["layers"]
            required = {
                "candidate_local_effect_removed",
                "repair_local_residual",
            }
            if not required.issubset(layers):
                raise ValueError("source screen is missing local candidate/repair roles")
            condition_id = str(state["state_id"])
            candidate_statuses.append(str(_fixed_prefix_certificate(
                layers["candidate_local_effect_removed"],
                condition_id=condition_id,
                repeats=repeats,
            )["status"]))
            repair_statuses.append(str(_fixed_prefix_certificate(
                layers["repair_local_residual"],
                condition_id=condition_id,
                repeats=repeats,
            )["status"]))
            for role in (
                "candidate_gradient_effect_removed",
                "candidate_sgd_update_effect_removed",
                "candidate_adamw_zero_update_effect_removed",
            ):
                if role in layers:
                    downstream_statuses.setdefault(role, []).append(str(
                        _fixed_prefix_certificate(
                            layers[role], condition_id=condition_id, repeats=repeats,
                        )["status"]
                    ))
        biased = BiasV22Status.CONDITIONAL_BIAS.value
        centered = BiasV22Status.CONDITIONAL_CENTERED.value
        candidate_biased = sum(value == biased for value in candidate_statuses)
        repair_centered = sum(value == centered for value in repair_statuses)
        repair_biased = sum(value == biased for value in repair_statuses)
        all_candidate_biased = candidate_biased == len(states)
        all_repair_centered = repair_centered == len(states)
        downstream_all_biased = {
            role: sum(value == biased for value in values)
            for role, values in downstream_statuses.items()
        }
        measurements = {
            "repeat_budget": repeats,
            "condition_count": len(states),
            "screen_is_existence_witness_not_population_prevalence": (
                condition_limit is not None
            ),
            "candidate_local_biased_conditions": candidate_biased,
            "repair_local_centered_conditions": repair_centered,
            "repair_local_biased_conditions": repair_biased,
            "repair_local_unresolved_conditions": (
                len(states) - repair_centered - repair_biased
            ),
            "downstream_biased_conditions": downstream_all_biased,
            "full_repeat_or_historical_verdict_read": False,
        }
        if all_candidate_biased:
            complete_downstream = bool(downstream_all_biased) and all(
                count == len(states) for count in downstream_all_biased.values()
            )
            if all_repair_centered and complete_downstream:
                disposition = RecoveryDisposition.DIRECT_RISK_SOURCE
                reason = (
                    "four-repeat fixed-condition screen finds candidate-local bias, "
                    "a centered repair residual, and directional downstream effects"
                )
            else:
                disposition = RecoveryDisposition.DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE
                reason = (
                    "candidate-local risk is direct, but repair centering or the "
                    "complete downstream chain remains unresolved at this budget"
                )
            return RecoveryPrediction(
                case_id=case_id,
                disposition=disposition.value,
                direct_risk=True,
                routed_for_exact_followup=(disposition != RecoveryDisposition.DIRECT_RISK_SOURCE),
                safe_release=False,
                evidence_kind="FOUR_REPEAT_FIXED_CONDITION_COMPLETE_GRAM",
                measurements=measurements,
                reason=reason,
            )
        any_candidate_biased = candidate_biased > 0
        return RecoveryPrediction(
            case_id=case_id,
            disposition=(
                RecoveryDisposition.DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE.value
                if any_candidate_biased else RecoveryDisposition.UNRESOLVED.value
            ),
            direct_risk=any_candidate_biased,
            routed_for_exact_followup=True,
            safe_release=False,
            evidence_kind="FOUR_REPEAT_FIXED_CONDITION_COMPLETE_GRAM",
            measurements=measurements,
            reason=(
                "risk appears in only part of the fixed-condition screen"
                if any_candidate_biased else
                "the four-repeat screen does not resolve candidate-local risk"
            ),
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return RecoveryPrediction(
            case_id=case_id,
            disposition=RecoveryDisposition.INVALID.value,
            direct_risk=False,
            routed_for_exact_followup=True,
            safe_release=False,
            evidence_kind="FOUR_REPEAT_FIXED_CONDITION_COMPLETE_GRAM",
            measurements={"repeat_budget": repeats},
            reason=str(exc),
        )


def predict_response_rectification_risk(
    case_id: str,
    artifact: Mapping[str, Any],
    *,
    minimum_nonoddness_ratio: float = 0.05,
    minimum_crossing_even_fraction: float = 0.5,
) -> RecoveryPrediction:
    """Detect a strong exact antithetic optimizer response-even component.

    Thresholds are engineering sensitivity gates for this development recovery
    audit, not universal SEUP or bias constants.  The continuous measurements
    remain the scientific output.
    """

    aggregate = artifact.get("aggregate")
    records = artifact.get("records")
    if not isinstance(aggregate, Mapping) or not isinstance(records, list) or not records:
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "EXACT_ANTITHETIC_OPTIMIZER_RESPONSE", {},
            "response artifact lacks aggregate or step records",
        )
    ratio = aggregate.get("optimizer_oddness_resultant_ratio")
    even_energy = math.fsum(
        float(row.get("response_even_l2", 0.0)) ** 2 for row in records
    )
    crossing = (
        math.fsum(
            float(row.get("response_even_l2", 0.0)) ** 2
            * float(row.get("response_even_energy_on_sign_crossings", 0.0))
            for row in records
        ) / max(even_energy, 1e-30)
    )
    all_losses_equal = aggregate.get("all_forward_losses_equal")
    values = (ratio, crossing)
    if not all(_finite_number(value) for value in values) or all_losses_equal is not True:
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "EXACT_ANTITHETIC_OPTIMIZER_RESPONSE", {},
            "response metrics are nonfinite or the common-forward gate failed",
        )
    exact_pair_fields = (
        "antithetic_gradient_update_persistence",
        "antithetic_update_persistence",
    )
    if not any(field in aggregate for field in exact_pair_fields):
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "EXACT_ANTITHETIC_OPTIMIZER_RESPONSE", {},
            "artifact does not declare an antithetic gradient-response arm",
        )
    measurements = {
        "steps": len(records),
        "optimizer_nonoddness_resultant_ratio": float(ratio),
        "energy_weighted_response_even_on_sign_crossings": float(crossing),
        "minimum_nonoddness_ratio": minimum_nonoddness_ratio,
        "minimum_crossing_even_fraction": minimum_crossing_even_fraction,
        "all_forward_losses_equal": True,
        "historical_mechanism_verdict_read": False,
    }
    hit = (
        float(ratio) >= minimum_nonoddness_ratio
        and float(crossing) >= minimum_crossing_even_fraction
    )
    return RecoveryPrediction(
        case_id=case_id,
        disposition=(
            RecoveryDisposition.DIRECT_RISK_RESPONSE_RECTIFICATION.value
            if hit else RecoveryDisposition.UNRESOLVED.value
        ),
        direct_risk=hit,
        routed_for_exact_followup=not hit,
        safe_release=False,
        evidence_kind="EXACT_ANTITHETIC_OPTIMIZER_RESPONSE",
        measurements=measurements,
        reason=(
            "an exact sign-symmetric gradient pair has a strong Adam response-even component"
            if hit else
            "the response-even screen does not cross its frozen development sensitivity gates"
        ),
    )


def predict_source_fidelity_boundary(
    case_id: str, artifact: Mapping[str, Any],
) -> RecoveryPrediction:
    gates = artifact.get("validity_gates")
    if not isinstance(gates, Mapping):
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "ANTITHETIC_SOURCE_FIDELITY", {}, "validity gates are missing",
        )
    fidelity = gates.get("natural_source_fidelity_every_condition")
    if fidelity is False:
        return RecoveryPrediction(
            case_id=case_id,
            disposition=RecoveryDisposition.ABSTAIN_SOURCE_FIDELITY_FAILED.value,
            direct_risk=False,
            routed_for_exact_followup=True,
            safe_release=False,
            evidence_kind="ANTITHETIC_SOURCE_FIDELITY",
            measurements={
                "natural_source_fidelity_every_condition": False,
                "minimum_natural_source_fidelity": artifact.get(
                    "minimum_natural_source_fidelity"
                ),
                "mechanism_verdict_read": False,
            },
            reason=(
                "the exact projected antithetic probe does not faithfully represent "
                "the natural source in every frozen condition"
            ),
        )
    return RecoveryPrediction(
        case_id=case_id,
        disposition=RecoveryDisposition.UNRESOLVED.value,
        direct_risk=False,
        routed_for_exact_followup=True,
        safe_release=False,
        evidence_kind="ANTITHETIC_SOURCE_FIDELITY",
        measurements={"natural_source_fidelity_every_condition": fidelity},
        reason="source fidelity alone is not a risk verdict",
    )


def predict_reference_relative_risk(
    case_id: str, artifact: Mapping[str, Any],
) -> RecoveryPrediction:
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) < 4:
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "REFERENCE_RELATIVE_MOVING_FRAME", {}, "reference-relative rows are missing",
        )
    try:
        observations = [ReferenceRelativeObservation(
            condition_id=str(row["condition_id"]),
            error_reference_dot=float(row["error_reference_dot"]),
            error_energy=float(row["error_energy"]),
            reference_energy=float(row["reference_energy"]),
        ) for row in rows]
        certificate = certify_reference_relative(observations)
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "REFERENCE_RELATIVE_MOVING_FRAME", {}, str(exc),
        )
    hit = certificate.status == "REFERENCE_RELATIVE_DIRECTIONAL_RISK"
    return RecoveryPrediction(
        case_id=case_id,
        disposition=(
            RecoveryDisposition.DIRECT_RISK_REFERENCE_RELATIVE.value
            if hit else RecoveryDisposition.UNRESOLVED.value
        ),
        direct_risk=hit,
        routed_for_exact_followup=not hit,
        safe_release=False,
        evidence_kind="REFERENCE_RELATIVE_MOVING_FRAME",
        measurements={
            **certificate.as_dict(),
            "reference_frame_uses_candidate_error": False,
            "trajectory_direction_used": False,
        },
        reason=(
            "candidate-repair error has a nonzero mean multiplicative component "
            "in the same-state repair-update frame"
            if hit else
            "same-state reference-relative component is centered or unresolved"
        ),
    )


def predict_population_coherence_risk(
    case_id: str,
    population: Mapping[str, Any],
    *,
    states: int = 4,
    risk_margin: float = 0.25,
) -> RecoveryPrediction:
    """Cheap sufficient screen for a shared parameter-coordinate component.

    This is deliberately one-way: a hit is a risk witness, while a miss is not
    a safety certificate because state-conditioned moving directions may exist.
    """

    try:
        gram = population["complete_gram"]
        if not isinstance(gram, list) or len(gram) < states:
            raise ValueError("population Gram has too few states")
        subset = [[float(gram[i][j]) for j in range(states)] for i in range(states)]
        trace = math.fsum(subset[i][i] for i in range(states))
        if trace <= 0.0:
            raise ValueError("population Gram has no energy")
        resultant = math.fsum(math.fsum(row) for row in subset)
        cross_state_ratio = (resultant - trace) / ((states - 1) * trace)
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "FOUR_STATE_POPULATION_COHERENCE", {}, str(exc),
        )
    hit = cross_state_ratio >= risk_margin
    return RecoveryPrediction(
        case_id=case_id,
        disposition=(
            RecoveryDisposition.DIRECT_RISK_POPULATION_COHERENCE.value
            if hit else RecoveryDisposition.UNRESOLVED.value
        ),
        direct_risk=hit,
        routed_for_exact_followup=not hit,
        safe_release=False,
        evidence_kind="FOUR_STATE_POPULATION_COHERENCE",
        measurements={
            "state_count": states,
            "cross_state_ratio": cross_state_ratio,
            "risk_margin": risk_margin,
            "candidate_error_fitted_direction_used": False,
            "miss_is_safety_certificate": False,
        },
        reason=(
            "complete-vector Gram exposes a shared parameter-coordinate component"
            if hit else
            "no shared component at the screening margin; moving-frame tests remain required"
        ),
    )


def predict_crossfit_projection_risk(
    case_id: str,
    projections: Sequence[float],
    *,
    evaluation_count: int = 8,
    basis_frozen_before_evaluation: bool,
    minimum_absolute_mean: float = 1e-8,
    bootstrap_draws: int = 4000,
    seed: int = 20260820,
) -> RecoveryPrediction:
    """Pilot-updated scalar screen with a separately frozen carrier basis."""

    values = [float(value) for value in projections[:evaluation_count]]
    if (
        not basis_frozen_before_evaluation or len(values) < evaluation_count
        or any(not math.isfinite(value) for value in values)
    ):
        return RecoveryPrediction(
            case_id, RecoveryDisposition.INVALID.value, False, True, False,
            "CROSSFIT_FROZEN_PROJECTION", {},
            "projection basis is not frozen or evaluation values are invalid",
        )
    mean = math.fsum(values) / len(values)
    rng = random.Random(seed)
    boot = sorted(
        math.fsum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(bootstrap_draws)
    )
    lower = boot[int(0.025 * bootstrap_draws)]
    upper = boot[min(bootstrap_draws - 1, int(0.975 * bootstrap_draws))]
    hit = (
        (lower > 0.0 or upper < 0.0)
        and abs(mean) >= minimum_absolute_mean
    )
    return RecoveryPrediction(
        case_id=case_id,
        disposition=(
            RecoveryDisposition.DIRECT_RISK_CROSSFIT_PROJECTION.value
            if hit else RecoveryDisposition.UNRESOLVED.value
        ),
        direct_risk=hit,
        routed_for_exact_followup=not hit,
        safe_release=False,
        evidence_kind="CROSSFIT_FROZEN_PROJECTION",
        measurements={
            "evaluation_count": len(values),
            "mean_projection": mean,
            "bootstrap_interval": [lower, upper],
            "positive": sum(value > 0.0 for value in values),
            "negative": sum(value < 0.0 for value in values),
            "basis_frozen_before_evaluation": True,
            "minimum_absolute_mean": minimum_absolute_mean,
        },
        reason=(
            "a calibration-frozen direction confirms on untouched evaluation states"
            if hit else
            "the cross-fit projection mean is centered or unresolved"
        ),
    )


def missing_screen_prediction(
    case_id: str, *, missing: str,
) -> RecoveryPrediction:
    if missing == "EVENT_MOMENT":
        disposition = RecoveryDisposition.ESCALATE_MISSING_EVENT_MOMENT
        reason = "no label-blind schedule-conditioned event-moment screen is bound"
    elif missing == "TRANSPORT_JOINT_MOMENT":
        disposition = RecoveryDisposition.ESCALATE_MISSING_TRANSPORT_JOINT_MOMENT
        reason = "no label-blind residual/transport joint-moment screen is bound"
    else:
        raise ValueError("unknown missing screen capability")
    return RecoveryPrediction(
        case_id=case_id,
        disposition=disposition.value,
        direct_risk=False,
        routed_for_exact_followup=True,
        safe_release=False,
        evidence_kind="CAPABILITY_GATE",
        measurements={"missing_screen_input": missing},
        reason=reason,
    )


def compare_recovery(
    predictions: Sequence[RecoveryPrediction],
    frozen_targets: Mapping[str, str],
) -> dict[str, Any]:
    """Compare already-built predictions with development targets."""

    rows = []
    for prediction in predictions:
        target = frozen_targets.get(prediction.case_id)
        if target is None:
            raise ValueError(f"missing frozen target for {prediction.case_id}")
        if target == "STRICT_POSITIVE":
            recovered = prediction.direct_risk
            routed = prediction.direct_risk or prediction.routed_for_exact_followup
            false_safe = prediction.safe_release
        elif target == "PARTIAL_POSITIVE":
            recovered = prediction.direct_risk
            routed = prediction.direct_risk or prediction.routed_for_exact_followup
            false_safe = prediction.safe_release
        elif target == "ABSTAIN_BOUNDARY":
            recovered = (
                prediction.disposition
                == RecoveryDisposition.ABSTAIN_SOURCE_FIDELITY_FAILED.value
            )
            routed = not prediction.safe_release
            false_safe = prediction.safe_release
        else:
            raise ValueError(f"unknown frozen target {target}")
        rows.append({
            "case_id": prediction.case_id,
            "target": target,
            "prediction": prediction.disposition,
            "directly_recovered": recovered,
            "fail_closed_routed": routed,
            "false_safe": false_safe,
        })
    strict = [row for row in rows if row["target"] == "STRICT_POSITIVE"]
    return {
        "rows": rows,
        "strict_positive_count": len(strict),
        "strict_direct_recall": (
            sum(row["directly_recovered"] for row in strict) / len(strict)
            if strict else 0.0
        ),
        "strict_routed_recall": (
            sum(row["fail_closed_routed"] for row in strict) / len(strict)
            if strict else 0.0
        ),
        "false_safe_count": sum(row["false_safe"] for row in rows),
        "all_boundaries_preserved": all(row["fail_closed_routed"] for row in rows),
    }
