"""State-conditioned and trajectory-aware bias formation semantics.

BiasFormation v2.1 measured one deliberately narrow quantity: a directional
cross-state population statistic.  That statistic remains useful, but it is
not a necessary condition for a training trajectory to diverge.  This module
keeps the three claims separate:

* CONDITIONAL: a directional effect inside a predeclared comparable state
  condition.  A mixed population is never used to impute a conditional null.
* TRAJECTORY: a causally paired candidate/repair trajectory separates in the
  actual parameter state.  No fixed global carrier is required.
* GLOBAL: the old cross-state population result, valid only when a
  state-comparability certificate is present.

The module intentionally does not turn a single deterministic state contrast
into a statistical conditional bias.  Conditional claims require repeated
observations in each declared condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from .bias_formation_v21 import (
    FormationPolicy,
    FormationStatus,
    _certificate_from_gram,
    summarize_state_vectors,
)


class BiasLevel(str, Enum):
    CONDITIONAL = "CONDITIONAL"
    TRAJECTORY = "TRAJECTORY"
    GLOBAL = "GLOBAL"


class BiasV22Status(str, Enum):
    CONDITIONAL_BIAS = "CONDITIONAL_BIAS"
    CONDITIONAL_CENTERED = "CONDITIONAL_CENTERED"
    CONDITIONAL_UNRESOLVED = "CONDITIONAL_UNRESOLVED"
    TRAJECTORY_BIAS = "TRAJECTORY_BIAS"
    TRAJECTORY_EFFECT = "TRAJECTORY_EFFECT"
    TRAJECTORY_UNRESOLVED = "TRAJECTORY_UNRESOLVED"
    GLOBAL_BIAS = "GLOBAL_BIAS"
    GLOBAL_CENTERED = "GLOBAL_CENTERED"
    GLOBAL_NOT_IDENTIFIABLE = "GLOBAL_NOT_IDENTIFIABLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ConditionalPolicy:
    """Policy for within-condition population measurements."""

    min_replicates: int = 4
    bootstrap_samples: int = 2000
    centered_margin: float = 0.05
    bias_margin: float = 0.10
    canceling_margin: float = 0.10
    bootstrap_seed: int = 20260820

    def as_formation_policy(self) -> FormationPolicy:
        return FormationPolicy(
            min_states=self.min_replicates,
            centered_margin=self.centered_margin,
            bias_margin=self.bias_margin,
            canceling_margin=self.canceling_margin,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_replicates": self.min_replicates,
            "centered_margin": self.centered_margin,
            "bias_margin": self.bias_margin,
            "canceling_margin": self.canceling_margin,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "condition_statistic": "within_condition_off_diagonal_cross_state_u_statistic",
        }


def summarize_conditional_vectors(
    condition_vectors: Mapping[str, Sequence[Sequence[float]]],
    *,
    policy: ConditionalPolicy | None = None,
) -> dict[str, Any]:
    """Summarize each predeclared condition without pooling conditions.

    A condition with too few observations is unresolved.  The aggregate status
    is biased only if at least one condition is biased and no condition is
    unresolved; otherwise unresolved conditions remain explicit.  This is a
    conservative formation label, not a claim that all conditions share one
    direction.
    """

    policy = policy or ConditionalPolicy()
    if not condition_vectors:
        return {
            "schema": "kernel-analyzer-conditional-bias-certificate-v1",
            "level": BiasLevel.CONDITIONAL.value,
            "status": BiasV22Status.CONDITIONAL_UNRESOLVED.value,
            "reason": "NO_DECLARED_CONDITIONS",
            "conditions": {},
            "policy": policy.as_dict(),
        }
    conditions: dict[str, Any] = {}
    for condition_id, vectors in condition_vectors.items():
        cid = str(condition_id)
        rows = [tuple(float(x) for x in row) for row in vectors]
        if len(rows) < policy.min_replicates:
            conditions[cid] = {
                "status": BiasV22Status.CONDITIONAL_UNRESOLVED.value,
                "reason": "INSUFFICIENT_REPLICATES",
                "observation_count": len(rows),
            }
            continue
        try:
            cert = summarize_state_vectors(
                rows,
                state_ids=[f"{cid}:{i}" for i in range(len(rows))],
                layer="CONDITIONAL",
                partition=cid,
                policy=policy.as_formation_policy(),
            ).as_dict()
        except (TypeError, ValueError, FloatingPointError) as exc:
            conditions[cid] = {
                "status": BiasV22Status.INVALID.value,
                "reason": str(exc),
                "observation_count": len(rows),
            }
            continue
        old_status = cert["status"]
        if old_status == FormationStatus.BIASED.value:
            status = BiasV22Status.CONDITIONAL_BIAS.value
        elif old_status == FormationStatus.CENTERED.value:
            status = BiasV22Status.CONDITIONAL_CENTERED.value
        else:
            status = BiasV22Status.CONDITIONAL_UNRESOLVED.value
        cert["status"] = status
        cert["v21_status"] = old_status
        conditions[cid] = cert
    statuses = [value["status"] for value in conditions.values()]
    if any(status in {BiasV22Status.INVALID.value} for status in statuses):
        overall = BiasV22Status.INVALID.value
    elif any(status == BiasV22Status.CONDITIONAL_UNRESOLVED.value for status in statuses):
        overall = BiasV22Status.CONDITIONAL_UNRESOLVED.value
    elif any(status == BiasV22Status.CONDITIONAL_BIAS.value for status in statuses):
        overall = BiasV22Status.CONDITIONAL_BIAS.value
    else:
        overall = BiasV22Status.CONDITIONAL_CENTERED.value
    return {
        "schema": "kernel-analyzer-conditional-bias-certificate-v1",
        "level": BiasLevel.CONDITIONAL.value,
        "status": overall,
        "condition_count": len(conditions),
        "conditions": conditions,
        "policy": policy.as_dict(),
        "pooled_cross_condition_direction_not_tested": True,
    }


def summarize_conditional_gram(
    gram: Sequence[Sequence[float]],
    *,
    condition_id: str,
    coordinate_count: int,
    replicate_ids: Sequence[str],
    vector_digests: Sequence[str],
    estimand: str,
    reference: str,
    policy: ConditionalPolicy | None = None,
) -> dict[str, Any]:
    """Certify one fixed condition from a complete replicate Gram matrix.

    This is the disk-backed counterpart of :func:`summarize_conditional_vectors`.
    Replicates must differ only in the declared intervention randomness.  They
    are *not* unrelated training states, even though the underlying v2.1
    complete-Gram statistic is reused.

    ``estimand`` and ``reference`` are required because two scientifically
    different questions otherwise look identical numerically:

    * ``REPAIR_RESIDUAL`` compares a repaired value with an exact declared
      reference and can test whether the repair itself is conditionally
      centered.
    * ``CANDIDATE_MINUS_REPAIR_ENSEMBLE`` tests whether the repair removes a
      systematic candidate effect.  Without an exact downstream reference it
      cannot certify that the repaired gradient/update is itself unbiased.
    """

    policy = policy or ConditionalPolicy()
    cid = str(condition_id)
    ids = tuple(str(value) for value in replicate_ids)
    digests = tuple(str(value) for value in vector_digests)
    if estimand not in {
        "REPAIR_RESIDUAL",
        "CANDIDATE_MINUS_REPAIR_ENSEMBLE",
    }:
        raise ValueError("unknown conditional debiasing estimand")
    if not cid or not reference:
        raise ValueError("condition and reference must be declared")
    if len(ids) < policy.min_replicates:
        return {
            "schema": "kernel-analyzer-conditional-debias-certificate-v1",
            "level": BiasLevel.CONDITIONAL.value,
            "status": BiasV22Status.CONDITIONAL_UNRESOLVED.value,
            "reason": "INSUFFICIENT_REPLICATES",
            "condition_id": cid,
            "replicate_count": len(ids),
            "estimand": estimand,
            "reference": reference,
            "policy": policy.as_dict(),
        }
    certificate = _certificate_from_gram(
        gram,
        coordinate_count=coordinate_count,
        state_ids=ids,
        vector_digests=digests,
        layer=estimand,
        partition=cid,
        policy=policy.as_formation_policy(),
    ).as_dict()
    old_status = certificate["status"]
    if old_status == FormationStatus.BIASED.value:
        status = BiasV22Status.CONDITIONAL_BIAS.value
    elif old_status in {
        FormationStatus.CENTERED.value,
        FormationStatus.CANCELING_STRUCTURE.value,
    }:
        status = BiasV22Status.CONDITIONAL_CENTERED.value
    else:
        status = BiasV22Status.CONDITIONAL_UNRESOLVED.value
    identifies_repair_residual = estimand == "REPAIR_RESIDUAL"
    certificate.update({
        "schema": "kernel-analyzer-conditional-debias-certificate-v1",
        "level": BiasLevel.CONDITIONAL.value,
        "condition_id": cid,
        "replicate_count": len(ids),
        "replicates_are_intervention_randomness_not_training_states": True,
        "estimand": estimand,
        "reference": reference,
        "status": status,
        "v21_status": old_status,
        "identifies_repair_residual_bias": identifies_repair_residual,
        "identifies_candidate_effect_removed": not identifies_repair_residual,
        "certifies_downstream_repair_is_unbiased": (
            identifies_repair_residual and reference == "EXACT_DOWNSTREAM_REFERENCE"
        ),
        "policy": policy.as_dict(),
    })
    return certificate


def aggregate_conditional_debias(
    conditions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail-closed aggregate over separately certified fixed conditions.

    This function counts conditions; it never pools their vectors or requires
    one common parameter direction.  Every condition must expose the same
    roles.  A downstream candidate effect is kept separate from an absolute
    repair-residual claim.
    """

    if not conditions:
        return {
            "status": "CONDITIONAL_DEBIAS_UNRESOLVED",
            "reason": "NO_FIXED_CONDITIONS",
            "roles": {},
            "absolute_downstream_repair_bias": (
                "NOT_IDENTIFIABLE_MISSING_EXACT_REFERENCE"
            ),
            "global_direction_required": False,
        }
    role_sets = [set(layers) for layers in conditions.values()]
    if not role_sets[0] or any(roles != role_sets[0] for roles in role_sets[1:]):
        raise ValueError("fixed conditions do not expose the same debiasing roles")
    roles = {}
    for role in sorted(role_sets[0]):
        statuses = [str(layers[role].get("status", "")) for layers in conditions.values()]
        if any(not status for status in statuses):
            raise ValueError("conditional certificate is missing a status")
        roles[role] = {
            "condition_count": len(statuses),
            "status_counts": {
                status: statuses.count(status) for status in sorted(set(statuses))
            },
            "all_conditions_centered": all(
                status == BiasV22Status.CONDITIONAL_CENTERED.value
                for status in statuses
            ),
            "all_conditions_biased": all(
                status == BiasV22Status.CONDITIONAL_BIAS.value
                for status in statuses
            ),
        }
    required = {
        "repair_local_residual",
        "candidate_gradient_effect_removed",
    }
    if not required <= set(roles):
        raise ValueError("conditional debiasing roles are incomplete")
    local_centered = roles["repair_local_residual"]["all_conditions_centered"]
    downstream_effect = roles[
        "candidate_gradient_effect_removed"
    ]["all_conditions_biased"]
    return {
        "status": (
            "LOCAL_SOURCE_CONDITIONALLY_DEBIASED_WITH_SYSTEMATIC_CANDIDATE_F_B_EFFECT"
            if local_centered and downstream_effect
            else "CONDITIONAL_DEBIAS_UNRESOLVED"
        ),
        "condition_count": len(conditions),
        "roles": roles,
        "absolute_downstream_repair_bias": (
            "NOT_IDENTIFIABLE_MISSING_EXACT_REFERENCE"
        ),
        "global_direction_required": False,
    }


@dataclass(frozen=True)
class TrajectoryPolicy:
    """Fail-closed policy for basis-free paired trajectory separation."""

    min_steps: int = 8
    minimum_final_over_initial: float = 1.0
    recurrence_tolerance: float = 1e-6

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_steps": self.min_steps,
            "minimum_final_over_initial": self.minimum_final_over_initial,
            "recurrence_tolerance": self.recurrence_tolerance,
            "fixed_global_carrier_required": False,
        }


def certify_trajectory_separation(
    rows: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    policy: TrajectoryPolicy | None = None,
    drift_norm_key: str = "drift_norm",
    recurrence_residual_key: str | None = None,
) -> dict[str, Any]:
    """Certify a paired trajectory without a fixed cross-state direction.

    The input is a causal candidate/repair separation, not a population of
    unrelated states.  Existing runners may supply a declared live-weight
    separation norm.  A matched sham and a nonzero repair effect are required;
    the old ``directional_live_weight_accumulation`` gate is deliberately not
    consulted.
    """

    policy = policy or TrajectoryPolicy()
    required = (
        "repair_effect_present_every_step",
        "matched_sham_exact",
        "parameter_scope_closed",
    )
    missing_gates = [
        name
        for name in required
        if (
            gates.get(name) is not True
            and not (
                name == "parameter_scope_closed"
                and (
                    gates.get("only_declared_parameter_updated") is True
                    or gates.get("full_step_two_arm_scope_closed") is True
                )
            )
        )
    ]
    if len(rows) < policy.min_steps:
        return {
            "schema": "kernel-analyzer-trajectory-bias-certificate-v1",
            "level": BiasLevel.TRAJECTORY.value,
            "status": BiasV22Status.TRAJECTORY_UNRESOLVED.value,
            "reason": "INSUFFICIENT_STEPS",
            "step_count": len(rows),
            "policy": policy.as_dict(),
        }
    try:
        norms = [float(row[drift_norm_key]) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in norms):
            raise ValueError("drift norms must be finite and nonnegative")
        residuals = []
        if recurrence_residual_key is not None:
            residuals = [abs(float(row[recurrence_residual_key])) for row in rows]
            if any(not math.isfinite(value) for value in residuals):
                raise ValueError("recurrence residual is nonfinite")
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema": "kernel-analyzer-trajectory-bias-certificate-v1",
            "level": BiasLevel.TRAJECTORY.value,
            "status": BiasV22Status.INVALID.value,
            "reason": str(exc),
            "step_count": len(rows),
            "policy": policy.as_dict(),
        }
    if missing_gates:
        status = BiasV22Status.TRAJECTORY_UNRESOLVED.value
        reason = "MISSING_CAUSAL_OR_SHAM_GATE"
    elif residuals and max(residuals) > policy.recurrence_tolerance:
        status = BiasV22Status.TRAJECTORY_UNRESOLVED.value
        reason = "RECURRENCE_NOT_CLOSED"
    elif norms[-1] > norms[0] * policy.minimum_final_over_initial:
        status = BiasV22Status.TRAJECTORY_BIAS.value
        reason = "PAIRED_PARAMETER_SEPARATION_WITHOUT_GLOBAL_DIRECTION"
    elif max(norms) > norms[0]:
        status = BiasV22Status.TRAJECTORY_EFFECT.value
        reason = "NONMONOTONE_PAIRED_SEPARATION"
    else:
        status = BiasV22Status.TRAJECTORY_UNRESOLVED.value
        reason = "NO_SEPARATION_GROWTH"
    return {
        "schema": "kernel-analyzer-trajectory-bias-certificate-v1",
        "level": BiasLevel.TRAJECTORY.value,
        "status": status,
        "reason": reason,
        "step_count": len(rows),
        "initial_drift_norm": norms[0],
        "final_drift_norm": norms[-1],
        "max_drift_norm": max(norms),
        "path_length_in_norm_space": math.fsum(abs(b - a) for a, b in zip(norms, norms[1:])),
        "required_gates": list(required),
        "missing_gates": missing_gates,
        "recurrence_max_abs": max(residuals) if residuals else None,
        "fixed_global_carrier_required": False,
        "policy": policy.as_dict(),
    }


def classify_global_certificate(
    certificate: Mapping[str, Any],
    *,
    state_comparable: bool,
) -> dict[str, Any]:
    """Relabel v2.1 population output as a global-scope statement only."""

    if not state_comparable:
        status = BiasV22Status.GLOBAL_NOT_IDENTIFIABLE.value
    elif certificate.get("status") == FormationStatus.BIASED.value:
        status = BiasV22Status.GLOBAL_BIAS.value
    elif certificate.get("status") == FormationStatus.CENTERED.value:
        status = BiasV22Status.GLOBAL_CENTERED.value
    else:
        status = BiasV22Status.GLOBAL_NOT_IDENTIFIABLE.value
    return {
        "schema": "kernel-analyzer-global-bias-scope-v1",
        "level": BiasLevel.GLOBAL.value,
        "status": status,
        "state_comparable": bool(state_comparable),
        "v21_status": certificate.get("status"),
        "v21_cross_state_ratio": certificate.get("cross_state_ratio"),
        "claim_boundary": "This is not a necessary condition for trajectory bias.",
    }
