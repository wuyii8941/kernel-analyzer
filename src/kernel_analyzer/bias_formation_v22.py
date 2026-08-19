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

from .bias_formation_v21 import FormationPolicy, FormationStatus, summarize_state_vectors


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
        "only_declared_parameter_updated",
    )
    missing_gates = [name for name in required if gates.get(name) is not True]
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
