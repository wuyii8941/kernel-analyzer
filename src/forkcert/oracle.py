"""Legacy finite-grid discrepancy profiles for DL implementation comparison.

Describes paired implementation discrepancies with three components:

  B: signed coordinate-average of the finite-grid mean effect (scalar observables only)
  H: state-conditioned effect heterogeneity after repeat-noise correction
  N: same-state paired-difference runtime variance under a declared coupling

These are implementation-relative estimands, not correctness or training-safety
claims. Zero mean can hide structured effects, and neither zero-mean noise nor a
nonzero mean has a universal long-run interpretation without dynamics assumptions.

This module is not the authoritative trajectory-level Bias Oracle.  In particular,
its input IDs do not establish a target state distribution, its normal intervals do
not supply independent-trajectory inference, and averaging vector coordinates is
not a semantic scalar endpoint.  The trajectory/phase record and confirmation path
under ``theory_oracle`` supplies those contracts.
"""

from __future__ import annotations

import math
import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class Verdict(Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INDETERMINATE = "INDETERMINATE"
    UNINSTANTIATED = "UNINSTANTIATED"
    INVALID = "INVALID"


@dataclass
class VerdictResult:
    verdict: Verdict
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceCriteria:
    """Pre-declared descriptive bounds, never universal safety/correctness limits."""
    max_relative_bias: float | None = None
    max_heterogeneity_cv: float | None = None
    max_runtime_cv: float | None = None
    max_step_param_bias: float | None = None
    max_step_loss_bias: float | None = None
    training_steps: int | None = None
    learning_rate: float | None = None

    def is_instantiated(self) -> bool:
        return any(v is not None for v in [
            self.max_relative_bias,
            self.max_heterogeneity_cv,
            self.max_runtime_cv,
            self.max_step_param_bias,
            self.max_step_loss_bias,
        ])

    def operator_is_instantiated(self) -> bool:
        return any(v is not None for v in [
            self.max_relative_bias,
            self.max_heterogeneity_cv,
            self.max_runtime_cv,
        ])

    def step_is_instantiated(self) -> bool:
        return any(v is not None for v in [
            self.max_step_param_bias,
            self.max_step_loss_bias,
        ])


# ---------------------------------------------------------------------------
# Raw measurement
# ---------------------------------------------------------------------------

@dataclass
class OperatorMeasurement:
    """Raw paired observations for one operator on one input."""
    input_id: int
    repeat_id: int
    ref_output: np.ndarray
    cand_output: np.ndarray

    @property
    def diff(self) -> np.ndarray:
        return self.cand_output - self.ref_output


# ---------------------------------------------------------------------------
# Statistical profile (Layer 2)
# ---------------------------------------------------------------------------

@dataclass
class OperatorProfile:
    """Finite-grid B/H/N description; population meaning requires an external design."""
    name: str
    bias: float
    bias_std_err: float
    relative_bias: float
    heterogeneity: float
    heterogeneity_cv: float
    runtime_var: float
    runtime_cv: float
    output_scale: float
    n_inputs: int
    n_repeats: int
    per_input_bias: np.ndarray | None = None
    element_bias: np.ndarray | None = None
    bias_norm: float = 0.0
    relative_bias_norm: float = 0.0
    relative_bias_lower: float = 0.0
    relative_bias_upper: float = 0.0
    heterogeneity_lower: float = 0.0
    heterogeneity_upper: float = 0.0
    runtime_var_lower: float = 0.0
    runtime_var_upper: float = 0.0
    ref_runtime_var: float = 0.0
    cand_runtime_var: float = 0.0
    runtime_identified: bool = False
    balanced_design: bool = True
    output_shape: tuple[int, ...] = ()
    uncertainty_method: str = "state-cluster normal approximation"


@dataclass
class StepProfile:
    """B/H/N at the training-step level."""
    loss_bias: float
    loss_bias_relative: float
    loss_heterogeneity: float
    loss_runtime_var: float
    grad_bias_norm: float
    grad_bias_relative: float
    grad_heterogeneity: float
    grad_runtime_var: float
    param_update_bias_norm: float
    param_update_bias_relative: float
    n_inputs: int
    n_repeats: int
    operator_profiles: dict[str, OperatorProfile] = field(default_factory=dict)
    loss_bias_relative_lower: float = 0.0
    loss_bias_relative_upper: float = 0.0
    param_update_bias_relative_lower: float = 0.0
    param_update_bias_relative_upper: float = 0.0
    param_update_heterogeneity: float = 0.0
    param_update_runtime_var: float = 0.0
    balanced_design: bool = True


# ---------------------------------------------------------------------------
# Measurement collector
# ---------------------------------------------------------------------------

def collect_operator_measurements(
    ref_fn: Callable[[Any], np.ndarray],
    cand_fn: Callable[[Any], np.ndarray],
    inputs: list[Any],
    n_repeats: int = 1,
) -> list[OperatorMeasurement]:
    measurements = []
    for i, x in enumerate(inputs):
        for r in range(n_repeats):
            ref_out = np.asarray(ref_fn(x), dtype=np.float64)
            cand_out = np.asarray(cand_fn(x), dtype=np.float64)
            measurements.append(OperatorMeasurement(
                input_id=i, repeat_id=r,
                ref_output=ref_out, cand_output=cand_out,
            ))
    return measurements


# ---------------------------------------------------------------------------
# Profile computation (Layer 2)
# ---------------------------------------------------------------------------

def compute_operator_profile(
    name: str,
    measurements: list[OperatorMeasurement],
) -> OperatorProfile:
    if not measurements:
        raise ValueError("no measurements")

    input_ids = sorted(set(m.input_id for m in measurements))
    repeat_ids = sorted(set(m.repeat_id for m in measurements))
    n_inputs = len(input_ids)
    n_repeats = len(repeat_ids)

    by_input: dict[int, list[OperatorMeasurement]] = {}
    for m in measurements:
        by_input.setdefault(m.input_id, []).append(m)

    expected_pairs = {(iid, rid) for iid in input_ids for rid in repeat_ids}
    observed_pairs = [(m.input_id, m.repeat_id) for m in measurements]
    balanced_design = len(observed_pairs) == len(set(observed_pairs)) and set(observed_pairs) == expected_pairs
    if not balanced_design:
        raise ValueError("measurements must form one balanced input x repeat grid")

    shapes = {tuple(np.asarray(m.ref_output).shape) for m in measurements}
    shapes.update(tuple(np.asarray(m.cand_output).shape) for m in measurements)
    if len(shapes) != 1:
        raise ValueError(f"reference/candidate output shapes differ: {sorted(shapes)}")
    output_shape = next(iter(shapes))

    per_input_mean_diff: list[np.ndarray] = []
    per_input_var_diff: list[np.ndarray] = []
    per_input_var_ref: list[np.ndarray] = []
    per_input_var_cand: list[np.ndarray] = []
    ref_scales: list[float] = []

    for iid in input_ids:
        group = by_input[iid]
        group = sorted(group, key=lambda m: m.repeat_id)
        refs = np.stack([m.ref_output.ravel() for m in group], axis=0)
        cands = np.stack([m.cand_output.ravel() for m in group], axis=0)
        diffs = cands - refs
        mean_d = diffs.mean(axis=0)
        per_input_mean_diff.append(mean_d)
        if n_repeats > 1:
            var_d = diffs.var(axis=0, ddof=1)
            var_ref = refs.var(axis=0, ddof=1)
            var_cand = cands.var(axis=0, ddof=1)
        else:
            var_d = np.zeros_like(mean_d)
            var_ref = np.zeros_like(mean_d)
            var_cand = np.zeros_like(mean_d)
        per_input_var_diff.append(var_d)
        per_input_var_ref.append(var_ref)
        per_input_var_cand.append(var_cand)
        ref_scale = np.mean([np.abs(m.ref_output.ravel()).mean() for m in group])
        ref_scales.append(float(ref_scale))

    mean_diff_stack = np.stack(per_input_mean_diff, axis=0)

    element_bias = mean_diff_stack.mean(axis=0)
    B = float(element_bias.mean())  # signed scalar summary, retained for scalar observables/reporting
    bias_norm = float(np.sqrt(np.mean(np.square(element_bias))))

    per_input_scalar_bias = np.array([float(m.mean()) for m in per_input_mean_diff])

    output_scale = float(np.mean(ref_scales)) if ref_scales else 1.0
    output_scale = max(output_scale, 1e-30)

    relative_bias = B / output_scale
    relative_bias_norm = bias_norm / output_scale

    H_observed_per_element = mean_diff_stack.var(axis=0, ddof=1) if n_inputs > 1 else np.zeros_like(element_bias)

    N_per_element = np.stack(per_input_var_diff, axis=0).mean(axis=0)
    N = float(N_per_element.mean())
    H = max(0.0, float(H_observed_per_element.mean()) - N / max(n_repeats, 1))
    ref_N = float(np.stack(per_input_var_ref, axis=0).mean())
    cand_N = float(np.stack(per_input_var_cand, axis=0).mean())

    bias_std_err = float(per_input_scalar_bias.std(ddof=1) / math.sqrt(n_inputs)) if n_inputs > 1 else math.inf
    vector_se = (
        float(np.sqrt(np.mean(H_observed_per_element) / n_inputs))
        if n_inputs > 1 else math.inf
    )
    z = 1.96
    relative_bias_lower = max(0.0, bias_norm - z * vector_se) / output_scale
    relative_bias_upper = (bias_norm + z * vector_se) / output_scale

    state_h_contrib = np.mean(np.square(mean_diff_stack - element_bias), axis=1)
    if n_inputs > 1:
        h_se = float(state_h_contrib.std(ddof=1) / math.sqrt(n_inputs))
    else:
        h_se = math.inf
    h_lower = max(0.0, H - z * h_se)
    h_upper = H + z * h_se if math.isfinite(h_se) else math.inf

    state_n_contrib = np.array([float(v.mean()) for v in per_input_var_diff])
    if n_repeats > 1 and n_inputs > 1:
        n_se = float(state_n_contrib.std(ddof=1) / math.sqrt(n_inputs))
        n_lower = max(0.0, N - z * n_se)
        n_upper = N + z * n_se
    elif n_repeats > 1:
        n_lower, n_upper = 0.0, math.inf
    else:
        n_lower, n_upper = 0.0, math.inf

    h_cv = math.sqrt(H) / output_scale if output_scale > 0 else 0.0
    n_cv = math.sqrt(N) / output_scale if output_scale > 0 else 0.0

    return OperatorProfile(
        name=name,
        bias=B,
        bias_std_err=bias_std_err,
        relative_bias=relative_bias,
        heterogeneity=H,
        heterogeneity_cv=h_cv,
        runtime_var=N,
        runtime_cv=n_cv,
        output_scale=output_scale,
        n_inputs=n_inputs,
        n_repeats=n_repeats,
        per_input_bias=per_input_scalar_bias,
        element_bias=element_bias,
        bias_norm=bias_norm,
        relative_bias_norm=relative_bias_norm,
        relative_bias_lower=relative_bias_lower,
        relative_bias_upper=relative_bias_upper,
        heterogeneity_lower=h_lower,
        heterogeneity_upper=h_upper,
        runtime_var_lower=n_lower,
        runtime_var_upper=n_upper,
        ref_runtime_var=ref_N,
        cand_runtime_var=cand_N,
        runtime_identified=n_repeats > 1,
        balanced_design=balanced_design,
        output_shape=output_shape,
    )


# ---------------------------------------------------------------------------
# Step-level profile (Layer 3)
# ---------------------------------------------------------------------------

def compute_step_profile(
    step_measurements: list[dict[str, Any]],
    operator_profiles: dict[str, OperatorProfile] | None = None,
) -> StepProfile:
    """Compute step-level B/H/N from paired training step observations.

    Each entry in step_measurements is a dict with keys:
      input_id, repeat_id, ref_loss, cand_loss,
      ref_grad (flat np.ndarray), cand_grad (flat np.ndarray),
      ref_param_update (flat np.ndarray), cand_param_update (flat np.ndarray)
    """
    if not step_measurements:
        raise ValueError("no step measurements")
    input_ids = sorted(set(s["input_id"] for s in step_measurements))
    repeat_ids = sorted(set(s["repeat_id"] for s in step_measurements))
    n_inputs = len(input_ids)
    n_repeats = len(repeat_ids)
    observed_pairs = [(s["input_id"], s["repeat_id"]) for s in step_measurements]
    expected_pairs = {(iid, rid) for iid in input_ids for rid in repeat_ids}
    balanced = len(observed_pairs) == len(set(observed_pairs)) and set(observed_pairs) == expected_pairs
    if not balanced:
        raise ValueError("step measurements must form one balanced input x repeat grid")

    by_input: dict[int, list[dict]] = {}
    for s in step_measurements:
        by_input.setdefault(s["input_id"], []).append(s)

    per_input_loss_bias: list[float] = []
    per_input_loss_var: list[float] = []
    per_input_grad_bias: list[np.ndarray] = []
    per_input_grad_var: list[np.ndarray] = []
    per_input_update_bias: list[np.ndarray] = []
    per_input_update_var: list[np.ndarray] = []
    ref_loss_vals: list[float] = []
    ref_grad_norms: list[float] = []
    ref_update_norms: list[float] = []

    for iid in input_ids:
        group = by_input[iid]
        loss_diffs = [s["cand_loss"] - s["ref_loss"] for s in group]
        per_input_loss_bias.append(float(np.mean(loss_diffs)))
        per_input_loss_var.append(float(np.var(loss_diffs, ddof=1)) if len(loss_diffs) > 1 else 0.0)
        ref_loss_vals.append(float(np.mean([abs(s["ref_loss"]) for s in group])))

        grad_diffs = np.stack([s["cand_grad"] - s["ref_grad"] for s in group], axis=0)
        per_input_grad_bias.append(grad_diffs.mean(axis=0))
        per_input_grad_var.append(grad_diffs.var(axis=0, ddof=1) if len(group) > 1 else np.zeros(grad_diffs.shape[1]))
        ref_grad_norms.append(float(np.mean([np.linalg.norm(s["ref_grad"]) for s in group])))

        update_diffs = np.stack([s["cand_param_update"] - s["ref_param_update"] for s in group], axis=0)
        per_input_update_bias.append(update_diffs.mean(axis=0))
        per_input_update_var.append(
            update_diffs.var(axis=0, ddof=1) if len(group) > 1 else np.zeros(update_diffs.shape[1])
        )
        ref_update_norms.append(float(np.mean([np.linalg.norm(s["ref_param_update"]) for s in group])))

    loss_bias = float(np.mean(per_input_loss_bias))
    loss_scale = float(np.mean(ref_loss_vals)) if ref_loss_vals else 1.0
    loss_scale = max(loss_scale, 1e-30)

    grad_bias_vec = np.mean(per_input_grad_bias, axis=0)
    grad_bias_norm = float(np.linalg.norm(grad_bias_vec))
    grad_scale = float(np.mean(ref_grad_norms)) if ref_grad_norms else 1.0
    grad_scale = max(grad_scale, 1e-30)

    update_bias_vec = np.mean(per_input_update_bias, axis=0)
    update_bias_norm = float(np.linalg.norm(update_bias_vec))
    update_scale = float(np.mean(ref_update_norms)) if ref_update_norms else 1.0
    update_scale = max(update_scale, 1e-30)

    loss_H = float(np.var(per_input_loss_bias, ddof=1)) if n_inputs > 1 else 0.0
    loss_N = float(np.mean(per_input_loss_var))

    grad_H_per_element = np.var(np.stack(per_input_grad_bias), axis=0, ddof=1) if n_inputs > 1 else np.zeros_like(grad_bias_vec)
    grad_H = float(grad_H_per_element.mean())
    grad_N = float(np.mean([v.mean() for v in per_input_grad_var]))
    update_H_observed = (
        float(np.var(np.stack(per_input_update_bias), axis=0, ddof=1).mean())
        if n_inputs > 1 else 0.0
    )
    update_N = float(np.mean([v.mean() for v in per_input_update_var]))
    update_H = max(0.0, update_H_observed - update_N / max(n_repeats, 1))

    z = 1.96
    loss_se = (
        float(np.std(per_input_loss_bias, ddof=1) / math.sqrt(n_inputs))
        if n_inputs > 1 else math.inf
    )
    loss_rel_lower = max(0.0, abs(loss_bias) - z * loss_se) / loss_scale
    loss_rel_upper = (abs(loss_bias) + z * loss_se) / loss_scale
    update_se = (
        math.sqrt(update_H_observed / n_inputs) if n_inputs > 1 else math.inf
    )
    update_rel_lower = max(0.0, update_bias_norm - z * update_se) / update_scale
    update_rel_upper = (update_bias_norm + z * update_se) / update_scale

    return StepProfile(
        loss_bias=loss_bias,
        loss_bias_relative=loss_bias / loss_scale,
        loss_heterogeneity=loss_H,
        loss_runtime_var=loss_N,
        grad_bias_norm=grad_bias_norm,
        grad_bias_relative=grad_bias_norm / grad_scale,
        grad_heterogeneity=grad_H,
        grad_runtime_var=grad_N,
        param_update_bias_norm=update_bias_norm,
        param_update_bias_relative=update_bias_norm / update_scale,
        n_inputs=n_inputs,
        n_repeats=n_repeats,
        operator_profiles=operator_profiles or {},
        loss_bias_relative_lower=loss_rel_lower,
        loss_bias_relative_upper=loss_rel_upper,
        param_update_bias_relative_lower=update_rel_lower,
        param_update_bias_relative_upper=update_rel_upper,
        param_update_heterogeneity=update_H,
        param_update_runtime_var=update_N,
        balanced_design=balanced,
    )


# ---------------------------------------------------------------------------
# Verdict (Layer 4)
# ---------------------------------------------------------------------------

def judge_operator(
    profile: OperatorProfile,
    criteria: AcceptanceCriteria,
) -> VerdictResult:
    if not criteria.operator_is_instantiated():
        return VerdictResult(
            verdict=Verdict.UNINSTANTIATED,
            reason="no operator-level acceptance criteria declared",
        )

    if profile.n_inputs < 2:
        return VerdictResult(
            verdict=Verdict.INVALID,
            reason=f"need >= 2 inputs for statistical decomposition, got {profile.n_inputs}",
        )
    if criteria.max_heterogeneity_cv is not None and not profile.runtime_identified:
        return VerdictResult(
            verdict=Verdict.INVALID,
            reason="repeat-noise-corrected heterogeneity requires at least two same-state repeats",
        )

    violations = []
    indeterminate = []
    checks = {}

    if criteria.max_relative_bias is not None:
        estimate = profile.relative_bias_norm
        lower, upper = profile.relative_bias_lower, profile.relative_bias_upper
        checks["relative_bias_norm"] = {
            "value": estimate, "lower": lower, "upper": upper,
            "limit": criteria.max_relative_bias,
        }
        if lower > criteria.max_relative_bias:
            violations.append(
                f"relative_bias_lower={lower:.2e} > {criteria.max_relative_bias:.2e}"
            )
            checks["relative_bias_norm"]["status"] = "violates"
        elif upper <= criteria.max_relative_bias:
            checks["relative_bias_norm"]["status"] = "within"
        else:
            indeterminate.append("relative_bias_norm interval overlaps limit")
            checks["relative_bias_norm"]["status"] = "indeterminate"

    if criteria.max_heterogeneity_cv is not None:
        lower = math.sqrt(profile.heterogeneity_lower) / profile.output_scale
        upper = math.sqrt(profile.heterogeneity_upper) / profile.output_scale
        val = profile.heterogeneity_cv
        checks["heterogeneity_cv"] = {
            "value": val, "lower": lower, "upper": upper,
            "limit": criteria.max_heterogeneity_cv,
        }
        if lower > criteria.max_heterogeneity_cv:
            violations.append(f"heterogeneity_cv_lower={lower:.2e} > {criteria.max_heterogeneity_cv:.2e}")
            checks["heterogeneity_cv"]["status"] = "violates"
        elif upper <= criteria.max_heterogeneity_cv:
            checks["heterogeneity_cv"]["status"] = "within"
        else:
            indeterminate.append("heterogeneity_cv interval overlaps limit")
            checks["heterogeneity_cv"]["status"] = "indeterminate"

    if criteria.max_runtime_cv is not None:
        if not profile.runtime_identified:
            return VerdictResult(
                verdict=Verdict.INVALID,
                reason="runtime variance criterion requires at least two same-state repeats",
            )
        lower = math.sqrt(profile.runtime_var_lower) / profile.output_scale
        upper = math.sqrt(profile.runtime_var_upper) / profile.output_scale
        val = profile.runtime_cv
        checks["runtime_cv"] = {
            "value": val, "lower": lower, "upper": upper,
            "limit": criteria.max_runtime_cv,
        }
        if lower > criteria.max_runtime_cv:
            violations.append(f"runtime_cv_lower={lower:.2e} > {criteria.max_runtime_cv:.2e}")
            checks["runtime_cv"]["status"] = "violates"
        elif upper <= criteria.max_runtime_cv:
            checks["runtime_cv"]["status"] = "within"
        else:
            indeterminate.append("runtime_cv interval overlaps limit")
            checks["runtime_cv"]["status"] = "indeterminate"

    if violations:
        return VerdictResult(
            verdict=Verdict.REJECT,
            reason="; ".join(violations),
            details=checks,
        )

    if indeterminate:
        return VerdictResult(
            verdict=Verdict.INDETERMINATE,
            reason="; ".join(indeterminate),
            details=checks,
        )

    return VerdictResult(
        verdict=Verdict.ACCEPT,
        reason="all declared operator criteria satisfied; unchecked components and conditional/semantic effects are not accepted",
        details={
            **checks,
            "verdict_scope": "DECLARED_FINITE_GRID_CRITERIA_ONLY",
            "conditional_effects_tested": False,
            "semantic_effects_tested": False,
            "unchecked_components": [
                name for name, value in (
                    ("B", criteria.max_relative_bias),
                    ("H", criteria.max_heterogeneity_cv),
                    ("N", criteria.max_runtime_cv),
                ) if value is None
            ],
        },
    )


def judge_step(
    profile: StepProfile,
    criteria: AcceptanceCriteria,
) -> VerdictResult:
    if not criteria.step_is_instantiated():
        return VerdictResult(
            verdict=Verdict.UNINSTANTIATED,
            reason="no step-level acceptance criteria declared",
        )

    violations = []
    checks = {}

    if criteria.max_step_loss_bias is not None:
        val = abs(profile.loss_bias_relative)
        lower, upper = profile.loss_bias_relative_lower, profile.loss_bias_relative_upper
        checks["step_loss_bias"] = {"value": val, "lower": lower, "upper": upper, "limit": criteria.max_step_loss_bias}
        if lower > criteria.max_step_loss_bias:
            violations.append(f"step_loss_bias_lower={lower:.2e} > {criteria.max_step_loss_bias:.2e}")
            checks["step_loss_bias"]["status"] = "violates"
        elif upper <= criteria.max_step_loss_bias:
            checks["step_loss_bias"]["status"] = "within"
        else:
            checks["step_loss_bias"]["status"] = "indeterminate"

    if criteria.max_step_param_bias is not None:
        val = profile.param_update_bias_relative
        lower, upper = profile.param_update_bias_relative_lower, profile.param_update_bias_relative_upper
        checks["step_param_bias"] = {"value": val, "lower": lower, "upper": upper, "limit": criteria.max_step_param_bias}
        if lower > criteria.max_step_param_bias:
            violations.append(f"param_update_bias_lower={lower:.2e} > {criteria.max_step_param_bias:.2e}")
            checks["step_param_bias"]["status"] = "violates"
        elif upper <= criteria.max_step_param_bias:
            checks["step_param_bias"]["status"] = "within"
        else:
            checks["step_param_bias"]["status"] = "indeterminate"

    if violations:
        return VerdictResult(
            verdict=Verdict.REJECT,
            reason="; ".join(violations),
            details=checks,
        )

    if any(value.get("status") == "indeterminate" for value in checks.values()):
        return VerdictResult(
            verdict=Verdict.INDETERMINATE,
            reason="step-effect confidence interval overlaps an acceptance limit",
            details=checks,
        )

    return VerdictResult(
        verdict=Verdict.ACCEPT,
        reason="step-level criteria satisfied",
        details=checks,
    )


# ---------------------------------------------------------------------------
# High-level Oracle
# ---------------------------------------------------------------------------

class Oracle:
    """Compute a matched-state B/H/N profile for declared acceptance criteria.

    ACCEPT is scoped only to criteria actually supplied. It is not a universal
    substitution-safety or correctness verdict.

    Usage:
        oracle = Oracle(AcceptanceCriteria(max_relative_bias=1e-5, ...))
        measurements = oracle.measure(ref_fn, cand_fn, inputs, n_repeats=3)
        profile = oracle.profile("my_op", measurements)
        verdict = oracle.judge(profile)
    """

    def __init__(self, criteria: AcceptanceCriteria):
        self.criteria = criteria

    def measure(
        self,
        ref_fn: Callable,
        cand_fn: Callable,
        inputs: list,
        n_repeats: int = 1,
    ) -> list[OperatorMeasurement]:
        return collect_operator_measurements(ref_fn, cand_fn, inputs, n_repeats)

    def profile(
        self,
        name: str,
        measurements: list[OperatorMeasurement],
    ) -> OperatorProfile:
        return compute_operator_profile(name, measurements)

    def judge(self, profile: OperatorProfile) -> VerdictResult:
        return judge_operator(profile, self.criteria)

    def measure_and_judge(
        self,
        name: str,
        ref_fn: Callable,
        cand_fn: Callable,
        inputs: list,
        n_repeats: int = 1,
    ) -> tuple[OperatorProfile, VerdictResult]:
        m = self.measure(ref_fn, cand_fn, inputs, n_repeats)
        p = self.profile(name, m)
        v = self.judge(p)
        return p, v


# ---------------------------------------------------------------------------
# Torch integration helpers
# ---------------------------------------------------------------------------

def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class TorchOperatorCapture:
    """Capture operator outputs from a torch model during forward pass.

    Hooks into named modules and records their outputs for ref/cand comparison.
    """

    def __init__(self, model, target_modules: list[str] | None = None):
        self.torch = _try_import_torch()
        if self.torch is None:
            raise RuntimeError("torch required")
        self.model = model
        self.target_modules = target_modules
        self._captures: dict[str, list] = {}
        self._hooks: list = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            t = output
            if isinstance(t, tuple):
                t = t[0]
            if self.torch.is_tensor(t):
                self._captures.setdefault(name, []).append(
                    t.detach().float().cpu().numpy().copy()
                )
        return hook

    def attach(self):
        for name, module in self.model.named_modules():
            if self.target_modules is None or name in self.target_modules:
                if name:
                    h = module.register_forward_hook(self._make_hook(name))
                    self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self._captures.clear()

    @property
    def captures(self) -> dict[str, list[np.ndarray]]:
        return dict(self._captures)


def profile_torch_model(
    ref_model,
    cand_model,
    inputs: list,
    target_modules: list[str] | None = None,
    n_repeats: int = 1,
    criteria: AcceptanceCriteria | None = None,
) -> dict[str, tuple[OperatorProfile, VerdictResult | None]]:
    """Profile all target modules comparing ref_model vs cand_model.

    Returns a dict mapping module name to (profile, verdict_or_none).
    """
    torch = _try_import_torch()
    if torch is None:
        raise RuntimeError("torch required")

    ref_cap = TorchOperatorCapture(ref_model, target_modules)
    cand_cap = TorchOperatorCapture(cand_model, target_modules)

    ref_cap.attach()
    cand_cap.attach()

    ref_cap.reset()
    cand_cap.reset()
    try:
        for r in range(n_repeats):
            for i, x in enumerate(inputs):
                with torch.no_grad():
                    if isinstance(x, dict):
                        ref_model(**x)
                        cand_model(**x)
                    else:
                        ref_model(x)
                        cand_model(x)
    finally:
        ref_cap.detach()
        cand_cap.detach()

    all_names = sorted(set(ref_cap.captures.keys()) & set(cand_cap.captures.keys()))

    results = {}
    for name in all_names:
        ref_outs = ref_cap.captures[name]
        cand_outs = cand_cap.captures[name]
        expected = n_repeats * len(inputs)
        if len(ref_outs) != expected or len(cand_outs) != expected:
            raise ValueError(
                f"module {name!r} must execute exactly once per forward; "
                f"expected {expected}, got ref={len(ref_outs)}, cand={len(cand_outs)}"
            )
        n = expected
        measurements = []
        for idx in range(n):
            input_id = idx % len(inputs)
            repeat_id = idx // len(inputs)
            measurements.append(OperatorMeasurement(
                input_id=input_id,
                repeat_id=repeat_id,
                ref_output=ref_outs[idx],
                cand_output=cand_outs[idx],
            ))

        if len(set(m.input_id for m in measurements)) < 2:
            continue

        profile = compute_operator_profile(name, measurements)
        verdict = judge_operator(profile, criteria) if criteria else None
        results[name] = (profile, verdict)

    return results


def profile_torch_training_step(
    ref_model,
    cand_model,
    ref_optimizer,
    cand_optimizer,
    inputs: list,
    loss_fn,
    n_repeats: int = 1,
    criteria: AcceptanceCriteria | None = None,
    device: str = "cpu",
) -> tuple[StepProfile, VerdictResult | None]:
    """Profile a matched model/loss/optimizer step without scheduler or AMP.

    Model and optimizer states plus Torch RNG are restored before every arm and
    repeat. This helper is not a complete natural-training transition when scaler,
    scheduler, data-loader or other state is outcome-relevant.
    """
    torch = _try_import_torch()
    if torch is None:
        raise RuntimeError("torch required")

    step_measurements = []
    ref_initial_model = copy.deepcopy(ref_model.state_dict())
    cand_initial_model = copy.deepcopy(cand_model.state_dict())
    ref_initial_optimizer = copy.deepcopy(ref_optimizer.state_dict())
    cand_initial_optimizer = copy.deepcopy(cand_optimizer.state_dict())

    ref_initial_params = [p.detach().cpu().clone() for p in ref_model.parameters()]
    cand_initial_params = [p.detach().cpu().clone() for p in cand_model.parameters()]
    if len(ref_initial_params) != len(cand_initial_params) or any(
        a.shape != b.shape or not torch.equal(a, b)
        for a, b in zip(ref_initial_params, cand_initial_params, strict=True)
    ):
        raise ValueError("reference and candidate must start from identical parameter tensors")

    for r in range(n_repeats):
        for i, x in enumerate(inputs):
            ref_model.load_state_dict(ref_initial_model)
            cand_model.load_state_dict(cand_initial_model)
            ref_optimizer.load_state_dict(copy.deepcopy(ref_initial_optimizer))
            cand_optimizer.load_state_dict(copy.deepcopy(cand_initial_optimizer))
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
            ref_before_tensors = [p.detach().float().cpu().clone() for p in ref_model.parameters()]
            cand_before_tensors = [p.detach().float().cpu().clone() for p in cand_model.parameters()]

            ref_model.zero_grad()
            if isinstance(x, dict):
                ref_out = ref_model(**x)
            else:
                ref_out = ref_model(x)
            ref_loss = loss_fn(ref_out)
            ref_loss.backward()
            ref_grad = torch.cat([p.grad.detach().float().cpu().ravel() for p in ref_model.parameters() if p.grad is not None]).numpy()
            ref_optimizer.step()
            ref_after = torch.cat([p.detach().float().cpu().ravel() for p in ref_model.parameters()]).numpy()
            ref_before = torch.cat([v.ravel() for v in ref_before_tensors]).numpy()
            ref_update = ref_after - ref_before

            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)

            cand_model.zero_grad()
            if isinstance(x, dict):
                cand_out = cand_model(**x)
            else:
                cand_out = cand_model(x)
            cand_loss = loss_fn(cand_out)
            cand_loss.backward()
            cand_grad = torch.cat([p.grad.detach().float().cpu().ravel() for p in cand_model.parameters() if p.grad is not None]).numpy()
            cand_optimizer.step()
            cand_after = torch.cat([p.detach().float().cpu().ravel() for p in cand_model.parameters()]).numpy()
            cand_before = torch.cat([v.ravel() for v in cand_before_tensors]).numpy()
            cand_update = cand_after - cand_before

            step_measurements.append({
                "input_id": i,
                "repeat_id": r,
                "ref_loss": float(ref_loss.item()),
                "cand_loss": float(cand_loss.item()),
                "ref_grad": ref_grad,
                "cand_grad": cand_grad,
                "ref_param_update": ref_update,
                "cand_param_update": cand_update,
            })

    ref_model.load_state_dict(ref_initial_model)
    cand_model.load_state_dict(cand_initial_model)
    ref_optimizer.load_state_dict(ref_initial_optimizer)
    cand_optimizer.load_state_dict(cand_initial_optimizer)

    profile = compute_step_profile(step_measurements)
    verdict = judge_step(profile, criteria) if criteria else None
    return profile, verdict


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_operator_report(profile: OperatorProfile, verdict: VerdictResult | None = None) -> str:
    lines = [
        f"=== Operator: {profile.name} ===",
        f"  Inputs: {profile.n_inputs}, Repeats: {profile.n_repeats}",
        f"  Output scale: {profile.output_scale:.4e}",
        f"  Signed mean summary: {profile.bias:.4e}",
        f"  B norm: {profile.bias_norm:.4e}  (relative: {profile.relative_bias_norm:.4e}, "
        f"95% approx [{profile.relative_bias_lower:.4e}, {profile.relative_bias_upper:.4e}])",
        f"  Heterogeneity (H, repeat-corrected): {profile.heterogeneity:.4e}  (CV: {profile.heterogeneity_cv:.4e})",
        f"  Runtime var (N, paired difference): {profile.runtime_var:.4e}  (CV: {profile.runtime_cv:.4e})",
        f"  Runtime var ref/cand: {profile.ref_runtime_var:.4e} / {profile.cand_runtime_var:.4e}",
    ]
    if verdict:
        lines.append(f"  Verdict: {verdict.verdict.value} — {verdict.reason}")
    return "\n".join(lines)


def format_step_report(profile: StepProfile, verdict: VerdictResult | None = None) -> str:
    lines = [
        "=== Training Step Profile ===",
        f"  Inputs: {profile.n_inputs}, Repeats: {profile.n_repeats}",
        f"  Loss bias: {profile.loss_bias:.4e}  (relative: {profile.loss_bias_relative:.4e})",
        f"  Loss heterogeneity: {profile.loss_heterogeneity:.4e}",
        f"  Loss runtime var: {profile.loss_runtime_var:.4e}",
        f"  Gradient bias norm: {profile.grad_bias_norm:.4e}  (relative: {profile.grad_bias_relative:.4e})",
        f"  Gradient heterogeneity: {profile.grad_heterogeneity:.4e}",
        f"  Gradient runtime var: {profile.grad_runtime_var:.4e}",
        f"  Param update bias norm: {profile.param_update_bias_norm:.4e}  (relative: {profile.param_update_bias_relative:.4e})",
    ]
    if verdict:
        lines.append(f"  Verdict: {verdict.verdict.value} — {verdict.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training Oracle — multi-step, multi-module training monitoring
# ---------------------------------------------------------------------------

@dataclass
class _OnlineStats:
    """Welford online mean/variance for scalars."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std_err(self) -> float:
        return math.sqrt(self.variance / self.n) if self.n > 0 else 0.0


@dataclass
class ModuleTimeSeries:
    """Per-module temporal summaries on freely evolving twin trajectories."""
    name: str
    bias_stats: _OnlineStats = field(default_factory=_OnlineStats)
    scale_stats: _OnlineStats = field(default_factory=_OnlineStats)
    per_step_bias: list[float] = field(default_factory=list)
    per_step_scale: list[float] = field(default_factory=list)

    @property
    def bias(self) -> float:
        return self.bias_stats.mean

    @property
    def heterogeneity(self) -> float:
        return self.bias_stats.variance

    @property
    def output_scale(self) -> float:
        return max(self.scale_stats.mean, 1e-30)

    @property
    def relative_bias(self) -> float:
        return self.bias / self.output_scale

    @property
    def bias_std_err(self) -> float:
        return self.bias_stats.std_err

    @property
    def n_steps(self) -> int:
        return self.bias_stats.n


@dataclass
class TrainingSnapshot:
    """State captured at one training step."""
    step: int
    ref_loss: float
    cand_loss: float
    loss_diff: float
    param_divergence: float
    param_divergence_relative: float
    module_diffs: dict[str, float]
    verdict: Verdict | None = None


class TwinTrajectoryMonitor:
    """Describes two freely evolving training trajectories.

    Hook differences after the first update combine current implementation effects
    with accumulated parameter-state differences. Consequently temporal means and
    variances from this class are not matched-state B/H/N and cannot establish
    operator-substitution safety. Use :class:`TrainingOracle` for matched states.

    Usage:
        monitor = TwinTrajectoryMonitor(ref_model, cand_model, criteria,
                                target_modules=["layer1", "layer2"])

        for step, (x, y) in enumerate(dataloader):
            monitor.begin_step()
            # ... run ref forward/backward/step, cand forward/backward/step ...
            ref_loss = ...
            cand_loss = ...
            snapshot = monitor.end_step(step, ref_loss, cand_loss,
                                       ref_model, cand_model)
            if snapshot.verdict == Verdict.REJECT:
                print(f"Training divergence detected at step {step}")
                break

        report = monitor.report()
    """

    def __init__(
        self,
        ref_model,
        cand_model,
        criteria: AcceptanceCriteria,
        target_modules: list[str] | None = None,
    ):
        self.torch = _try_import_torch()
        if self.torch is None:
            raise RuntimeError("torch required")

        self.ref_model = ref_model
        self.cand_model = cand_model
        self.criteria = criteria
        self.target_modules = target_modules

        self._ref_captures: dict[str, list[np.ndarray]] = {}
        self._cand_captures: dict[str, list[np.ndarray]] = {}
        self._ref_hooks: list = []
        self._cand_hooks: list = []

        self.module_series: dict[str, ModuleTimeSeries] = {}
        self.loss_stats = _OnlineStats()
        self.snapshots: list[TrainingSnapshot] = []

        self._initial_params: np.ndarray | None = None

        self._install_hooks(ref_model, self._ref_captures, self._ref_hooks)
        self._install_hooks(cand_model, self._cand_captures, self._cand_hooks)

        self._capture_initial_params()

    def _install_hooks(self, model, captures: dict, hooks: list):
        for name, module in model.named_modules():
            if not name:
                continue
            if self.target_modules is not None and name not in self.target_modules:
                continue

            def make_hook(n):
                def hook(mod, inp, out):
                    t = out
                    if isinstance(t, tuple):
                        t = t[0]
                    if self.torch.is_tensor(t):
                        captures.setdefault(n, []).append(
                            t.detach().float().cpu().numpy().copy()
                        )
                return hook

            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    def _capture_initial_params(self):
        self._initial_params = self.torch.cat([
            p.detach().float().cpu().ravel()
            for p in self.ref_model.parameters()
        ]).numpy().copy()

    def begin_step(self):
        self._ref_captures.clear()
        self._cand_captures.clear()

    def end_step(
        self,
        step: int,
        ref_loss: float,
        cand_loss: float,
        ref_model=None,
        cand_model=None,
    ) -> TrainingSnapshot:
        ref_model = ref_model or self.ref_model
        cand_model = cand_model or self.cand_model

        loss_diff = cand_loss - ref_loss
        self.loss_stats.update(loss_diff)

        module_diffs = {}
        common_names = sorted(
            set(self._ref_captures.keys()) & set(self._cand_captures.keys())
        )

        for name in common_names:
            ref_outs = self._ref_captures[name]
            cand_outs = self._cand_captures[name]
            if not ref_outs or not cand_outs:
                continue

            ref_arr = ref_outs[-1]
            cand_arr = cand_outs[-1]

            diff_mean = float(np.mean(cand_arr - ref_arr))
            ref_scale = float(np.abs(ref_arr).mean())

            if name not in self.module_series:
                self.module_series[name] = ModuleTimeSeries(name=name)

            series = self.module_series[name]
            series.bias_stats.update(diff_mean)
            series.scale_stats.update(ref_scale)
            series.per_step_bias.append(diff_mean)
            series.per_step_scale.append(ref_scale)
            module_diffs[name] = diff_mean

        ref_params = self.torch.cat([
            p.detach().float().cpu().ravel() for p in ref_model.parameters()
        ]).numpy()
        cand_params = self.torch.cat([
            p.detach().float().cpu().ravel() for p in cand_model.parameters()
        ]).numpy()

        param_div = float(np.linalg.norm(cand_params - ref_params))
        param_scale = float(np.linalg.norm(self._initial_params))
        param_div_rel = param_div / max(param_scale, 1e-30)

        verdict = self._check_criteria(step)

        snapshot = TrainingSnapshot(
            step=step,
            ref_loss=ref_loss,
            cand_loss=cand_loss,
            loss_diff=loss_diff,
            param_divergence=param_div,
            param_divergence_relative=param_div_rel,
            module_diffs=module_diffs,
            verdict=verdict,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def _check_criteria(self, step: int) -> Verdict | None:
        if not self.criteria.is_instantiated():
            return Verdict.UNINSTANTIATED
        return Verdict.INDETERMINATE

    def detach(self):
        for h in self._ref_hooks:
            h.remove()
        for h in self._cand_hooks:
            h.remove()
        self._ref_hooks.clear()
        self._cand_hooks.clear()

    def report(self) -> str:
        lines = ["=" * 70, "  Twin Trajectory Monitor (not matched-state Oracle)", "=" * 70, ""]

        n = len(self.snapshots)
        if n == 0:
            lines.append("No training steps recorded.")
            return "\n".join(lines)

        lines.append(f"Total steps: {n}")
        lines.append(f"Loss diff — mean: {self.loss_stats.mean:.4e}, "
                      f"var: {self.loss_stats.variance:.4e}")

        last = self.snapshots[-1]
        lines.append(f"Final param divergence: {last.param_divergence:.4e} "
                      f"(relative: {last.param_divergence_relative:.4e})")

        if n >= 10:
            early_divs = [s.param_divergence for s in self.snapshots[:n // 3]]
            late_divs = [s.param_divergence for s in self.snapshots[2 * n // 3:]]
            if len(early_divs) >= 2 and len(late_divs) >= 2:
                early_rate = (early_divs[-1] - early_divs[0]) / max(len(early_divs), 1)
                late_rate = (late_divs[-1] - late_divs[0]) / max(len(late_divs), 1)
                if early_rate > 0:
                    accel = late_rate / early_rate
                    growth = "accelerating" if accel > 1.5 else "linear" if accel > 0.5 else "decelerating"
                    lines.append(f"Divergence growth: {growth} (late/early rate ratio: {accel:.2f})")

        lines.append("")
        lines.append("Per-module temporal differences (not matched-state B/H/N):")
        lines.append(f"  {'Module':<30s} {'Mean(rel)':>12s} {'TimeVar':>12s} {'StdErr':>12s} {'Steps':>6s} {'Verdict':>12s}")
        lines.append("  " + "-" * 86)

        for name in sorted(self.module_series.keys()):
            s = self.module_series[name]
            rel_b = s.relative_bias
            h = s.heterogeneity
            se = s.bias_std_err

            v_str = ""
            if self.criteria.max_relative_bias is not None and s.n_steps >= 5:
                sig = abs(s.bias) / max(se, 1e-30)
                if abs(rel_b) > self.criteria.max_relative_bias and sig > 2.0:
                    v_str = "REJECT"
                elif abs(rel_b) > self.criteria.max_relative_bias:
                    v_str = "INDETERMINATE"
                else:
                    v_str = "ACCEPT"

            lines.append(f"  {name:<30s} {rel_b:>12.4e} {h:>12.4e} {se:>12.4e} {s.n_steps:>6d} {v_str:>12s}")

        lines.append("")

        overall = self._check_criteria(n - 1)
        lines.append(f"Overall verdict: {overall.value if overall else 'PENDING'}")

        return "\n".join(lines)


class TrainingOracle:
    """Aggregate B/H/N across predeclared matched training states.

    This class does not run two free trajectories. Callers provide paired repeats
    collected from the same complete pre-step state. State sampling, reference,
    candidate, randomness and coupling protocols are named at construction time.
    """

    def __init__(
        self,
        criteria: AcceptanceCriteria,
        *,
        query_id: str,
        state_distribution: str,
        randomness_protocol: str,
        coupling_protocol: str,
    ) -> None:
        if not all((query_id, state_distribution, randomness_protocol, coupling_protocol)):
            raise ValueError("matched TrainingOracle requires a named query, state distribution and RNG/coupling protocols")
        self.criteria = criteria
        self.query_id = query_id
        self.state_distribution = state_distribution
        self.randomness_protocol = randomness_protocol
        self.coupling_protocol = coupling_protocol
        self._state_ids: dict[str, int] = {}
        self._operator_measurements: dict[str, list[OperatorMeasurement]] = {}
        self._operator_state_keys: set[tuple[str, str]] = set()
        self._operator_states: dict[str, set[str]] = {}
        self._step_measurements: list[dict[str, Any]] = []
        self._step_state_keys: set[str] = set()

    def _input_id(self, state_id: str) -> int:
        if state_id not in self._state_ids:
            self._state_ids[state_id] = len(self._state_ids)
        return self._state_ids[state_id]

    def record_operator_state(
        self,
        name: str,
        state_id: str,
        ref_repeats: list[np.ndarray],
        cand_repeats: list[np.ndarray],
    ) -> None:
        key = (name, state_id)
        if key in self._operator_state_keys:
            raise ValueError(f"duplicate operator/state record: {key}")
        if not ref_repeats or len(ref_repeats) != len(cand_repeats):
            raise ValueError("reference and candidate require the same positive repeat count")
        input_id = self._input_id(state_id)
        rows = self._operator_measurements.setdefault(name, [])
        for repeat_id, (ref, cand) in enumerate(zip(ref_repeats, cand_repeats, strict=True)):
            rows.append(OperatorMeasurement(
                input_id=input_id,
                repeat_id=repeat_id,
                ref_output=np.asarray(ref, dtype=np.float64),
                cand_output=np.asarray(cand, dtype=np.float64),
            ))
        self._operator_state_keys.add(key)
        self._operator_states.setdefault(name, set()).add(state_id)

    def record_step_state(self, state_id: str, paired_repeats: list[dict[str, Any]]) -> None:
        if state_id in self._step_state_keys:
            raise ValueError(f"duplicate step state: {state_id}")
        if not paired_repeats:
            raise ValueError("step state requires at least one paired repeat")
        input_id = self._input_id(state_id)
        for repeat_id, row in enumerate(paired_repeats):
            required = {
                "ref_loss", "cand_loss", "ref_grad", "cand_grad",
                "ref_param_update", "cand_param_update",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"step repeat missing fields: {missing}")
            self._step_measurements.append({
                "input_id": input_id,
                "repeat_id": repeat_id,
                "ref_loss": float(row["ref_loss"]),
                "cand_loss": float(row["cand_loss"]),
                "ref_grad": np.asarray(row["ref_grad"], dtype=np.float64).ravel(),
                "cand_grad": np.asarray(row["cand_grad"], dtype=np.float64).ravel(),
                "ref_param_update": np.asarray(row["ref_param_update"], dtype=np.float64).ravel(),
                "cand_param_update": np.asarray(row["cand_param_update"], dtype=np.float64).ravel(),
            })
        self._step_state_keys.add(state_id)

    def operator_profiles(self) -> dict[str, OperatorProfile]:
        state_sets = list(self._operator_states.values())
        if state_sets and any(states != state_sets[0] for states in state_sets[1:]):
            coverage = {name: sorted(states) for name, states in self._operator_states.items()}
            raise ValueError(f"operators do not cover the same matched state bank: {coverage}")
        return {
            name: compute_operator_profile(name, rows)
            for name, rows in sorted(self._operator_measurements.items())
        }

    def operator_verdicts(self) -> dict[str, VerdictResult]:
        return {
            name: judge_operator(profile, self.criteria)
            for name, profile in self.operator_profiles().items()
        }

    def step_profile(self) -> StepProfile:
        if not self._step_measurements:
            raise ValueError("no matched step states recorded")
        if self._operator_states:
            operator_states = next(iter(self._operator_states.values()))
            if self._step_state_keys != operator_states:
                raise ValueError(
                    "step and operator profiles must cover the same matched state bank: "
                    f"operator={sorted(operator_states)}, step={sorted(self._step_state_keys)}"
                )
        return compute_step_profile(self._step_measurements, self.operator_profiles())

    def step_verdict(self) -> VerdictResult:
        return judge_step(self.step_profile(), self.criteria)

    @property
    def n_states(self) -> int:
        return len(self._state_ids)

    def summary(self) -> dict[str, Any]:
        profiles = self.operator_profiles()
        verdicts = {
            name: result.verdict.value for name, result in self.operator_verdicts().items()
        }
        return {
            "profile_kind": "LEGACY_FINITE_GRID_DISCREPANCY_DESCRIPTION",
            "query_id": self.query_id,
            "state_distribution": self.state_distribution,
            "randomness_protocol": self.randomness_protocol,
            "coupling_protocol": self.coupling_protocol,
            "n_states": self.n_states,
            "operators": {
                name: {
                    "B_norm": profile.bias_norm,
                    "B_relative": profile.relative_bias_norm,
                    "H": profile.heterogeneity,
                    "N_paired_difference": profile.runtime_var,
                    "sampling_interval_B_relative": [
                        profile.relative_bias_lower, profile.relative_bias_upper,
                    ],
                    "verdict": verdicts[name],
                }
                for name, profile in profiles.items()
            },
            "step": (
                {
                    "profile": asdict(self.step_profile()),
                    "verdict": self.step_verdict().verdict.value,
                }
                if self._step_measurements else None
            ),
            "nonclaims": [
                "no correctness claim without an independent authority",
                "no long-run accumulation claim from B alone",
                "no operator causality from observation alone",
                "an ACCEPT verdict covers only explicitly declared finite-grid criteria",
                "global-mean acceptance does not accept H, boundary-conditioned effects, or semantic disagreement",
            ],
        }
