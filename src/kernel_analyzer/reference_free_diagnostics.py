"""Reference-free *evidence* checks for unfamiliar numerical kernels.

This module intentionally does not claim to infer a kernel's mathematical
meaning.  A raw Triton callable is not enough to decide whether its arithmetic
is correct: the intended function, dtype contract, masks, and saved-state
relationships are part of the specification.  When no complete reference is
available, callers can still provide three kinds of bounded evidence:

* controlled arithmetic variants (for example FP32 versus FP64 or two legal
  reduction orders),
* a declared input/intermediate consistency check, and
* a declared specification check for a property or expected value.

The result is deliberately ``bias_decision=NOT_ASSESSED``.  These checks help
select the next investigation; they are not a zero-configuration bias oracle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable, Optional

from .bias_checker import _clone_call, _flatten_tensors, _normalise_call, _snapshot_rng, _restore_rng


CheckCallback = Callable[[int, tuple[Any, ...], dict[str, Any], Any], Any]


def _json_value(value: Any) -> Any:
    """Convert simple callback diagnostics into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    return str(value)


def _callback_observation(value: Any) -> dict[str, Any]:
    """Normalize a callback result without inventing a truth value."""

    if isinstance(value, bool):
        return {"passed": value}
    if isinstance(value, Mapping):
        observation = {str(key): _json_value(item) for key, item in value.items()}
        if "passed" not in observation:
            if "ok" in observation:
                observation["passed"] = bool(observation["ok"])
            elif "status" in observation:
                observation["passed"] = str(observation["status"]).upper() in {
                    "PASS", "PASSED", "VALID", "OK"
                }
            else:
                raise TypeError("diagnostic callback mapping must contain passed, ok, or status")
        observation["passed"] = bool(observation["passed"])
        return observation
    raise TypeError("diagnostic callback must return bool or a mapping")


def _output_signature(value: Any) -> tuple[Any, ...]:
    leaves = _flatten_tensors(value)
    return tuple(
        (path, tuple(int(dim) for dim in tensor.shape), str(tensor.dtype))
        for path, tensor in leaves
    )


def _compare_outputs(candidate: Any, control: Any, *, torch: Any) -> dict[str, Any]:
    """Compare two output trees and return absolute, non-inferential metrics."""

    candidate_leaves = _flatten_tensors(candidate)
    control_leaves = _flatten_tensors(control)
    candidate_paths = [path for path, _ in candidate_leaves]
    control_paths = [path for path, _ in control_leaves]
    if not candidate_leaves or candidate_paths != control_paths:
        return {
            "status": "INVALID_OUTPUT_STRUCTURE",
            "candidate_signature": _output_signature(candidate),
            "control_signature": _output_signature(control),
        }

    all_effects = []
    all_controls = []
    for (_, candidate_tensor), (_, control_tensor) in zip(candidate_leaves, control_leaves):
        if candidate_tensor.shape != control_tensor.shape or candidate_tensor.numel() == 0:
            return {
                "status": "INVALID_OUTPUT_SHAPE",
                "candidate_signature": _output_signature(candidate),
                "control_signature": _output_signature(control),
            }
        if not candidate_tensor.is_floating_point() or not control_tensor.is_floating_point():
            return {"status": "INVALID_NON_FLOAT_OUTPUT"}
        candidate_float = candidate_tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        control_float = control_tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if not bool(torch.isfinite(candidate_float).all() and torch.isfinite(control_float).all()):
            return {"status": "INVALID_NONFINITE_OUTPUT"}
        all_effects.append(candidate_float - control_float)
        all_controls.append(control_float)

    effect = torch.cat(all_effects)
    baseline = torch.cat(all_controls)
    effect_rms = float(torch.sqrt(torch.mean(effect.square())).item())
    baseline_rms = float(torch.sqrt(torch.mean(baseline.square())).item())
    relative_rms = effect_rms / max(baseline_rms, 1e-30)
    return {
        "status": "VALID",
        "element_count": int(effect.numel()),
        "effect_rms": effect_rms,
        "control_rms": baseline_rms,
        "relative_rms": relative_rms,
        "signed_mean_delta": float(effect.mean().item()),
        "max_abs": float(effect.abs().max().item()),
        "nonzero_count": int(torch.count_nonzero(effect).item()),
    }


def _check_group_summary(
    observations: list[dict[str, Any]],
    *,
    declared: bool,
    check_name: str,
) -> dict[str, Any]:
    if not declared:
        return {
            "status": "NOT_DECLARED",
            "sample_count": 0,
            "check_name": check_name,
            "scope": "NO_CONTRACT_PROVIDED",
        }
    if not observations:
        return {
            "status": "NOT_MEASURED",
            "sample_count": 0,
            "check_name": check_name,
        }
    passed = sum(bool(item.get("passed", False)) for item in observations)
    failed = len(observations) - passed
    summary = {
        "status": "PASSED" if failed == 0 else "VIOLATION_OBSERVED",
        "sample_count": len(observations),
        "passed_count": passed,
        "failed_count": failed,
        "check_name": check_name,
        "observations": observations,
        "scope": "CALLER_DECLARED_CONTRACT_ONLY",
    }
    residuals = []
    for item in observations:
        value = item.get("residual")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            residuals.append(float(value))
    if residuals:
        summary["property_residual_count"] = len(residuals)
        summary["property_residual_mean"] = math.fsum(residuals) / len(residuals)
        summary["property_mean_inference"] = _conditional_mean_report(residuals)
    return summary


def _student_interval(values: list[float], *, alpha: float) -> dict[str, Any]:
    if len(values) < 2:
        return {"status": "NOT_IDENTIFIABLE_TOO_FEW_VALUES", "count": len(values), "interval": None}
    mean = math.fsum(values) / len(values)
    centered = [value - mean for value in values]
    variance = math.fsum(value * value for value in centered) / (len(values) - 1)
    standard_error = math.sqrt(max(0.0, variance) / len(values))
    if standard_error == 0.0:
        return {
            "status": "NOT_IDENTIFIABLE_ZERO_VARIANCE",
            "count": len(values),
            "mean": mean,
            "standard_error": 0.0,
            "interval": [mean, mean],
        }
    try:
        from .mean_inference import student_quantile

        critical = student_quantile(len(values) - 1, 1.0 - alpha / 2.0)
    except Exception as error:
        return {
            "status": "INTERVAL_UNAVAILABLE",
            "count": len(values),
            "mean": mean,
            "standard_error": standard_error,
            "interval": None,
            "reason": str(error),
        }
    half_width = critical * standard_error
    return {
        "status": "VALID",
        "count": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "degrees_of_freedom": len(values) - 1,
        "interval": [mean - half_width, mean + half_width],
        "assumptions": [
            "independent identically distributed draws are needed for a population interpretation",
            "Student interval is conditional and approximate unless its distributional assumptions hold",
        ],
    }


def _conditional_mean_report(values: list[float], *, alpha: float = 0.05) -> dict[str, Any]:
    """Report a conditional Student interval for a scalar declared residual.

    This is intentionally separate from ``bias_decision``: a residual has a
    population-zero interpretation only when its caller-declared property is
    valid.  Constant observations are not upgraded to a population theorem.
    """

    interval = _student_interval(values, alpha=alpha)
    if interval.get("status") != "VALID":
        return {
            **interval,
            "decision": "NOT_IDENTIFIABLE",
            "scope": "CONDITIONAL_DECLARED_PROPERTY_RESIDUAL",
        }
    low, high = interval["interval"]
    if low > 0.0 or high < 0.0:
        decision = "PROPERTY_MEAN_NONZERO_CONFIRMED"
    else:
        decision = "PROPERTY_MEAN_NOT_CONFIRMED"
    return {
        **interval,
        "decision": decision,
        "scope": "CONDITIONAL_DECLARED_PROPERTY_RESIDUAL",
        "assumptions": interval.get("assumptions", []) + [
            "the callback residual is the declared zero-valued property residual",
            "this endpoint does not establish a full output-vector bias",
        ],
    }


def diagnose_kernel(
    candidate: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    rounding_variants: Optional[Mapping[str, Callable[..., Any]]] = None,
    input_consistency_check: Optional[CheckCallback] = None,
    specification_check: Optional[CheckCallback] = None,
    samples: int = 8,
    preserve_rng: bool = True,
) -> dict[str, Any]:
    """Collect evidence for a kernel when no complete reference is available.

    ``rounding_variants`` must contain implementations that the caller has
    declared to have the same semantics while changing only an arithmetic
    choice.  Their output differences are reported as *controlled arithmetic
    evidence*, not as a bias verdict.

    The two callbacks receive ``(index, args, kwargs, candidate_output)``.
    They must return either a boolean or a mapping containing ``passed`` (or
    ``ok``/``status``).  They are the place to state relationships such as a
    saved probability summing to one, a recurrence state matching its input,
    or a candidate output satisfying a declared formula.

    The function imports PyTorch lazily, so the package remains importable in a
    CPU-only environment.  Failed samples are retained in the report and are
    never reclassified as negative evidence.
    """

    if not callable(candidate) or not callable(make_inputs):
        raise TypeError("candidate and make_inputs must be callable")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be an integer >= 1")
    variants = dict(rounding_variants or {})
    if any(not isinstance(name, str) or not name for name in variants):
        raise ValueError("rounding variant names must be non-empty strings")
    if any(not callable(control) for control in variants.values()):
        raise TypeError("rounding variants must be callable")

    try:
        import torch
    except Exception as error:  # pragma: no cover - optional dependency
        return {
            "schema": "kernel-analyzer-reference-free-diagnostics-v1",
            "measurement_status": "INVALID",
            "diagnostic_status": "UNAVAILABLE",
            "bias_decision": "NOT_ASSESSED",
            "reference_status": "NOT_PROVIDED",
            "errors": [{"where": "IMPORT", "error": str(error)}],
        }

    variant_records = {name: [] for name in variants}
    input_observations: list[dict[str, Any]] = []
    spec_observations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    candidate_signatures: list[Any] = []

    for index in range(samples):
        try:
            raw_args, raw_kwargs = _normalise_call(make_inputs(index))
            candidate_args, candidate_kwargs = _clone_call(
                raw_args, raw_kwargs, torch=torch, requires_grad=False
            )
            # Preserve the pre-call values for consistency checks.  This is
            # important for in-place kernels whose output mutates an input.
            check_args, check_kwargs = _clone_call(
                raw_args, raw_kwargs, torch=torch, requires_grad=False
            )
        except Exception as error:
            errors.append({"sample": index, "where": "INPUT", "error": str(error)})
            continue

        try:
            rng_state = _snapshot_rng(torch) if preserve_rng else None
            candidate_output = candidate(*candidate_args, **candidate_kwargs)
            if not _flatten_tensors(candidate_output):
                raise ValueError("candidate output contains no tensor leaves")
            for _, tensor in _flatten_tensors(candidate_output):
                if not bool(torch.isfinite(tensor.detach()).all()):
                    raise ValueError("candidate output contains non-finite values")
            candidate_signatures.append(_output_signature(candidate_output))
        except Exception as error:
            errors.append({"sample": index, "where": "CANDIDATE", "error": str(error)})
            continue

        if input_consistency_check is not None:
            try:
                observation = _callback_observation(
                    input_consistency_check(index, check_args, check_kwargs, candidate_output)
                )
                observation["sample"] = index
                input_observations.append(observation)
            except Exception as error:
                input_observations.append({
                    "passed": False,
                    "status": "CALLBACK_ERROR",
                    "sample": index,
                    "error": str(error),
                })
                errors.append({"sample": index, "where": "INPUT_CONSISTENCY", "error": str(error)})

        if specification_check is not None:
            try:
                observation = _callback_observation(
                    specification_check(index, check_args, check_kwargs, candidate_output)
                )
                observation["sample"] = index
                spec_observations.append(observation)
            except Exception as error:
                spec_observations.append({
                    "passed": False,
                    "status": "CALLBACK_ERROR",
                    "sample": index,
                    "error": str(error),
                })
                errors.append({"sample": index, "where": "SPECIFICATION", "error": str(error)})

        for name, control in variants.items():
            try:
                if preserve_rng:
                    _restore_rng(torch, rng_state)
                control_args, control_kwargs = _clone_call(
                    raw_args, raw_kwargs, torch=torch, requires_grad=False
                )
                control_output = control(*control_args, **control_kwargs)
                comparison = _compare_outputs(candidate_output, control_output, torch=torch)
                comparison["sample"] = index
                variant_records[name].append(comparison)
            except Exception as error:
                variant_records[name].append({
                    "sample": index,
                    "status": "ERROR",
                    "error": str(error),
                })

    variant_summary: dict[str, Any] = {}
    evidence_classes: list[str] = []
    for name, records in variant_records.items():
        valid = [record for record in records if record.get("status") == "VALID"]
        failed = [record for record in records if record.get("status") != "VALID"]
        nonzero = [record for record in valid if record.get("nonzero_count", 0) > 0]
        summary: dict[str, Any] = {
            "status": "NOT_MEASURED" if not records else ("VALID" if not failed else "PARTIAL"),
            "variant_name": name,
            "sample_count": len(records),
            "valid_count": len(valid),
            "failed_count": len(failed),
            "difference_observed_count": len(nonzero),
            "records": records,
            "comparison_scope": (
                "candidate_vs_caller_declared_same_semantics_arithmetic_variant"
            ),
            "interpretation": (
                "Difference is evidence that the declared arithmetic choice matters; "
                "it is not by itself rounding bias or a training claim."
            ),
        }
        if valid:
            signed_values = [float(record["signed_mean_delta"]) for record in valid]
            positive_count = sum(value > 0.0 for value in signed_values)
            negative_count = sum(value < 0.0 for value in signed_values)
            zero_count = len(signed_values) - positive_count - negative_count
            summary["mean_effect_rms"] = math.fsum(
                float(record["effect_rms"]) for record in valid
            ) / len(valid)
            summary["max_effect_rms"] = max(float(record["effect_rms"]) for record in valid)
            summary["max_relative_rms"] = max(float(record["relative_rms"]) for record in valid)
            summary["mean_signed_delta"] = math.fsum(signed_values) / len(signed_values)
            summary["signed_delta_positive_count"] = positive_count
            summary["signed_delta_negative_count"] = negative_count
            summary["signed_delta_zero_count"] = zero_count
            summary["signed_mean_inference"] = _conditional_mean_report(signed_values)
            summary["directional_evidence"] = (
                "CONSISTENT_SIGN_IN_DECLARED_CONTROL_COMPARISON"
                if (positive_count == len(signed_values) or negative_count == len(signed_values))
                and zero_count == 0
                else "MIXED_OR_ZERO_SIGN"
            )
        if nonzero:
            evidence_classes.append("ROUNDING_VARIANT_DIFFERENCE_OBSERVED")
            if valid and summary.get("directional_evidence") == "CONSISTENT_SIGN_IN_DECLARED_CONTROL_COMPARISON":
                evidence_classes.append("CONTROLLED_DIRECTIONAL_EVIDENCE")
        variant_summary[name] = summary

    input_summary = _check_group_summary(
        input_observations,
        declared=input_consistency_check is not None,
        check_name="input_or_intermediate_consistency",
    )
    spec_summary = _check_group_summary(
        spec_observations,
        declared=specification_check is not None,
        check_name="declared_specification",
    )
    if input_summary["status"] == "VIOLATION_OBSERVED":
        evidence_classes.append("INPUT_CONSISTENCY_VIOLATION")
    if spec_summary["status"] == "VIOLATION_OBSERVED":
        evidence_classes.append("DECLARED_SPEC_DEVIATION")

    measured = len(candidate_signatures)
    measurement_status = "VALID" if measured == samples and not errors else (
        "PARTIAL" if measured else "INVALID"
    )
    diagnostic_status = "EVIDENCE_ONLY" if measured else "UNRESOLVED_MEASUREMENT"
    return {
        "schema": "kernel-analyzer-reference-free-diagnostics-v1",
        "measurement_status": measurement_status,
        "diagnostic_status": diagnostic_status,
        "bias_decision": "NOT_ASSESSED",
        "reference_status": "NOT_PROVIDED",
        "sample_count": samples,
        "measured_sample_count": measured,
        "evidence_classes": sorted(set(evidence_classes)),
        "rounding_variants": variant_summary,
        "input_consistency": input_summary,
        "declared_specification": spec_summary,
        "errors": errors,
        "scope": "CALLER_DECLARED_INPUT_AND_CONTROL_CONTRACTS_ONLY",
        "limitations": [
            "No complete semantic reference was supplied or inferred.",
            "Controlled arithmetic variants must be independently declared to preserve semantics.",
            "A detected difference is not a proof of nonzero population mean bias.",
            "A passed property is not a proof of full functional equivalence.",
            "Training gradient, optimizer update, and loss consequences are not measured here.",
        ],
    }


__all__ = ["diagnose_kernel"]
