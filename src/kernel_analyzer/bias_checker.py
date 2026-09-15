"""Small, callable-first bias check for new numerical implementations.

This module deliberately sits below the training-analysis runners.  It does
not try to discover a reference implementation, inspect Triton source, or
infer a root cause.  Given two callable implementations and an input factory,
it performs paired evaluations and reports whether a reproducible signed
effect is visible in the declared input distribution.

The split-sample direction check is descriptive/conditional on the declared
input generator.  The exact sign-prevalence endpoint is valid only when the
confirmation inputs are independent draws and the calibration direction is
chosen without looking at them.  A positive result is never a claim about an
unseen training distribution.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional


def _flatten_tensors(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Return tensor leaves in a deterministic order without importing torch."""

    # ``torch.Tensor`` is intentionally detected by capability so importing
    # this module does not make torch a hard dependency of the package.
    if hasattr(value, "detach") and hasattr(value, "shape") and hasattr(value, "numel"):
        return [(path, value)]
    if isinstance(value, Mapping):
        leaves: list[tuple[tuple[str, ...], Any]] = []
        for key in sorted(value, key=lambda item: str(item)):
            leaves.extend(_flatten_tensors(value[key], path + ("{" + str(key) + "}",)))
        return leaves
    if isinstance(value, (tuple, list)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_flatten_tensors(item, path + ("[" + str(index) + "]",)))
        return leaves
    return []


def _clone_value(value: Any, *, torch: Any, requires_grad: bool) -> Any:
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        if cloned.is_floating_point() or cloned.is_complex():
            cloned.requires_grad_(bool(requires_grad or value.requires_grad))
        return cloned
    if isinstance(value, tuple):
        return tuple(_clone_value(item, torch=torch, requires_grad=requires_grad) for item in value)
    if isinstance(value, list):
        return [_clone_value(item, torch=torch, requires_grad=requires_grad) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _clone_value(item, torch=torch, requires_grad=requires_grad)
            for key, item in value.items()
        }
    try:
        return copy.deepcopy(value)
    except Exception:
        # Immutable arguments (integers, strings, lightweight descriptors) do
        # not need a deep copy.  If a mutable custom object cannot be copied,
        # the callable is responsible for not mutating it.
        return value


def _normalise_call(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(value, Mapping) and set(value).issubset({"args", "kwargs"}) and (
        "args" in value or "kwargs" in value
    ):
        args = value.get("args", ())
        kwargs = value.get("kwargs", {})
        if not isinstance(args, (tuple, list)) or not isinstance(kwargs, Mapping):
            raise TypeError("make_inputs mapping must contain tuple/list args and mapping kwargs")
        return tuple(args), dict(kwargs)
    if isinstance(value, tuple):
        return value, {}
    return (value,), {}


def _snapshot_rng(torch: Any) -> Any:
    cuda_states = None
    if bool(torch.cuda.is_available()):
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return torch.get_rng_state().clone(), cuda_states


def _restore_rng(torch: Any, state: Any) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    if cuda_states is not None and bool(torch.cuda.is_available()):
        torch.cuda.set_rng_state_all(cuda_states)


def _student_interval(values: Sequence[float], *, alpha: float) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) < 2:
        return {
            "status": "NOT_IDENTIFIABLE_TOO_FEW_VALUES",
            "count": len(values),
            "interval": None,
        }
    mean = math.fsum(values) / len(values)
    centered = [value - mean for value in values]
    variance = math.fsum(value * value for value in centered) / (len(values) - 1)
    standard_error = math.sqrt(max(0.0, variance) / len(values))
    try:
        from .mean_inference import student_quantile

        critical = student_quantile(len(values) - 1, 1.0 - alpha / 2.0)
    except Exception as error:  # scipy is an optional package extra
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
            "Student interval is an approximation unless its distributional assumptions hold",
        ],
    }


def _stage_report(
    effects: list[Any],
    references: list[Any],
    *,
    alpha: float,
    reference_energy_floor: float,
    allclose_results: list[bool],
    stage_name: str,
    torch: Any,
    calibration_count: Optional[int] = None,
) -> dict[str, Any]:
    import numpy as np

    effect_arrays = [effect.detach().to(device="cpu", dtype=torch.float64).reshape(-1) for effect in effects]
    reference_arrays = [reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1) for reference in references]
    x_values: list[float] = []
    b_values: list[float] = []
    a_values: list[float] = []
    aligned_values: list[float] = []
    for index, (effect, reference) in enumerate(zip(effect_arrays, reference_arrays)):
        x = float(torch.dot(effect, effect).item())
        b = float(torch.dot(reference, reference).item())
        a = float(torch.dot(effect, reference).item())
        x_values.append(x)
        b_values.append(b)
        a_values.append(a)
        if b > reference_energy_floor:
            aligned_values.append(a / b)

    total_effect_energy = math.fsum(x_values)
    total_reference_energy = math.fsum(b_values)
    total_rms = (
        math.sqrt(total_effect_energy / total_reference_energy)
        if total_reference_energy > 0.0
        else None
    )
    mean_effect = torch.stack(effect_arrays).mean(dim=0)
    mean_reference_energy = math.fsum(b_values) / len(b_values)
    mean_effect_ratio = (
        float(torch.linalg.vector_norm(mean_effect).item()) / math.sqrt(mean_reference_energy)
        if mean_reference_energy > 0.0
        else None
    )
    aligned_ratio_of_sums = (
        math.fsum(a_values) / total_reference_energy
        if total_reference_energy > 0.0
        else None
    )
    aligned_interval = _student_interval(aligned_values, alpha=alpha)

    direction: dict[str, Any]
    calibration_count = calibration_count if calibration_count is not None else len(effects) // 2
    confirmation_effects = effect_arrays[calibration_count:]
    calibration_effects = effect_arrays[:calibration_count]
    calibration_mean = torch.stack(calibration_effects).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(calibration_mean).item())
    if direction_norm <= 1e-30:
        direction = {
            "status": "NOT_IDENTIFIABLE_ZERO_CALIBRATION_MEAN",
            "calibration_count": calibration_count,
            "confirmation_count": len(confirmation_effects),
            "interval": None,
            "sign_prevalence": None,
        }
    else:
        unit_direction = calibration_mean / direction_norm
        projections = [float(torch.dot(effect, unit_direction).item()) for effect in confirmation_effects]
        projection_interval = _student_interval(projections, alpha=alpha)
        nonzero_projections = [value for value in projections if value != 0.0]
        try:
            from .population_direction import population_positive_direction_prevalence

            sign_prevalence = (
                population_positive_direction_prevalence(nonzero_projections, alpha=alpha)
                if nonzero_projections
                else {"decision": "NOT_IDENTIFIABLE_ZERO_EFFECTS"}
            )
            sign_prevalence["confirmation_zero_projection_count"] = len(projections) - len(nonzero_projections)
            sign_prevalence["sign_test_excludes_zero_projections"] = True
        except Exception as error:
            sign_prevalence = {"decision": "UNAVAILABLE", "reason": str(error)}
        direction = {
            "status": "VALID",
            "calibration_count": calibration_count,
            "confirmation_count": len(confirmation_effects),
            "calibration_direction_norm": direction_norm,
            "confirmation_projection_interval": projection_interval,
            "confirmation_projection_values": projections,
            "sign_prevalence": sign_prevalence,
            "selection": "direction learned only from calibration samples",
            "scope": "held_out_directional_reproducibility_under_make_inputs",
        }

    reasons: list[str] = []
    interval = direction.get("confirmation_projection_interval", {})
    if interval.get("status") == "VALID":
        lower, upper = interval["interval"]
        if lower > 0.0 or upper < 0.0:
            reasons.append("HELD_OUT_DIRECTIONAL_MEAN_NONZERO")
    sign = direction.get("sign_prevalence") or {}
    if sign.get("decision") in {
        "DIRECTION_PREVALENCE_CONFIRMED",
        "POSITIVE_DIRECTION_FREQUENCY_BELOW_NULL",
    }:
        reasons.append("HELD_OUT_DIRECTIONAL_SIGN_PREVALENCE")
    aligned = aligned_interval.get("interval")
    if aligned_interval.get("status") == "VALID" and aligned is not None:
        if aligned[0] > 0.0 or aligned[1] < 0.0:
            reasons.append("STATEWISE_ALIGNED_GAIN_NONZERO")

    if not allclose_results:
        allclose = None
    else:
        allclose = all(allclose_results)
    report: dict[str, Any] = {
        "stage": stage_name,
        "measurement_status": "VALID",
        "sample_count": len(effects),
        "allclose_auxiliary": {
            "allclose": allclose,
            "allclose_is_not_the_bias_decision": True,
        },
        "total_effect_energy": total_effect_energy,
        "total_reference_energy": total_reference_energy,
        "total_rms": total_rms,
        "mean_effect_norm_ratio": mean_effect_ratio,
        "aligned_ratio_of_sums": aligned_ratio_of_sums,
        "aligned_statewise_gain_interval": aligned_interval,
        "direction": direction,
        "systematic_bias_reasons": reasons,
        "decision": (
            "SYSTEMATIC_BIAS_CONFIRMED"
            if reasons
            else "SYSTEMATIC_BIAS_NOT_CONFIRMED"
        ),
        "scope": "DECLARED_INPUT_DISTRIBUTION_ONLY",
        "inference_note": (
            "Total RMS and aligned ratio are descriptive for this finite paired sample; "
            "the split-sample direction endpoint is conditional on the input generator."
        ),
    }
    # Keep numpy imported here only to ensure non-finite checks below use a
    # stable scalar conversion across CPU/GPU torch versions.
    if not np.isfinite(np.asarray(x_values)).all() or not np.isfinite(np.asarray(b_values)).all():
        report["measurement_status"] = "INVALID"
        report["decision"] = "UNRESOLVED_MEASUREMENT"
    return report


def check_bias(
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    samples: int = 32,
    calibration_samples: Optional[int] = None,
    alpha: float = 0.05,
    check_backward: bool = False,
    make_cotangent: Optional[Callable[..., Any]] = None,
    reference_energy_floor: float = 1e-20,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    preserve_rng: bool = True,
) -> dict[str, Any]:
    """Run a small paired bias check on callable implementations.

    ``make_inputs(index)`` may return positional arguments as a tuple, an
    explicit ``{"args": ..., "kwargs": ...}`` mapping, or one positional
    argument.  Each input is cloned before either implementation runs.

    The default check compares output tensors.  ``check_backward=True`` adds
    a gradient check with respect to all floating-point tensor inputs.  For a
    backward check the output must be one tensor; by default its cotangent is
    all ones, or ``make_cotangent(output, index)`` may provide one.

    The result is a JSON-compatible dictionary.  ``SYSTEMATIC_BIAS_CONFIRMED``
    means at least one held-out directional or aligned endpoint is nonzero;
    ``SYSTEMATIC_BIAS_NOT_CONFIRMED`` is not a proof of no bias.  Any failed
    execution, malformed output, non-finite value, or unidentifiable split
    returns ``UNRESOLVED_MEASUREMENT`` rather than a negative claim.
    """

    if not callable(candidate) or not callable(reference) or not callable(make_inputs):
        raise TypeError("candidate, reference, and make_inputs must be callable")
    if not isinstance(samples, int) or samples < 4:
        raise ValueError("samples must be an integer >= 4")
    if calibration_samples is None:
        calibration_samples = samples // 2
    if not isinstance(calibration_samples, int) or not 2 <= calibration_samples <= samples - 2:
        raise ValueError("calibration_samples must leave at least two confirmation samples")
    if not math.isfinite(alpha) or not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    if not math.isfinite(reference_energy_floor) or reference_energy_floor < 0.0:
        raise ValueError("reference_energy_floor must be finite and nonnegative")
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be finite and nonnegative")

    try:
        import torch
    except Exception as error:  # pragma: no cover - package can still import without torch
        return {
            "schema": "kernel-analyzer-bias-check-v1",
            "status": "UNRESOLVED_MEASUREMENT",
            "measurement_status": "INVALID",
            "scope": "DECLARED_INPUT_DISTRIBUTION_OUTPUT_BIAS",
            "errors": [{"stage": "IMPORT", "error": str(error)}],
        }

    stage_data: dict[str, dict[str, Any]] = {
        "OUTPUT": {"effects": [], "references": [], "allclose": [], "errors": []}
    }
    if check_backward:
        stage_data["BACKWARD"] = {"effects": [], "references": [], "allclose": [], "errors": []}

    for index in range(samples):
        try:
            raw_inputs = make_inputs(index)
            raw_args, raw_kwargs = _normalise_call(raw_inputs)
            candidate_args = _clone_value(raw_args, torch=torch, requires_grad=check_backward)
            candidate_kwargs = _clone_value(raw_kwargs, torch=torch, requires_grad=check_backward)
            reference_args = _clone_value(raw_args, torch=torch, requires_grad=check_backward)
            reference_kwargs = _clone_value(raw_kwargs, torch=torch, requires_grad=check_backward)
        except Exception as error:
            stage_data["OUTPUT"]["errors"].append({"sample": index, "where": "INPUT", "error": str(error)})
            continue

        rng_state = _snapshot_rng(torch) if preserve_rng else None
        try:
            candidate_output = candidate(*candidate_args, **candidate_kwargs)
            if preserve_rng:
                _restore_rng(torch, rng_state)
            reference_output = reference(*reference_args, **reference_kwargs)
        except Exception as error:
            stage_data["OUTPUT"]["errors"].append({"sample": index, "where": "FORWARD", "error": str(error)})
            continue

        candidate_leaves = _flatten_tensors(candidate_output)
        reference_leaves = _flatten_tensors(reference_output)
        candidate_paths = [path for path, _ in candidate_leaves]
        reference_paths = [path for path, _ in reference_leaves]
        if candidate_paths != reference_paths or not candidate_leaves:
            stage_data["OUTPUT"]["errors"].append({
                "sample": index,
                "where": "FORWARD_OUTPUT",
                "error": "candidate/reference tensor output structures do not match",
            })
            continue
        output_effect_parts = []
        output_reference_parts = []
        output_allclose = True
        output_valid = True
        for (_, candidate_tensor), (_, reference_tensor) in zip(candidate_leaves, reference_leaves):
            if candidate_tensor.shape != reference_tensor.shape or candidate_tensor.numel() == 0:
                output_valid = False
                break
            if not candidate_tensor.is_floating_point() or not reference_tensor.is_floating_point():
                output_valid = False
                break
            # Move only the small reporting view to CPU.  This also permits a
            # wrapper whose reference runs on a different device, provided
            # its outputs have matching structure and shape.
            candidate_float = candidate_tensor.detach().to(device="cpu", dtype=torch.float64)
            reference_float = reference_tensor.detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(candidate_float).all() and torch.isfinite(reference_float).all()):
                output_valid = False
                break
            output_effect_parts.append(candidate_float.reshape(-1) - reference_float.reshape(-1))
            output_reference_parts.append(reference_float.reshape(-1))
            output_allclose = output_allclose and bool(torch.allclose(
                candidate_float, reference_float, rtol=rtol, atol=atol
            ))
        if not output_valid:
            stage_data["OUTPUT"]["errors"].append({
                "sample": index,
                "where": "FORWARD_OUTPUT",
                "error": "non-floating, empty, non-finite, or shape-mismatched output",
            })
            continue
        stage_data["OUTPUT"]["effects"].append(torch.cat(output_effect_parts))
        stage_data["OUTPUT"]["references"].append(torch.cat(output_reference_parts))
        stage_data["OUTPUT"]["allclose"].append(output_allclose)

        if not check_backward:
            continue
        candidate_output_leaves = _flatten_tensors(candidate_output)
        reference_output_leaves = _flatten_tensors(reference_output)
        if len(candidate_output_leaves) != 1:
            stage_data["BACKWARD"]["errors"].append({
                "sample": index,
                "where": "BACKWARD",
                "error": "check_backward requires a single tensor output",
            })
            continue
        candidate_tensor = candidate_output_leaves[0][1]
        reference_tensor = reference_output_leaves[0][1]
        cotangent = (
            make_cotangent(reference_tensor.detach(), index)
            if make_cotangent is not None
            else torch.ones_like(reference_tensor.detach())
        )
        if not isinstance(cotangent, torch.Tensor) or cotangent.shape != reference_tensor.shape:
            stage_data["BACKWARD"]["errors"].append({
                "sample": index,
                "where": "BACKWARD",
                "error": "cotangent must be a tensor with the output shape",
            })
            continue
        candidate_cotangent = cotangent.to(device=candidate_tensor.device)
        reference_cotangent = cotangent.to(device=reference_tensor.device)
        candidate_inputs = [leaf for _, leaf in _flatten_tensors(candidate_args) if leaf.requires_grad and leaf.is_floating_point()]
        reference_inputs = [leaf for _, leaf in _flatten_tensors(reference_args) if leaf.requires_grad and leaf.is_floating_point()]
        if not candidate_inputs or not reference_inputs or len(candidate_inputs) != len(reference_inputs):
            stage_data["BACKWARD"]["errors"].append({
                "sample": index,
                "where": "BACKWARD",
                "error": "no matched differentiable tensor inputs",
            })
            continue
        try:
            if preserve_rng:
                backward_rng = _snapshot_rng(torch)
            candidate_grads = torch.autograd.grad(
                (candidate_tensor * candidate_cotangent).sum(), candidate_inputs,
                allow_unused=True,
            )
            if preserve_rng:
                _restore_rng(torch, backward_rng)
            reference_grads = torch.autograd.grad(
                (reference_tensor * reference_cotangent).sum(), reference_inputs,
                allow_unused=True,
            )
            candidate_parts = [
                (grad if grad is not None else torch.zeros_like(inp)).detach().to(dtype=torch.float64).reshape(-1)
                for grad, inp in zip(candidate_grads, candidate_inputs)
            ]
            reference_parts = [
                (grad if grad is not None else torch.zeros_like(inp)).detach().to(dtype=torch.float64).reshape(-1)
                for grad, inp in zip(reference_grads, reference_inputs)
            ]
            candidate_vector = torch.cat(candidate_parts).to(device="cpu")
            reference_vector = torch.cat(reference_parts).to(device="cpu")
            if not bool(torch.isfinite(candidate_vector).all() and torch.isfinite(reference_vector).all()):
                raise ValueError("non-finite gradient")
            stage_data["BACKWARD"]["effects"].append(candidate_vector - reference_vector)
            stage_data["BACKWARD"]["references"].append(reference_vector)
            stage_data["BACKWARD"]["allclose"].append(bool(torch.allclose(
                candidate_vector, reference_vector, rtol=rtol, atol=atol
            )))
        except Exception as error:
            stage_data["BACKWARD"]["errors"].append({"sample": index, "where": "BACKWARD", "error": str(error)})

    result: dict[str, Any] = {
        "schema": "kernel-analyzer-bias-check-v1",
        "status": "SYSTEMATIC_BIAS_NOT_CONFIRMED",
        "measurement_status": "VALID",
        "scope": "DECLARED_INPUT_DISTRIBUTION_OUTPUT_BIAS",
        "sample_count": samples,
        "calibration_samples": calibration_samples,
        "confirmation_samples": samples - calibration_samples,
        "alpha": alpha,
        "sample_assumption": (
            "make_inputs(index) must represent independent draws for population interpretations; "
            "the result otherwise remains a finite declared-input-sample description"
        ),
        "training_claim": False,
        "errors": [],
        "stages": {},
        "limitations": [
            "A not-confirmed result is not proof that bias is absent.",
            "This entry point does not infer a root cause or inspect Triton source.",
            "The output stage is the default; backward is optional and requires a single tensor output.",
            "Reference, call binding, and input distribution are supplied by the caller and are not validated semantically.",
        ],
    }
    for stage_name, data in stage_data.items():
        if len(data["effects"]) != samples:
            stage_result = {
                "stage": stage_name,
                "measurement_status": "PARTIAL" if data["effects"] else "INVALID",
                "sample_count": len(data["effects"]),
                "expected_sample_count": samples,
                "errors": data["errors"],
                "decision": "UNRESOLVED_MEASUREMENT",
            }
            result["stages"][stage_name] = stage_result
            result["errors"].extend({"stage": stage_name, **error} for error in data["errors"])
            result["measurement_status"] = "PARTIAL" if data["effects"] else "INVALID"
            result["status"] = "UNRESOLVED_MEASUREMENT"
            continue
        stage_result = _stage_report(
            data["effects"], data["references"], alpha=alpha,
            reference_energy_floor=reference_energy_floor,
            allclose_results=data["allclose"], stage_name=stage_name, torch=torch,
            calibration_count=calibration_samples,
        )
        stage_result["errors"] = data["errors"]
        result["stages"][stage_name] = stage_result
        if stage_result["decision"] == "SYSTEMATIC_BIAS_CONFIRMED":
            result["status"] = "SYSTEMATIC_BIAS_CONFIRMED"
    if check_backward:
        result["scope"] = "DECLARED_INPUT_DISTRIBUTION_OUTPUT_AND_BACKWARD_BIAS"
    if "OUTPUT" in result["stages"]:
        # Keep the explicit stage map while making the common one-stage case
        # convenient to consume as report["output"].
        result["output"] = result["stages"]["OUTPUT"]
    if "BACKWARD" in result["stages"]:
        result["backward"] = result["stages"]["BACKWARD"]
    return result


__all__ = ["check_bias"]
