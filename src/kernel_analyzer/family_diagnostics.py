"""Small declarative diagnostics for the first supported numerical families.

The functions in this module are deliberately callable-first.  They do not
inspect Triton IR or invent a reference implementation.  A family contract is
declared once here; a new implementation supplies a callable and an input
generator, while :func:`diagnose_kernel` handles execution, failures and the
conditional scalar inference for a declared zero-valued property.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Optional

from .bias_checker import (
    _clone_call,
    _flatten_tensors,
    _normalise_call,
    _restore_rng,
    _snapshot_rng,
)
from .reference_free_diagnostics import diagnose_kernel


def _select_output(value: Any, selector: Any) -> Any:
    if selector is None:
        selected = value
    elif callable(selector):
        selected = selector(value)
    elif isinstance(selector, int):
        if not isinstance(value, (tuple, list)):
            raise TypeError("output_index requires a tuple/list output")
        selected = value[selector]
    elif isinstance(selector, str):
        if not isinstance(value, Mapping):
            raise TypeError("output_key requires a mapping output")
        selected = value[selector]
    else:
        raise TypeError("output selector must be None, int, str, or callable")
    if not hasattr(selected, "is_floating_point") or not selected.is_floating_point():
        raise TypeError("selected output must be a floating point tensor")
    return selected


def _softmax_property(
    *,
    output_selector: Any,
    probability_axis: int,
    tolerance: float,
) -> Callable[..., Mapping[str, Any]]:
    def check(index: int, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Mapping[str, Any]:
        del index, args, kwargs
        probability = _select_output(output, output_selector)
        axis = probability_axis if probability_axis >= 0 else probability.ndim + probability_axis
        if axis < 0 or axis >= probability.ndim:
            raise ValueError("probability_axis is outside the output rank")
        # Measure the stored values without rounding the diagnostic back to
        # the candidate's output dtype (FP16 could hide a nonzero mass defect).
        import torch
        mass = probability.detach().to(device="cpu", dtype=torch.float64).sum(dim=axis)
        residual = mass - 1.0
        max_abs = float(residual.detach().abs().max().item())
        return {
            "passed": bool(max_abs <= tolerance),
            "residual": float(residual.detach().mean().item()),
            "max_abs_residual": max_abs,
            "property": "probability_mass_minus_one",
            "tolerance": tolerance,
        }

    return check


def check_softmax_saved_state(
    candidate: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    output_selector: Any = None,
    probability_axis: int = -1,
    tolerance: float = 1e-5,
    samples: int = 32,
    preserve_rng: bool = True,
) -> dict[str, Any]:
    """Check the declared softmax saved-probability mass contract.

    The contract is only that the selected probability tensor sums to one over
    ``probability_axis``.  It catches a saved-score/statistic mismatch such as
    the saved-P case, but it is not a complete softmax correctness proof.  A
    scalar residual mean is reported with a conditional interval; the top-level
    ``bias_decision`` remains ``NOT_ASSESSED`` because no full reference exists.
    """

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    report = diagnose_kernel(
        candidate,
        make_inputs,
        input_consistency_check=_softmax_property(
            output_selector=output_selector,
            probability_axis=probability_axis,
            tolerance=tolerance,
        ),
        samples=samples,
        preserve_rng=preserve_rng,
    )
    report["family"] = "softmax_saved_probability"
    report["family_contract"] = {
        "property": "sum(selected_probability, axis=probability_axis) == 1",
        "property_is_necessary_not_sufficient": True,
        "output_selector": "callable" if callable(output_selector) else output_selector,
        "probability_axis": probability_axis,
        "tolerance": tolerance,
        "reference_required_for_full_bias": True,
    }
    return report


def _flatten_output(value: Any) -> list[Any]:
    leaves = _flatten_tensors(value)
    if not leaves:
        raise ValueError("output contains no tensor leaves")
    return [tensor for _, tensor in leaves]


def _paired_difference(
    candidate: Callable[..., Any],
    variant: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    index: int,
    *,
    torch: Any,
    rng_state: Any,
) -> Any:
    raw_args, raw_kwargs = _normalise_call(make_inputs(index))
    cand_args, cand_kwargs = _clone_call(raw_args, raw_kwargs, torch=torch, requires_grad=False)
    out_candidate = candidate(*cand_args, **cand_kwargs)
    _restore_rng(torch, rng_state)
    var_args, var_kwargs = _clone_call(raw_args, raw_kwargs, torch=torch, requires_grad=False)
    out_variant = variant(*var_args, **var_kwargs)
    candidate_leaves = _flatten_output(out_candidate)
    variant_leaves = _flatten_output(out_variant)
    if len(candidate_leaves) != len(variant_leaves):
        raise ValueError("candidate and arithmetic variant output structures differ")
    effects = []
    for left, right in zip(candidate_leaves, variant_leaves):
        if left.shape != right.shape or not left.is_floating_point() or not right.is_floating_point():
            raise ValueError("candidate and arithmetic variant outputs differ in shape or dtype")
        effects.append(left.detach().to(dtype=torch.float64).reshape(-1) - right.detach().to(dtype=torch.float64).reshape(-1))
    return torch.cat(effects)


def _reduction_symmetry(
    candidate: Callable[..., Any],
    variant: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    make_negated_inputs: Callable[[int], Any],
    samples: int,
    *,
    torch: Any,
    preserve_rng: bool,
) -> dict[str, Any]:
    records = []
    for index in range(samples):
        try:
            rng_state = _snapshot_rng(torch) if preserve_rng else None
            positive = _paired_difference(
                candidate, variant, make_inputs, index, torch=torch, rng_state=rng_state
            )
            if preserve_rng:
                _restore_rng(torch, rng_state)
            negative = _paired_difference(
                candidate, variant, make_negated_inputs, index, torch=torch, rng_state=rng_state
            )
            odd_residual = negative + positive
            positive_rms = float(torch.sqrt(torch.mean(positive.square())).item())
            odd_rms = float(torch.sqrt(torch.mean(odd_residual.square())).item())
            records.append({
                "sample": index,
                "status": "VALID",
                "positive_effect_rms": positive_rms,
                "odd_symmetry_residual_rms": odd_rms,
                "relative_odd_residual": odd_rms / max(positive_rms, 1e-30),
            })
        except Exception as error:
            records.append({"sample": index, "status": "ERROR", "error": str(error)})
    valid = [item for item in records if item["status"] == "VALID"]
    return {
        "status": "VALID" if len(valid) == samples else ("PARTIAL" if valid else "INVALID"),
        "sample_count": len(records),
        "valid_count": len(valid),
        "records": records,
        "interpretation": "For a symmetric round-to-nearest reduction, negating all operands should negate the order difference; this is a diagnostic symmetry check, not a bias proof.",
    }


def check_reduction_order(
    candidate: Callable[..., Any],
    variant: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    samples: int = 32,
    make_negated_inputs: Optional[Callable[[int], Any]] = None,
    preserve_rng: bool = True,
) -> dict[str, Any]:
    """Compare two caller-declared same-semantic reduction orders.

    The result is controlled arithmetic evidence.  If ``make_negated_inputs``
    is supplied, the exact odd-symmetry check is also run.  A nonzero mean of
    the candidate/variant scalar effect is only conditional evidence for the
    declared input bank; it is not a proof that the candidate is biased against
    an unknown specification.
    """

    report = diagnose_kernel(
        candidate,
        make_inputs,
        rounding_variants={"DECLARED_REDUCTION_ORDER_VARIANT": variant},
        samples=samples,
        preserve_rng=preserve_rng,
    )
    report["family"] = "same_dtype_reduction_order"
    report["family_contract"] = {
        "candidate_and_variant_same_semantics_declared_by_caller": True,
        "difference_is_controlled_evidence_not_reference_error": True,
        "population_scope": "CALLER_DECLARED_INPUT_DISTRIBUTION_ONLY",
        "reference_required_for_full_output_bias": True,
    }
    if make_negated_inputs is not None:
        try:
            import torch
        except Exception as error:  # pragma: no cover
            report["symmetry"] = {"status": "UNAVAILABLE", "error": str(error)}
        else:
            report["symmetry"] = _reduction_symmetry(
                candidate,
                variant,
                make_inputs,
                make_negated_inputs,
                samples,
                torch=torch,
                preserve_rng=preserve_rng,
            )
    return report


def _negate_floating_structure(value: Any, *, torch: Any) -> tuple[Any, bool]:
    """Negate floating tensors in a call tree and report whether any changed."""

    if torch.is_tensor(value):
        if value.is_floating_point():
            return -value, True
        return value, False
    if isinstance(value, tuple):
        values = []
        changed = False
        for item in value:
            negated, item_changed = _negate_floating_structure(item, torch=torch)
            values.append(negated)
            changed = changed or item_changed
        return tuple(values), changed
    if isinstance(value, list):
        values = []
        changed = False
        for item in value:
            negated, item_changed = _negate_floating_structure(item, torch=torch)
            values.append(negated)
            changed = changed or item_changed
        return values, changed
    if isinstance(value, Mapping):
        values = {}
        changed = False
        for key, item in value.items():
            negated, item_changed = _negate_floating_structure(item, torch=torch)
            values[key] = negated
            changed = changed or item_changed
        return values, changed
    return value, False


def check_odd_symmetry(
    candidate: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    output_selector: Any = None,
    tolerance: float = 1e-5,
    samples: int = 32,
    preserve_rng: bool = True,
) -> dict[str, Any]:
    """Run a one-callable metamorphic screen for an odd-output contract.

    The only runtime inputs are ``candidate`` and ``make_inputs``.  Floating
    input tensors are automatically negated and the declared property
    ``f(-x) == -f(x)`` is checked on paired executions.  This is useful for
    reductions, signed linear maps, and other operators whose specification
    includes odd symmetry.  It is *not* valid for arbitrary kernels (for
    example, a GEMM with both operands negated is even), so the property must
    be selected by the caller.  The result is evidence about that property,
    never a reference-free full bias verdict.
    """

    if not callable(candidate) or not callable(make_inputs):
        raise TypeError("candidate and make_inputs must be callable")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be an integer >= 1")
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    try:
        import torch
    except Exception as error:  # pragma: no cover
        return {
            "schema": "kernel-analyzer-reference-free-diagnostics-v1",
            "measurement_status": "INVALID",
            "diagnostic_status": "UNAVAILABLE",
            "bias_decision": "NOT_ASSESSED",
            "property_decision": "UNRESOLVED_MEASUREMENT",
            "reference_status": "NOT_PROVIDED",
            "errors": [{"where": "IMPORT", "error": str(error)}],
        }

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index in range(samples):
        try:
            raw_args, raw_kwargs = _normalise_call(make_inputs(index))
            neg_args, changed_args = _negate_floating_structure(raw_args, torch=torch)
            neg_kwargs, changed_kwargs = _negate_floating_structure(raw_kwargs, torch=torch)
            if not (changed_args or changed_kwargs):
                raise ValueError("automatic odd probe found no floating input tensor")
            rng_state = _snapshot_rng(torch) if preserve_rng else None
            pos_args, pos_kwargs = _clone_call(raw_args, raw_kwargs, torch=torch, requires_grad=False)
            positive = _select_output(candidate(*pos_args, **pos_kwargs), output_selector)
            if preserve_rng:
                _restore_rng(torch, rng_state)
            neg_args, neg_kwargs = _clone_call(neg_args, neg_kwargs, torch=torch, requires_grad=False)
            negative = _select_output(candidate(*neg_args, **neg_kwargs), output_selector)
            if positive.shape != negative.shape or not positive.is_floating_point() or not negative.is_floating_point():
                raise ValueError("positive and negated outputs differ in shape or dtype")
            pos = positive.detach().to(dtype=torch.float64)
            neg = negative.detach().to(dtype=torch.float64)
            if not bool(torch.isfinite(pos).all() and torch.isfinite(neg).all()):
                raise ValueError("candidate output contains non-finite values")
            residual = pos + neg
            residual_rms = float(torch.sqrt(torch.mean(residual.square())).item())
            control_rms = float(torch.sqrt(torch.mean(pos.square())).item())
            records.append({
                "sample": index,
                "status": "VALID",
                "property": "f(-x) == -f(x)",
                "passed": residual_rms <= tolerance,
                "residual_rms": residual_rms,
                "relative_residual_rms": residual_rms / max(control_rms, 1e-30),
                "signed_mean_residual": float(residual.mean().item()),
                "max_abs_residual": float(residual.abs().max().item()),
                "tolerance": tolerance,
            })
        except Exception as error:
            records.append({"sample": index, "status": "ERROR", "error": str(error)})
            errors.append({"sample": index, "where": "ODD_SYMMETRY", "error": str(error)})

    valid = [record for record in records if record.get("status") == "VALID"]
    violations = [record for record in valid if not record.get("passed", False)]
    measured = len(valid)
    measurement_status = "VALID" if measured == samples else ("PARTIAL" if measured else "INVALID")
    if not measured:
        property_decision = "UNRESOLVED_MEASUREMENT"
        bias_signal = "UNRESOLVED_MEASUREMENT"
    elif violations:
        property_decision = "PROPERTY_VIOLATION_OBSERVED"
        bias_signal = "BIAS_CANDIDATE_PROPERTY_VIOLATION"
    else:
        property_decision = "PROPERTY_NOT_VIOLATED_IN_DECLARED_SAMPLE"
        bias_signal = "NO_PROPERTY_VIOLATION_OBSERVED"
    return {
        "schema": "kernel-analyzer-reference-free-diagnostics-v1",
        "family": "odd_symmetry",
        "measurement_status": measurement_status,
        "diagnostic_status": "EVIDENCE_ONLY" if measured else "UNRESOLVED_MEASUREMENT",
        "bias_decision": "NOT_ASSESSED",
        "bias_signal": bias_signal,
        "property_decision": property_decision,
        "reference_status": "NOT_PROVIDED",
        "sample_count": samples,
        "measured_sample_count": measured,
        "violation_count": len(violations),
        "records": records,
        "evidence_classes": ["METAMORPHIC_PROPERTY_VIOLATION"] if violations else [],
        "property_contract": {
            "property": "f(-x) == -f(x)",
            "input_transform": "automatic_negation_of_all_floating_input_tensors",
            "output_selector": "callable" if callable(output_selector) else output_selector,
            "tolerance": tolerance,
            "caller_must_declare_odd_semantics": True,
        },
        "errors": errors,
        "scope": "CALLER_DECLARED_ODD_PROPERTY_ONLY",
        "limitations": [
            "Odd symmetry is not valid for arbitrary operators.",
            "A property violation is not a full-reference bias verdict.",
            "A property pass is not proof of equivalence or training safety.",
            "Training gradient, optimizer update, and loss consequences are not measured here.",
        ],
    }


__all__ = ["check_odd_symmetry", "check_reduction_order", "check_softmax_saved_state"]
