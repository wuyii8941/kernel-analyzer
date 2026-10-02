"""Small TileLang-to-``check_bias`` integration helpers.

The adapter deliberately does not inspect TileLang IR or infer a reference
implementation.  A caller supplies a compiled/callable TileLang kernel, a
reference callable, and an input generator; the common bias checker performs
the measurement and reports the statistical scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .bias_checker import check_bias
from .family_diagnostics import check_odd_symmetry, check_reduction_order, check_softmax_saved_state
from .operator_specs import SumSpec


@dataclass(frozen=True)
class TileLangCallConfig:
    """How to select the value returned by a compiled TileLang kernel."""

    output_index: Optional[int] = None
    output_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.output_index is not None and self.output_key is not None:
            raise ValueError("choose output_index or output_key, not both")


def select_tilelang_output(value: Any, config: TileLangCallConfig) -> Any:
    """Select one output from a TileLang result without changing tensors."""

    if config.output_key is not None:
        if not isinstance(value, dict):
            raise TypeError("output_key requires a mapping-valued kernel result")
        return value[config.output_key]
    if config.output_index is not None:
        if not isinstance(value, (tuple, list)):
            raise TypeError("output_index requires a tuple/list-valued kernel result")
        return value[config.output_index]
    return value


def wrap_tilelang_callable(
    kernel: Callable[..., Any],
    *,
    output_index: Optional[int] = None,
    output_key: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap a compiled TileLang callable for the common checker."""

    if not callable(kernel):
        raise TypeError("kernel must be callable; compile it before wrapping")
    config = TileLangCallConfig(output_index=output_index, output_key=output_key)

    def invoke(*args: Any, **kwargs: Any) -> Any:
        return select_tilelang_output(kernel(*args, **kwargs), config)

    return invoke


def check_tilelang_bias(
    candidate: Callable[..., Any],
    reference: Optional[Callable[..., Any]] = None,
    make_inputs: Optional[Callable[[int], Any]] = None,
    *,
    specification: Optional[SumSpec] = None,
    candidate_output_index: Optional[int] = None,
    candidate_output_key: Optional[str] = None,
    reference_output_index: Optional[int] = None,
    reference_output_key: Optional[str] = None,
    **check_kwargs: Any,
) -> dict[str, Any]:
    """Run the shared bias check on a TileLang candidate/reference pair.

    ``make_inputs`` follows :func:`kernel_analyzer.bias_checker.check_bias`:
    it may return a positional tuple, one positional value, or an
    ``{"args": ..., "kwargs": ...}`` mapping.  Candidate and reference are
    run on cloned inputs by the shared checker.
    Alternatively select ``specification=SumSpec(...)``: the tool supplies
    the audited sum reference, while the caller still declares the semantics.
    """

    if make_inputs is None:
        raise ValueError("make_inputs is required")
    if (reference is None) == (specification is None):
        raise ValueError("provide exactly one of reference or specification")
    if specification is not None:
        if not isinstance(specification, SumSpec):
            raise TypeError("unsupported specification; use SumSpec or an explicit reference")
        if check_kwargs.get("check_backward", False):
            raise ValueError("SumSpec currently supports output checks only")
        if reference_output_index is not None or reference_output_key is not None:
            raise ValueError("builtin specification returns a tensor; no reference selector is needed")
        reference = specification.reference
    candidate_call = wrap_tilelang_callable(
        candidate,
        output_index=candidate_output_index,
        output_key=candidate_output_key,
    )
    reference_call = wrap_tilelang_callable(
        reference,
        output_index=reference_output_index,
        output_key=reference_output_key,
    )
    report = check_bias(candidate_call, reference_call, make_inputs, **check_kwargs)
    report["backend"] = "tilelang"
    report["backend_contract"] = {
        "candidate_is_compiled_callable": True,
        "reference_is_user_supplied": specification is None,
        "reference_source": "caller_callable" if specification is None else "builtin_specification",
        "source_inspection": False,
        "automatic_root_cause_analysis": False,
        "autotune_selection_must_be_frozen_before_confirmation": True,
        "input_distribution_is_caller_declared": True,
    }
    if specification is not None:
        report["specification"] = specification.contract()
    return report


def _tilelang_family_contract(report: dict[str, Any], family: str) -> dict[str, Any]:
    """Attach the backend scope to a reference-free family report."""

    report["backend"] = "tilelang"
    report["backend_contract"] = {
        "family": family,
        "candidate_is_compiled_callable": True,
        "reference_is_required": False,
        "source_inspection": False,
        "automatic_root_cause_analysis": False,
        "autotune_selection_must_be_frozen_before_confirmation": True,
        "input_distribution_is_caller_declared": True,
    }
    return report


def check_tilelang_softmax_saved_state(
    candidate: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    candidate_output_index: Optional[int] = None,
    candidate_output_key: Optional[str] = None,
    **family_kwargs: Any,
) -> dict[str, Any]:
    """Check a TileLang callable's declared saved-probability contract.

    This is intentionally a property diagnostic, not a substitute for a
    reference implementation.  It is useful when a saved score/statistic
    pair should reconstruct a probability whose selected axis sums to one.
    """

    if candidate_output_index is not None and candidate_output_key is not None:
        raise ValueError("choose candidate_output_index or candidate_output_key, not both")
    selector: Any = (
        candidate_output_index if candidate_output_index is not None else candidate_output_key
    )
    report = check_softmax_saved_state(
        wrap_tilelang_callable(
            candidate,
            output_index=candidate_output_index,
            output_key=candidate_output_key,
        ),
        make_inputs,
        output_selector=None,
        **family_kwargs,
    )
    # Selection happens in the TileLang wrapper so the family checker sees a
    # tensor directly; retain the user's selector in the contract.
    report["family_contract"]["tilelang_output_selector"] = selector
    return _tilelang_family_contract(report, "softmax_saved_probability")


def check_tilelang_reduction_order(
    candidate: Callable[..., Any],
    variant: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    candidate_output_index: Optional[int] = None,
    candidate_output_key: Optional[str] = None,
    variant_output_index: Optional[int] = None,
    variant_output_key: Optional[str] = None,
    **family_kwargs: Any,
) -> dict[str, Any]:
    """Compare two TileLang reduction-order callables under one contract."""

    candidate_call = wrap_tilelang_callable(
        candidate,
        output_index=candidate_output_index,
        output_key=candidate_output_key,
    )
    variant_call = wrap_tilelang_callable(
        variant,
        output_index=variant_output_index,
        output_key=variant_output_key,
    )
    report = check_reduction_order(candidate_call, variant_call, make_inputs, **family_kwargs)
    report["family_contract"]["tilelang_output_selectors"] = {
        "candidate": candidate_output_index if candidate_output_index is not None else candidate_output_key,
        "variant": variant_output_index if variant_output_index is not None else variant_output_key,
    }
    return _tilelang_family_contract(report, "same_dtype_reduction_order")


def check_tilelang_odd_symmetry(
    candidate: Callable[..., Any],
    make_inputs: Callable[[int], Any],
    *,
    candidate_output_index: Optional[int] = None,
    candidate_output_key: Optional[str] = None,
    **family_kwargs: Any,
) -> dict[str, Any]:
    """Screen one TileLang callable using its declared odd-output property.

    No reference or arithmetic variant is required.  The adapter automatically
    negates floating input tensors and checks ``f(-x) == -f(x)``.  The caller
    must declare that property; arbitrary kernels are not assumed to be odd.
    """

    candidate_call = wrap_tilelang_callable(
        candidate,
        output_index=candidate_output_index,
        output_key=candidate_output_key,
    )
    report = check_odd_symmetry(candidate_call, make_inputs, **family_kwargs)
    report["property_contract"]["tilelang_output_selector"] = (
        candidate_output_index if candidate_output_index is not None else candidate_output_key
    )
    return _tilelang_family_contract(report, "odd_symmetry")


def compile_tilelang_program(program: Any, *, output_index: Any = None, **kwargs: Any) -> Any:
    """Compile a TileLang program lazily.

    TileLang remains an optional dependency.  Importing this package does not
    require it; calling this function gives an actionable error if it is not
    installed.  Additional keyword arguments are passed to ``tilelang.compile``
    so the adapter does not hard-code a TileLang version-specific API.
    """

    try:
        import tilelang
    except ImportError as error:
        raise RuntimeError(
            "TileLang is not installed; install a compatible TileLang build "
            "before compiling a TileLang program"
        ) from error
    if output_index is not None:
        kwargs = {"out_idx": output_index, **kwargs}
    return tilelang.compile(program, **kwargs)
