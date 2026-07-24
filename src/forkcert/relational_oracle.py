"""Generic implementation-relative endpoint Oracle.

This module intentionally knows nothing about Gather, reductions, clipping, or
any compiler-specific operator.  It turns a declared reference artifact and a
candidate output into auditable continuous and semantic measurements.  It is a
measurement/gating primitive, not a root-cause detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


def _fingerprint(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    finite = bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else True
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": __import__("hashlib").sha256(contiguous.tobytes()).hexdigest(),
        "min": float(array.min()) if array.size and finite else None,
        "max": float(array.max()) if array.size and finite else None,
    }


@dataclass(frozen=True)
class EndpointOracle:
    """Paired endpoint observation under a declared reference contract."""

    shape_match: bool
    exact_match: bool
    max_abs_delta: float
    mean_signed_delta: float | None
    l2_delta: float
    nonzero_delta_count: int
    disagreement_count: int
    finite_match: bool
    reference: dict[str, Any]
    candidate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_endpoint_oracle(reference: Any, candidate: Any) -> EndpointOracle:
    """Compute generic continuous and discrete disagreement measurements.

    ``disagreement_count`` is meaningful for boolean/integer endpoints and is
    still reported for numeric endpoints as exact element disagreement.  No
    threshold or operator-specific semantic rule is inferred here.
    """

    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    shape_match = ref.shape == cand.shape
    if not shape_match:
        return EndpointOracle(
            shape_match=False,
            exact_match=False,
            max_abs_delta=float("inf"),
            mean_signed_delta=None,
            l2_delta=float("inf"),
            nonzero_delta_count=-1,
            disagreement_count=-1,
            finite_match=False,
            reference=_fingerprint(ref),
            candidate=_fingerprint(cand),
        )
    ref_num = ref.astype(np.float64, copy=False)
    cand_num = cand.astype(np.float64, copy=False)
    delta = cand_num - ref_num
    equal = np.equal(ref, cand)
    finite_match = bool(np.isfinite(ref_num).all() and np.isfinite(cand_num).all())
    if finite_match:
        max_abs_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        mean_signed_delta = float(np.mean(delta)) if delta.size else 0.0
        l2_delta = float(np.sqrt(np.sum(np.square(delta)))) if delta.size else 0.0
    else:
        # Non-finite values are not a numerical estimate.  Preserve the
        # witness, but make every magnitude fail closed instead of emitting
        # non-standard JSON NaN values.
        max_abs_delta = float("inf")
        mean_signed_delta = None
        l2_delta = float("inf")
    return EndpointOracle(
        shape_match=True,
        exact_match=bool(np.array_equal(ref, cand)),
        max_abs_delta=max_abs_delta,
        mean_signed_delta=mean_signed_delta,
        l2_delta=l2_delta,
        nonzero_delta_count=int(np.count_nonzero(delta)),
        disagreement_count=int(np.count_nonzero(~equal)),
        finite_match=finite_match,
        reference=_fingerprint(ref),
        candidate=_fingerprint(cand),
    )


def compute_repeatability(outputs: Iterable[Any]) -> dict[str, Any]:
    """Measure within-execution variation without calling it compiler bias."""

    arrays = [np.asarray(value) for value in outputs]
    if not arrays:
        return {"instantiated": False, "reason": "no repeats"}
    if len({tuple(value.shape) for value in arrays}) != 1:
        return {"instantiated": False, "reason": "repeat shapes differ"}
    stack = np.stack([value.astype(np.float64, copy=False) for value in arrays])
    finite_all = bool(np.isfinite(stack).all())
    if not finite_all:
        return {
            "instantiated": len(arrays) > 1,
            "n_repeats": len(arrays),
            "exact_all_repeats": bool(np.array_equal(stack, np.broadcast_to(stack[0], stack.shape))),
            "finite_all_repeats": False,
            "nonfinite_count": int(np.size(stack) - np.isfinite(stack).sum()),
            "max_abs_pairwise_from_first": float("inf"),
            "mean_element_variance": float("inf"),
            "valid_numeric_variance": False,
        }
    return {
        "instantiated": len(arrays) > 1,
        "n_repeats": len(arrays),
        "exact_all_repeats": bool(np.all(stack == stack[0])),
        "finite_all_repeats": True,
        "nonfinite_count": 0,
        "max_abs_pairwise_from_first": float(np.max(np.abs(stack - stack[0]))) if stack.size else 0.0,
        "mean_element_variance": float(np.var(stack, axis=0, ddof=1).mean()) if len(arrays) > 1 else 0.0,
        "valid_numeric_variance": True,
    }
