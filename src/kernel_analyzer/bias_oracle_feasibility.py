"""Low-cost prototypes for screening conditional training bias.

The exact conditional-antithetic experiment remains the scientific
certificate.  This module contains deliberately smaller *screening* tools:

* a transported-mean probe for source/pairing asymmetry;
* a Rademacher antithetic probe for response curvature/rectification;
* an exact sample odd/even decomposition used to audit both channels; and
* helpers for retrospective repeat-budget ablations from retained Grams.

None of these functions turns a low-cost estimate into a correctness verdict.
An unreliable Taylor scale, a non-smooth response, or a missing local
perturbation boundary must be escalated to the exact experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Callable, Sequence


Vector = Sequence[float]
Response = Callable[[Vector], Vector]


def _vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def _same_dimension(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    _same_dimension(left, right)
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _subtract(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    _same_dimension(left, right)
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _scale(values: Sequence[float], factor: float) -> tuple[float, ...]:
    return tuple(factor * value for value in values)


def _mean(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("at least one vector is required")
    dimension = len(rows[0])
    if dimension < 1 or any(len(row) != dimension for row in rows):
        raise ValueError("vectors must share one nonzero dimension")
    return tuple(math.fsum(row[index] for row in rows) / len(rows) for index in range(dimension))


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _relative_difference(left: Sequence[float], right: Sequence[float]) -> float:
    return _l2(_subtract(left, right)) / max(_l2(left), _l2(right), 1e-30)


@dataclass(frozen=True)
class QuadraticBiasDecomposition:
    """Exact expected response for a quadratic local response map."""

    transported_mean: tuple[float, ...]
    curvature_rectification: tuple[float, ...]
    predicted_bias: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["transported_mean_l2"] = _l2(self.transported_mean)
        result["curvature_rectification_l2"] = _l2(self.curvature_rectification)
        result["predicted_bias_l2"] = _l2(self.predicted_bias)
        return result


def quadratic_bias_decomposition(
    source_mean: Sequence[float],
    source_second_moment: Sequence[Sequence[float]],
    jacobian: Sequence[Sequence[float]],
    output_hessians: Sequence[Sequence[Sequence[float]]],
) -> QuadraticBiasDecomposition:
    """Compute ``J E[e] + 1/2 H:E[ee^T]`` without choosing a carrier.

    ``source_second_moment`` is the raw moment ``E[e e^T]``, not merely the
    covariance.  For a quadratic response this is exact.  For a general F+B
    response it is only a local model and must pass an amplitude-consistency
    check before it can be used for screening.
    """

    mean = _vector(source_mean, name="source_mean")
    dimension = len(mean)
    second = tuple(tuple(float(value) for value in row) for row in source_second_moment)
    if len(second) != dimension or any(len(row) != dimension for row in second):
        raise ValueError("source_second_moment has the wrong shape")
    if any(not math.isfinite(value) for row in second for value in row):
        raise ValueError("source_second_moment must be finite")
    rows = tuple(tuple(float(value) for value in row) for row in jacobian)
    hessians = tuple(
        tuple(tuple(float(value) for value in row) for row in matrix)
        for matrix in output_hessians
    )
    if not rows or len(rows) != len(hessians):
        raise ValueError("jacobian and output Hessians must share output dimension")
    if any(len(row) != dimension for row in rows):
        raise ValueError("jacobian has the wrong input dimension")
    if any(
        len(matrix) != dimension or any(len(row) != dimension for row in matrix)
        for matrix in hessians
    ):
        raise ValueError("output Hessian has the wrong shape")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("jacobian must be finite")
    if any(not math.isfinite(value) for matrix in hessians for row in matrix for value in row):
        raise ValueError("output Hessians must be finite")
    transported = tuple(
        math.fsum(row[index] * mean[index] for index in range(dimension))
        for row in rows
    )
    curvature = tuple(
        0.5
        * math.fsum(
            matrix[i][j] * second[i][j]
            for i in range(dimension)
            for j in range(dimension)
        )
        for matrix in hessians
    )
    return QuadraticBiasDecomposition(
        transported_mean=transported,
        curvature_rectification=curvature,
        predicted_bias=_add(transported, curvature),
    )


@dataclass(frozen=True)
class PairedResponseDecomposition:
    """Exact empirical odd/even decomposition for natural residual samples."""

    sample_count: int
    natural_mean_response: tuple[float, ...]
    odd_mean_response: tuple[float, ...]
    even_mean_response: tuple[float, ...]
    closure_relative_error: float
    response_evaluations: int

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update({
            "natural_mean_l2": _l2(self.natural_mean_response),
            "odd_mean_l2": _l2(self.odd_mean_response),
            "even_mean_l2": _l2(self.even_mean_response),
        })
        return result


def paired_response_decomposition(
    residuals: Sequence[Sequence[float]],
    response: Response,
) -> PairedResponseDecomposition:
    """Decompose observed response bias into odd and even channels exactly.

    For every observed local residual ``e`` this evaluates the same F+B/update
    response at ``e`` and ``-e`` around a shared zero baseline.  The identity

    ``mean(F(e)-F(0)) = mean(F_odd(e)) + mean(F_even(e))``

    is algebraic; it does not assume a Taylor expansion.  The artificial
    negative arm may be infeasible for a natural low-precision representation,
    in which case the caller must abstain rather than project silently.
    """

    rows = [_vector(row, name="residual") for row in residuals]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("residuals must share one nonzero dimension")
    zero = (0.0,) * len(rows[0])
    baseline = _vector(response(zero), name="baseline response")
    plus_rows: list[tuple[float, ...]] = []
    odd_rows: list[tuple[float, ...]] = []
    even_rows: list[tuple[float, ...]] = []
    for row in rows:
        plus = _subtract(_vector(response(row), name="positive response"), baseline)
        minus = _subtract(_vector(response(_scale(row, -1.0)), name="negative response"), baseline)
        _same_dimension(plus, baseline)
        _same_dimension(minus, baseline)
        plus_rows.append(plus)
        odd_rows.append(_scale(_subtract(plus, minus), 0.5))
        even_rows.append(_scale(_add(plus, minus), 0.5))
    natural = _mean(plus_rows)
    odd = _mean(odd_rows)
    even = _mean(even_rows)
    return PairedResponseDecomposition(
        sample_count=len(rows),
        natural_mean_response=natural,
        odd_mean_response=odd,
        even_mean_response=even,
        closure_relative_error=_relative_difference(natural, _add(odd, even)),
        response_evaluations=1 + 2 * len(rows),
    )


@dataclass(frozen=True)
class MomentResponseSketch:
    """Dimension-independent local moment/response screening result."""

    transported_mean: tuple[float, ...]
    curvature_rectification: tuple[float, ...]
    predicted_bias: tuple[float, ...]
    baseline_evaluations: int
    transported_evaluations: int
    curvature_evaluations: int
    curvature_probe_count: int
    amplitude_relative_difference: float | None
    status: str

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update({
            "transported_mean_l2": _l2(self.transported_mean),
            "curvature_rectification_l2": _l2(self.curvature_rectification),
            "predicted_bias_l2": _l2(self.predicted_bias),
        })
        return result


def _rademacher_direction(
    factor: Sequence[Sequence[float]], rng: random.Random,
) -> tuple[float, ...]:
    rows = tuple(tuple(float(value) for value in row) for row in factor)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("covariance_factor must be a nonempty rectangular matrix")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("covariance_factor must be finite")
    signs = tuple(1.0 if rng.getrandbits(1) else -1.0 for _ in rows[0])
    return tuple(
        math.fsum(value * sign for value, sign in zip(row, signs, strict=True))
        for row in rows
    )


def _curvature_at_scale(
    response: Response,
    baseline: Sequence[float],
    directions: Sequence[Sequence[float]],
    scale: float,
) -> tuple[float, ...]:
    samples = []
    for direction in directions:
        plus = _vector(response(_scale(direction, scale)), name="positive probe response")
        minus = _vector(response(_scale(direction, -scale)), name="negative probe response")
        _same_dimension(plus, baseline)
        _same_dimension(minus, baseline)
        # [F(hv) + F(-hv)]/2 - F(0), divided by h^2.
        samples.append(
            _scale(_subtract(_scale(_add(plus, minus), 0.5), baseline), 1.0 / (scale * scale))
        )
    return _mean(samples)


def moment_response_sketch(
    source_mean: Sequence[float],
    covariance_factor: Sequence[Sequence[float]],
    response: Response,
    *,
    curvature_probes: int = 4,
    scale: float = 1.0,
    seed: int = 20260820,
    check_half_scale: bool = True,
    amplitude_tolerance: float = 0.25,
) -> MomentResponseSketch:
    """Screen first- and second-moment bias channels with shared probes.

    ``covariance_factor`` is any L satisfying approximately ``L L^T = Cov(e)``.
    The transported term uses one symmetric response pair at the conditional
    source mean.  The curvature term uses Rademacher directions through L and
    therefore costs ``2 * curvature_probes`` response evaluations independent
    of the local tensor dimension.  A half-scale repeat is a mandatory
    reliability gate for nonlinear/nonsmooth responses in scientific use.
    """

    mean = _vector(source_mean, name="source_mean")
    if curvature_probes < 1 or scale <= 0.0 or amplitude_tolerance < 0.0:
        raise ValueError("probe count and scale must be positive")
    if len(covariance_factor) != len(mean):
        raise ValueError("covariance factor and source mean dimensions differ")
    baseline = _vector(response((0.0,) * len(mean)), name="baseline response")
    mean_plus = _vector(response(mean), name="positive mean response")
    mean_minus = _vector(response(_scale(mean, -1.0)), name="negative mean response")
    _same_dimension(mean_plus, baseline)
    _same_dimension(mean_minus, baseline)
    transported = _scale(_subtract(mean_plus, mean_minus), 0.5)
    # The raw second moment is Cov(e) + E[e]E[e]^T.  The symmetric response
    # at +/- mean supplies the second term without another response pair.
    mean_curvature = _subtract(_scale(_add(mean_plus, mean_minus), 0.5), baseline)
    rng = random.Random(seed)
    directions = tuple(
        _rademacher_direction(covariance_factor, rng)
        for _ in range(curvature_probes)
    )
    covariance_curvature = _curvature_at_scale(response, baseline, directions, scale)
    curvature = _add(mean_curvature, covariance_curvature)
    amplitude_difference: float | None = None
    curvature_evaluations = 2 * curvature_probes
    status = "SCREEN"
    if check_half_scale:
        mean_plus_half = _vector(
            response(_scale(mean, 0.5)), name="half positive mean response"
        )
        mean_minus_half = _vector(
            response(_scale(mean, -0.5)), name="half negative mean response"
        )
        mean_curvature_half = _scale(
            _subtract(_scale(_add(mean_plus_half, mean_minus_half), 0.5), baseline),
            4.0,
        )
        covariance_half = _curvature_at_scale(response, baseline, directions, scale / 2.0)
        half = _add(mean_curvature_half, covariance_half)
        curvature_evaluations += 2 * curvature_probes
        amplitude_difference = _relative_difference(curvature, half)
        if amplitude_difference > amplitude_tolerance:
            status = "ESCALATE_NONLOCAL_OR_NONSMOOTH_RESPONSE"
    return MomentResponseSketch(
        transported_mean=transported,
        curvature_rectification=curvature,
        predicted_bias=_add(transported, curvature),
        baseline_evaluations=1,
        transported_evaluations=(4 if check_half_scale else 2),
        curvature_evaluations=curvature_evaluations,
        curvature_probe_count=curvature_probes,
        amplitude_relative_difference=amplitude_difference,
        status=status,
    )


def subset_square_matrix(
    matrix: Sequence[Sequence[float]], indices: Sequence[int],
) -> list[list[float]]:
    """Return a principal submatrix after strict index validation."""

    size = len(matrix)
    if size < 1 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    selected = tuple(int(index) for index in indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("indices must be nonempty and unique")
    if min(selected) < 0 or max(selected) >= size:
        raise ValueError("matrix subset index is out of range")
    return [[float(matrix[i][j]) for j in selected] for i in selected]


@dataclass(frozen=True)
class SharedBlockHvpSketch:
    """One scalar-output derivative sketch for every declared local block."""

    transported_mean_projections: tuple[float, ...]
    curvature_projections: tuple[float, ...]
    curvature_standard_errors: tuple[float, ...]
    probe_count: int
    shared_forward_evaluations: int
    shared_first_backward_passes: int
    shared_hvp_passes: int
    cross_block_terms_cancel_only_in_expectation: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def shared_block_hvp_sketch(
    scalar_response: Any,
    blocks: Sequence[Any],
    source_means: Sequence[Any],
    covariance_factors: Sequence[Any],
    *,
    probes: int = 8,
    seed: int = 20260820,
    probe_signs: Sequence[Sequence[Sequence[float]]] | None = None,
) -> SharedBlockHvpSketch:
    """Estimate all blockwise second-moment responses with shared HVPs.

    This is the main cost experiment.  Let all local injection variables be
    concatenated into ``z`` and let the scalar response be a fixed random
    projection of the effective optimizer update.  One reverse pass gives
    every block's first-order transported mean.  For each global Rademacher
    code, one HVP gives every block's curvature estimate simultaneously:

    ``0.5 * v_i^T (H v)_i``.

    Cross-block Hessian terms vanish in expectation because block codes are
    independent.  They are sampling variance, not silently discarded terms.
    ``covariance_factors[i]`` has shape ``(block_numel, latent_rank)`` and
    maps a Rademacher code to a perturbation with the desired covariance.

    The function intentionally imports torch lazily so the core analyzer can
    still be imported in CPU-only/minimal environments.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in torch envs
        raise RuntimeError("shared HVP screening requires torch") from exc
    if len(blocks) < 1 or len(blocks) != len(source_means) or len(blocks) != len(covariance_factors):
        raise ValueError("blocks, means, and covariance factors must be nonempty and aligned")
    if probes < 1:
        raise ValueError("probes must be positive")
    if getattr(scalar_response, "numel", lambda: 0)() != 1:
        raise ValueError("scalar_response must contain exactly one value")
    for block, mean, factor in zip(blocks, source_means, covariance_factors, strict=True):
        if not block.requires_grad:
            raise ValueError("every local injection block must require gradients")
        if tuple(mean.shape) != tuple(block.shape):
            raise ValueError("source mean shape does not match its block")
        if factor.ndim != 2 or factor.shape[0] != block.numel() or factor.shape[1] < 1:
            raise ValueError("covariance factor must have shape (block_numel, latent_rank)")
    gradients = torch.autograd.grad(
        scalar_response,
        tuple(blocks),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    transported = tuple(
        float((gradient.reshape(-1) * mean.reshape(-1)).sum().detach().cpu())
        for gradient, mean in zip(gradients, source_means, strict=True)
    )
    rng = random.Random(seed)
    if probe_signs is not None and len(probe_signs) != probes:
        raise ValueError("probe_signs must contain exactly probes global codes")
    samples: list[list[float]] = [[] for _ in blocks]
    for probe_index in range(probes):
        directions = []
        for block_index, (block, factor) in enumerate(zip(blocks, covariance_factors, strict=True)):
            rank = int(factor.shape[1])
            if probe_signs is None:
                signs = [1.0 if rng.getrandbits(1) else -1.0 for _ in range(rank)]
            else:
                signs = [float(value) for value in probe_signs[probe_index][block_index]]
                if len(signs) != rank or any(value not in {-1.0, 1.0} for value in signs):
                    raise ValueError("each supplied code must be a rank-aligned +/-1 vector")
            code = torch.as_tensor(signs, dtype=factor.dtype, device=factor.device)
            directions.append((factor @ code).reshape(block.shape))
        directional_derivative = sum(
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, directions, strict=True)
        )
        if directional_derivative.requires_grad:
            hvp = torch.autograd.grad(
                directional_derivative,
                tuple(blocks),
                retain_graph=True,
                create_graph=False,
                # A block can have an exactly linear response, in which case
                # its Hessian-vector product is mathematically zero and
                # autograd returns None. Unsupported double backward still
                # raises.
                allow_unused=True,
            )
        else:
            hvp = (None,) * len(blocks)
        for block_index, (direction, product) in enumerate(zip(directions, hvp, strict=True)):
            samples[block_index].append(
                0.0 if product is None else
                0.5 * float((direction * product).sum().detach().cpu())
            )
    means = tuple(math.fsum(values) / probes for values in samples)
    standard_errors = tuple(
        (
            math.sqrt(
                math.fsum((value - mean) ** 2 for value in values) / (probes - 1)
            )
            / math.sqrt(probes)
            if probes > 1
            else float("nan")
        )
        for values, mean in zip(samples, means, strict=True)
    )
    return SharedBlockHvpSketch(
        transported_mean_projections=transported,
        curvature_projections=means,
        curvature_standard_errors=standard_errors,
        probe_count=probes,
        shared_forward_evaluations=1,
        shared_first_backward_passes=1,
        shared_hvp_passes=probes,
        cross_block_terms_cancel_only_in_expectation=True,
    )
