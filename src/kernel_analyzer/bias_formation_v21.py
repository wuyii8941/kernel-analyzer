"""Corrected pre-measurement bias-formation protocol (v2.1).

The v2 scaffold stored a population statistic on every state row.  v2.1 has a
strict two-level data model: a state row owns one vector (or a one-shot vector
handle), while a ``LayerPopulationCertificate`` owns the single cross-state
Gram matrix and bootstrap interval for a partition/layer.

This module is intentionally independent of the v2 scaffold.  v1 and v2
artifacts remain immutable; v2.1 supersedes them before any GPU measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - the production measurement env has numpy
    np = None  # type: ignore[assignment]


class FormationLayer(str, Enum):
    LOCAL_ENDPOINT = "LOCAL_ENDPOINT"
    PARAMETER_GRADIENT = "PARAMETER_GRADIENT"
    EFFECTIVE_UPDATE = "EFFECTIVE_UPDATE"


class FormationStatus(str, Enum):
    COMPLETE = "COMPLETE"
    CENTERED = "CENTERED"
    BIASED = "BIASED"
    CANCELING_STRUCTURE = "CANCELING_STRUCTURE"
    UNRESOLVED_MISSING_LAYER = "UNRESOLVED_MISSING_LAYER"
    UNRESOLVED_INSUFFICIENT_STATES = "UNRESOLVED_INSUFFICIENT_STATES"
    # Enough states were collected, but the frozen equivalence/bias margins
    # do not separate the population from zero.  This is not a missing-state
    # error; keep it explicit so reports cannot misread an inconclusive result
    # as an incomplete capture.
    UNRESOLVED_INCONCLUSIVE = "UNRESOLVED_INCONCLUSIVE"
    INVALID_NONFINITE = "INVALID_NONFINITE"
    INVALID_PROJECTION = "INVALID_PROJECTION"
    INVALID_COMMON_STATE = "INVALID_COMMON_STATE"
    INVALID_MALFORMED = "INVALID_MALFORMED"


_LAYER_ORDER = (
    FormationLayer.LOCAL_ENDPOINT,
    FormationLayer.PARAMETER_GRADIENT,
    FormationLayer.EFFECTIVE_UPDATE,
)


@dataclass(frozen=True)
class FormationPolicy:
    """Frozen development margins; never fitted to confirmation verdicts."""

    min_states: int = 16
    # With only sixteen independent states, ordinary state bootstrap has a
    # nonzero finite-sample floor.  The equivalence margin is frozen from the
    # synthetic/repeat development controls, rather than from any case label.
    centered_margin: float = 0.20
    bias_margin: float = 0.25
    canceling_margin: float = 0.20
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260818
    energy_floor: float = 1e-30

    def __post_init__(self) -> None:
        if self.min_states < 2:
            raise ValueError("min_states must be at least two")
        if not (0 <= self.centered_margin < self.bias_margin):
            raise ValueError("centered and bias margins must be ordered")
        if self.canceling_margin < self.centered_margin:
            raise ValueError("canceling margin must be at least the centered margin")
        if self.bootstrap_samples < 2000:
            raise ValueError("v2.1 requires at least 2000 bootstrap samples")
        if self.energy_floor <= 0:
            raise ValueError("energy_floor must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_states": self.min_states,
            "centered_margin": self.centered_margin,
            "bias_margin": self.bias_margin,
            "canceling_margin": self.canceling_margin,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "energy_floor": self.energy_floor,
            "primary_statistic": "off_diagonal_cross_state_u_statistic_over_average_state_energy",
        }


@dataclass(frozen=True)
class CommonStateCertificate:
    """Component-wise candidate/repair pre-state equality certificate."""

    candidate_weights_digest: str
    repair_weights_digest: str
    candidate_optimizer_digest: str
    repair_optimizer_digest: str
    candidate_input_digest: str
    repair_input_digest: str
    candidate_rng_digest: str
    repair_rng_digest: str
    candidate_scheduler_digest: str
    repair_scheduler_digest: str
    candidate_loss_scaler_digest: str
    repair_loss_scaler_digest: str

    @classmethod
    def from_value(cls, value: Any) -> "CommonStateCertificate":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("common_state_certificate must be an object")
        fields = (
            "candidate_weights_digest", "repair_weights_digest",
            "candidate_optimizer_digest", "repair_optimizer_digest",
            "candidate_input_digest", "repair_input_digest",
            "candidate_rng_digest", "repair_rng_digest",
            "candidate_scheduler_digest", "repair_scheduler_digest",
            "candidate_loss_scaler_digest", "repair_loss_scaler_digest",
        )
        missing = [name for name in fields if not str(value.get(name, ""))]
        if missing:
            raise ValueError("common-state certificate is missing: " + ", ".join(missing))
        return cls(**{name: str(value[name]) for name in fields})

    @property
    def all_components_equal(self) -> bool:
        return (
            self.candidate_weights_digest == self.repair_weights_digest
            and self.candidate_optimizer_digest == self.repair_optimizer_digest
            and self.candidate_input_digest == self.repair_input_digest
            and self.candidate_rng_digest == self.repair_rng_digest
            and self.candidate_scheduler_digest == self.repair_scheduler_digest
            and self.candidate_loss_scaler_digest == self.repair_loss_scaler_digest
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_weights_digest": self.candidate_weights_digest,
            "repair_weights_digest": self.repair_weights_digest,
            "candidate_optimizer_digest": self.candidate_optimizer_digest,
            "repair_optimizer_digest": self.repair_optimizer_digest,
            "candidate_input_digest": self.candidate_input_digest,
            "repair_input_digest": self.repair_input_digest,
            "candidate_rng_digest": self.candidate_rng_digest,
            "repair_rng_digest": self.repair_rng_digest,
            "candidate_scheduler_digest": self.candidate_scheduler_digest,
            "repair_scheduler_digest": self.repair_scheduler_digest,
            "candidate_loss_scaler_digest": self.candidate_loss_scaler_digest,
            "repair_loss_scaler_digest": self.repair_loss_scaler_digest,
            "all_components_equal": self.all_components_equal,
        }


@dataclass(frozen=True)
class ProjectionCertificate:
    space: str
    basis_construction_method: str
    calibration_state_ids: Sequence[str]
    basis_digest: str
    orientation_rule: str
    uses_candidate_measurements: bool
    frozen_before_confirmation: bool

    def __post_init__(self) -> None:
        if not all((self.space, self.basis_construction_method, self.basis_digest, self.orientation_rule)):
            raise ValueError("projection provenance is incomplete")
        if not self.calibration_state_ids or not self.frozen_before_confirmation:
            raise ValueError("projection must be calibrated and frozen before confirmation")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "space": self.space,
            "basis_construction_method": self.basis_construction_method,
            "calibration_state_ids": list(self.calibration_state_ids),
            "basis_digest": self.basis_digest,
            "orientation_rule": self.orientation_rule,
            "uses_candidate_measurements": self.uses_candidate_measurements,
            "frozen_before_confirmation": self.frozen_before_confirmation,
        }


@dataclass(frozen=True)
class StateVectorObservation:
    """One state/layer vector; no cross-state matrix is stored here."""

    state_id: str
    coordinate_count: int
    vector_digest: str
    vector: tuple[float, ...] | None = field(default=None, repr=False, compare=False)
    vector_handle: Callable[[], Iterable[float]] | None = field(default=None, repr=False, compare=False)
    signed_projection: float | None = None
    projection: ProjectionCertificate | None = None

    def __post_init__(self) -> None:
        if not self.state_id or self.coordinate_count < 1 or not self.vector_digest:
            raise ValueError("state vector identity is incomplete")
        if self.vector is None and self.vector_handle is None:
            raise ValueError("a vector or one-shot vector handle is required")
        if self.vector is not None:
            if len(self.vector) != self.coordinate_count:
                raise ValueError("vector coordinate count mismatch")
            if any(not math.isfinite(x) for x in self.vector):
                raise ValueError("state vector contains a nonfinite value")
        if self.signed_projection is not None and not math.isfinite(self.signed_projection):
            raise ValueError("signed projection is nonfinite")
        if self.signed_projection is not None and self.projection is None:
            raise ValueError("scalar projection requires provenance")

    @classmethod
    def from_value(cls, state_id: str, value: Any) -> "StateVectorObservation":
        if isinstance(value, cls):
            if value.state_id != str(state_id):
                raise ValueError("state vector state_id mismatch")
            return value
        if isinstance(value, (list, tuple)):
            values = tuple(float(x) for x in value)
            if not values or any(not math.isfinite(x) for x in values):
                raise ValueError("state vector is empty or nonfinite")
            return cls(
                state_id=str(state_id), coordinate_count=len(values),
                vector_digest=_digest_vector(values), vector=values,
            )
        if not isinstance(value, Mapping):
            raise ValueError("state vector must be a vector or object")
        vector_value = value.get("vector")
        if vector_value is None:
            handle = value.get("vector_handle")
            if not callable(handle):
                raise ValueError("state vector object needs vector or callable vector_handle")
            coordinate_count = int(value.get("coordinate_count", 0))
            digest = str(value.get("vector_digest", ""))
            return cls(str(state_id), coordinate_count, digest, vector_handle=handle)
        values = tuple(float(x) for x in vector_value)
        if not values or any(not math.isfinite(x) for x in values):
            raise ValueError("state vector is empty or nonfinite")
        digest = str(value.get("vector_digest", _digest_vector(values)))
        return cls(
            state_id=str(state_id), coordinate_count=int(value.get("coordinate_count", len(values))),
            vector_digest=digest, vector=values,
            signed_projection=(None if value.get("signed_projection") is None else float(value["signed_projection"])),
            projection=_projection_from_value(value.get("projection")),
        )

    def materialize(self) -> tuple[float, ...]:
        if self.vector is not None:
            return self.vector
        assert self.vector_handle is not None
        values = tuple(float(x) for x in self.vector_handle())
        if len(values) != self.coordinate_count or any(not math.isfinite(x) for x in values):
            raise ValueError("streamed state vector is malformed or nonfinite")
        if _digest_vector(values) != self.vector_digest:
            raise ValueError("streamed state vector digest does not match provenance")
        return values

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "coordinate_count": self.coordinate_count,
            "vector_digest": self.vector_digest,
            "signed_projection": self.signed_projection,
            "projection": None if self.projection is None else self.projection.as_dict(),
        }


@dataclass(frozen=True)
class LayerPopulationCertificate:
    """The only object that owns a cross-state Gram matrix."""

    layer: str
    partition: str
    state_ids: tuple[str, ...]
    coordinate_count: int
    complete_gram: tuple[tuple[float, ...], ...]
    average_state_energy: float
    cross_state_u_statistic: float
    cross_state_ratio: float
    bootstrap_lower: float
    bootstrap_upper: float
    bootstrap_samples: int
    vector_digests: tuple[str, ...]
    status: str
    canceling_structure: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "partition": self.partition,
            "state_ids": list(self.state_ids),
            "state_count": len(self.state_ids),
            "coordinate_count": self.coordinate_count,
            "complete_gram": [list(row) for row in self.complete_gram],
            "average_state_energy": self.average_state_energy,
            "cross_state_u_statistic": self.cross_state_u_statistic,
            "cross_state_ratio": self.cross_state_ratio,
            "bootstrap_lower": self.bootstrap_lower,
            "bootstrap_upper": self.bootstrap_upper,
            "bootstrap_samples": self.bootstrap_samples,
            "vector_digests": list(self.vector_digests),
            "status": self.status,
            "canceling_structure": self.canceling_structure,
        }


def _digest_vector(values: Sequence[float]) -> str:
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def _projection_from_value(value: Any) -> ProjectionCertificate | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("projection must be an object")
    return ProjectionCertificate(
        space=str(value.get("space", "")),
        basis_construction_method=str(value.get("basis_construction_method", "")),
        calibration_state_ids=tuple(str(x) for x in value.get("calibration_state_ids", ())),
        basis_digest=str(value.get("basis_digest", "")),
        orientation_rule=str(value.get("orientation_rule", "")),
        uses_candidate_measurements=bool(value.get("uses_candidate_measurements", False)),
        frozen_before_confirmation=bool(value.get("frozen_before_confirmation", False)),
    )


def _gram(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[math.fsum(float(a) * float(b) for a, b in zip(left, right)) for right in rows] for left in rows]


def _ratio(rows: Sequence[Sequence[float]], policy: FormationPolicy) -> float:
    n = len(rows)
    if n < 2:
        raise ValueError("at least two rows are required")
    gram = _gram(rows)
    pairs = n * (n - 1) // 2
    cross = math.fsum(gram[i][j] for i in range(n) for j in range(i + 1, n)) / pairs
    energy = math.fsum(gram[i][i] for i in range(n)) / n
    return cross / max(energy, policy.energy_floor)


def _certificate_from_gram(
    gram: Any,
    *,
    coordinate_count: int,
    state_ids: Sequence[str],
    vector_digests: Sequence[str],
    layer: str,
    partition: str,
    policy: FormationPolicy,
) -> LayerPopulationCertificate:
    """Finish a population certificate once its complete Gram is available.

    This is shared by dense and disk-backed paths.  Keeping the status logic
    here is important: a streamed capture must be statistically identical to
    the in-memory reference path, not merely report a similar ratio.
    """
    if np is not None:
        matrix = np.asarray(gram, dtype=np.float64)
        if matrix.shape != (len(state_ids), len(state_ids)):
            raise ValueError("complete Gram shape does not match state IDs")
        diagonal = np.diag(matrix)
        average_energy = float(np.trace(matrix) / len(state_ids))
        pairs = len(state_ids) * (len(state_ids) - 1) / 2
        cross_u = float((matrix.sum() - np.trace(matrix)) / (2.0 * pairs))
        bootstrap = []
        rng = np.random.default_rng(policy.bootstrap_seed)
        while len(bootstrap) < policy.bootstrap_samples:
            indices = rng.integers(0, len(state_ids), size=len(state_ids))
            counts = np.bincount(indices, minlength=len(state_ids)).astype(np.float64)
            denominator = float(len(state_ids) * len(state_ids) - counts @ counts)
            if denominator <= 0.0:
                continue
            numerator = float(counts @ matrix @ counts - (counts * counts) @ diagonal)
            bootstrap.append(numerator / denominator / max(average_energy, policy.energy_floor))
        complete = matrix.tolist()
    else:
        matrix = [[float(x) for x in row] for row in gram]
        if len(matrix) != len(state_ids) or any(len(row) != len(state_ids) for row in matrix):
            raise ValueError("complete Gram shape does not match state IDs")
        diagonal = [matrix[i][i] for i in range(len(state_ids))]
        average_energy = math.fsum(diagonal) / len(state_ids)
        pairs = len(state_ids) * (len(state_ids) - 1) / 2
        cross_u = math.fsum(matrix[i][j] for i in range(len(state_ids)) for j in range(len(state_ids)) if i != j) / (2.0 * pairs)
        bootstrap = []
        rng = random.Random(policy.bootstrap_seed)
        for _ in range(policy.bootstrap_samples):
            sample = [rng.randrange(len(state_ids)) for _ in state_ids]
            counts = [sample.count(i) for i in range(len(state_ids))]
            denominator = float(len(state_ids) * len(state_ids) - sum(x * x for x in counts))
            if denominator <= 0.0:
                continue
            numerator = sum(counts[i] * counts[j] * matrix[i][j] for i in range(len(state_ids)) for j in range(len(state_ids)) if i != j)
            bootstrap.append(numerator / denominator / max(average_energy, policy.energy_floor))
        complete = matrix
    observed_ratio = cross_u / max(average_energy, policy.energy_floor)
    bootstrap.sort()
    lower = bootstrap[max(0, int(0.025 * len(bootstrap)))]
    upper = bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))]
    if lower >= policy.bias_margin or upper <= -policy.bias_margin:
        status = FormationStatus.BIASED.value
    elif lower >= -policy.centered_margin and upper <= policy.centered_margin:
        status = FormationStatus.CENTERED.value
    elif upper <= -policy.canceling_margin:
        status = FormationStatus.CANCELING_STRUCTURE.value
    else:
        status = FormationStatus.UNRESOLVED_INCONCLUSIVE.value
    return LayerPopulationCertificate(
        layer=str(layer), partition=str(partition), state_ids=tuple(str(x) for x in state_ids),
        coordinate_count=int(coordinate_count), complete_gram=tuple(tuple(float(x) for x in row) for row in complete),
        average_state_energy=average_energy, cross_state_u_statistic=cross_u,
        cross_state_ratio=observed_ratio, bootstrap_lower=lower,
        bootstrap_upper=upper, bootstrap_samples=policy.bootstrap_samples,
        vector_digests=tuple(str(x) for x in vector_digests), status=status,
        canceling_structure=(status == FormationStatus.CANCELING_STRUCTURE.value),
    )


def summarize_state_vectors(
    state_vectors: Sequence[Sequence[float]],
    *,
    state_ids: Sequence[str] | None = None,
    vector_digests: Sequence[str] | None = None,
    layer: str = "UNDECLARED",
    partition: str = "UNDECLARED",
    policy: FormationPolicy | None = None,
) -> LayerPopulationCertificate:
    """Dense population summary; one Gram is created for the whole partition."""

    policy = policy or FormationPolicy()
    rows = [tuple(float(x) for x in row) for row in state_vectors]
    if len(rows) < policy.min_states:
        raise ValueError("insufficient state vectors")
    dimension = len(rows[0])
    if dimension < 1 or any(len(row) != dimension for row in rows):
        raise ValueError("state vectors must share one coordinate count")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("state vectors must be finite")
    ids = tuple(str(x) for x in (state_ids or [str(i) for i in range(len(rows))]))
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError("state IDs must be unique and align with vectors")
    digests = tuple(vector_digests or [_digest_vector(row) for row in rows])
    if len(digests) != len(rows):
        raise ValueError("vector digests must align with vectors")
    if np is not None:
        gram = np.asarray(rows, dtype=np.float64) @ np.asarray(rows, dtype=np.float64).T
    else:
        gram = _gram(rows)
    return _certificate_from_gram(
        gram, coordinate_count=dimension, state_ids=ids, vector_digests=digests,
        layer=layer, partition=partition, policy=policy,
    )


def summarize_streamed_state_vector_files(
    rows: Sequence[Mapping[str, Any]], *, layer: str, partition: str,
    policy: FormationPolicy | None = None, chunk_elements: int = 1_048_576,
) -> LayerPopulationCertificate:
    """Summarize vectors retained as temporary float32/float64 files.

    Only a ``state_count × state_count`` Gram remains in memory.  This is the
    required path for large declared parameter carriers such as tied Qwen
    projections; callers own cleanup of the temporary files after finalizing.
    """
    policy = policy or FormationPolicy()
    if len(rows) < policy.min_states:
        raise ValueError("insufficient state vectors")
    coordinates = int(rows[0].get("coordinate_count", rows[0].get("coordinates", 0)))
    if coordinates < 1 or any(int(row.get("coordinate_count", row.get("coordinates", 0))) != coordinates for row in rows):
        raise ValueError("state vectors must share one coordinate count")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    arrays = []
    scales = []
    for row in rows:
        path = Path(str(row["path"]))
        storage_dtype = str(row.get("storage_dtype", "float32"))
        dtype = np.float32 if storage_dtype == "float32" else np.float64
        expected_bytes = coordinates * np.dtype(dtype).itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError("streamed vector file size does not match coordinate count")
        arrays.append(np.memmap(path, dtype=dtype, mode="r", shape=(coordinates,)))
        scales.append(float(row.get("scale", 1.0)))
    gram = np.zeros((len(arrays), len(arrays)), dtype=np.float64)
    for start in range(0, coordinates, chunk_elements):
        stop = min(coordinates, start + chunk_elements)
        block = np.stack([
            np.asarray(array[start:stop], dtype=np.float64) * scale
            for array, scale in zip(arrays, scales, strict=True)
        ])
        gram += block @ block.T
    certificate = _certificate_from_gram(
        gram, coordinate_count=coordinates,
        state_ids=[str(row["state_id"]) for row in rows],
        # Per-vector byte hashes are optional provenance.  The complete Gram,
        # coordinate count, state IDs, and common-state certificate are the
        # scientific measurement; hashing every transient multi-GB vector is
        # not required for the verdict.
        vector_digests=[str(row.get("vector_digest", "NOT_RETAINED")) for row in rows],
        layer=layer, partition=partition, policy=policy,
    )
    del arrays
    return certificate


def summarize_streamed_state_vectors(
    vectors: Iterable[Sequence[float]],
    **kwargs: Any,
) -> LayerPopulationCertificate:
    """Streaming input facade; it materializes only the partition population."""

    return summarize_state_vectors(list(vectors), **kwargs)


class BiasFormationTrace:
    """v2.1 open-loop formation trace with population-level statistics."""

    def __init__(
        self,
        case_id: str,
        calibration_state_ids: Sequence[str],
        confirmation_state_ids: Sequence[str],
        policy: FormationPolicy | None = None,
    ) -> None:
        self.case_id = str(case_id)
        self.calibration_state_ids = tuple(str(x) for x in calibration_state_ids)
        self.confirmation_state_ids = tuple(str(x) for x in confirmation_state_ids)
        if not self.case_id or not self.calibration_state_ids or not self.confirmation_state_ids:
            raise ValueError("case_id and two nonempty state partitions are required")
        if set(self.calibration_state_ids) & set(self.confirmation_state_ids):
            raise ValueError("calibration and confirmation IDs must be disjoint")
        self.policy = policy or FormationPolicy()
        self._rows: MutableMapping[str, Dict[str, Any]] = {}

    @property
    def expected_state_ids(self) -> tuple[str, ...]:
        return self.calibration_state_ids + self.confirmation_state_ids

    def add(
        self,
        state_id: str,
        partition: str,
        *,
        common_state_certificate: Any,
        local_endpoint: Any,
        parameter_gradient: Any,
        effective_update: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        state_id = str(state_id)
        if state_id in self._rows or state_id not in self.expected_state_ids:
            raise ValueError("state_id is duplicate or outside frozen split")
        expected = "calibration" if state_id in self.calibration_state_ids else "confirmation"
        if str(partition) != expected:
            raise ValueError("partition does not match frozen split")
        try:
            common = CommonStateCertificate.from_value(common_state_certificate)
            if not common.all_components_equal:
                common_error: str | None = "candidate_repair_state_components_differ"
            else:
                common_error = None
        except (TypeError, ValueError, KeyError) as exc:
            common = None
            common_error = str(exc)
        layers: Dict[str, Any] = {}
        for layer, value in {
            FormationLayer.LOCAL_ENDPOINT: local_endpoint,
            FormationLayer.PARAMETER_GRADIENT: parameter_gradient,
            FormationLayer.EFFECTIVE_UPDATE: effective_update,
        }.items():
            if value is None:
                layers[layer.value] = None
                continue
            try:
                layers[layer.value] = StateVectorObservation.from_value(state_id, value)
            except (TypeError, ValueError, KeyError) as exc:
                text = str(exc).lower()
                layers[layer.value] = "INVALID_NONFINITE" if "nonfinite" in text else "INVALID_MALFORMED"
        self._rows[state_id] = {
            "state_id": state_id, "partition": str(partition),
            "common_state": common, "common_error": common_error,
            "layers": layers, "metadata": dict(metadata or {}),
        }

    def _population(self, layer: FormationLayer, partition: str) -> tuple[LayerPopulationCertificate | None, str | None]:
        state_ids = self.calibration_state_ids if partition == "calibration" else self.confirmation_state_ids
        observations: list[StateVectorObservation] = []
        for state_id in state_ids:
            row = self._rows.get(state_id)
            if row is None or row["layers"].get(layer.value) is None:
                return None, FormationStatus.UNRESOLVED_MISSING_LAYER.value
            if row["common_error"] is not None:
                return None, FormationStatus.INVALID_COMMON_STATE.value
            value = row["layers"].get(layer.value)
            if isinstance(value, str) and value.startswith("INVALID"):
                return None, value
            observations.append(value)
        try:
            vectors = [obs.materialize() for obs in observations]
            cert = summarize_state_vectors(
                vectors, state_ids=state_ids,
                vector_digests=[obs.vector_digest for obs in observations],
                layer=layer.value, partition=partition, policy=self.policy,
            )
            if any(obs.signed_projection is not None and obs.projection is None for obs in observations):
                return cert, FormationStatus.INVALID_PROJECTION.value
            return cert, cert.status
        except ValueError as exc:
            text = str(exc).lower()
            return None, FormationStatus.INVALID_NONFINITE.value if "nonfinite" in text else FormationStatus.INVALID_MALFORMED.value

    def finalize(self) -> Dict[str, Any]:
        missing_rows = [state_id for state_id in self.expected_state_ids if state_id not in self._rows]
        unexpected_rows = [state_id for state_id in self._rows if state_id not in self.expected_state_ids]
        populations: Dict[str, Dict[str, Any]] = {"calibration": {}, "confirmation": {}}
        statuses: list[str] = []
        for partition in populations:
            for layer in _LAYER_ORDER:
                cert, status = self._population(layer, partition)
                populations[partition][layer.value] = None if cert is None else cert.as_dict()
                populations[partition][layer.value + "_status"] = status
                statuses.append(status)
        if missing_rows or unexpected_rows or any(s == FormationStatus.UNRESOLVED_MISSING_LAYER.value for s in statuses):
            overall = FormationStatus.UNRESOLVED_MISSING_LAYER.value
        elif FormationStatus.INVALID_COMMON_STATE.value in statuses:
            overall = FormationStatus.INVALID_COMMON_STATE.value
        elif FormationStatus.INVALID_NONFINITE.value in statuses:
            overall = FormationStatus.INVALID_NONFINITE.value
        elif FormationStatus.INVALID_MALFORMED.value in statuses:
            overall = FormationStatus.INVALID_MALFORMED.value
        elif FormationStatus.INVALID_PROJECTION.value in statuses:
            overall = FormationStatus.INVALID_PROJECTION.value
        elif any(s in {
            FormationStatus.UNRESOLVED_INSUFFICIENT_STATES.value,
            FormationStatus.UNRESOLVED_INCONCLUSIVE.value,
        } for s in statuses):
            overall = FormationStatus.UNRESOLVED_INCONCLUSIVE.value
        else:
            overall = FormationStatus.COMPLETE.value
        confirmation_statuses = [populations["confirmation"][layer.value + "_status"] for layer in _LAYER_ORDER]
        first_observed = next((layer.value for layer, status in zip(_LAYER_ORDER, confirmation_statuses) if status == FormationStatus.BIASED.value), None)
        # A confirmed formation point requires a complete confirmation
        # population.  In particular, a downstream BIASED layer must not be
        # promoted when an upstream layer is missing/malformed or when the
        # overall certificate is unresolved.
        first_confirmed = None
        if overall == FormationStatus.COMPLETE.value:
            prior_centered = True
            for layer, status in zip(_LAYER_ORDER, confirmation_statuses):
                if first_confirmed is None and prior_centered and status == FormationStatus.BIASED.value:
                    first_confirmed = layer.value
                if status != FormationStatus.CENTERED.value:
                    prior_centered = False
        return {
            "schema": "kernel-analyzer-bias-formation-certificate-v2_1",
            "case_id": self.case_id,
            "status": overall,
            "measurement_kind": "candidate_repair_ground_truth",
            "uses_candidate_measurements": True,
            "uses_historical_verdicts": False,
            "verdict_blind": True,
            "state_split": {
                "calibration_state_ids": list(self.calibration_state_ids),
                "confirmation_state_ids": list(self.confirmation_state_ids),
                "calibration_count": len(self.calibration_state_ids),
                "confirmation_count": len(self.confirmation_state_ids),
                "disjoint": not bool(set(self.calibration_state_ids) & set(self.confirmation_state_ids)),
                "both_open_loop_common_state": True,
            },
            "policy": self.policy.as_dict(),
            "populations": populations,
            "first_confirmed_bias_stage": first_confirmed,
            "first_observed_biased_stage": first_observed,
            "formation_point": "CONFIRMED" if first_confirmed and overall == FormationStatus.COMPLETE.value else "UNRESOLVED",
            "trajectory_drift_in_formation": False,
            "missing_rows": missing_rows,
            "unexpected_rows": unexpected_rows,
            "rows": [self._row_dict(state_id) for state_id in self.expected_state_ids if state_id in self._rows],
        }

    def _row_dict(self, state_id: str) -> Dict[str, Any]:
        row = self._rows[state_id]
        return {
            "state_id": state_id,
            "partition": row["partition"],
            "common_state": None if row["common_state"] is None else row["common_state"].as_dict(),
            "common_error": row["common_error"],
            "layers": {
                key: (value.as_dict() if isinstance(value, StateVectorObservation) else value)
                for key, value in row["layers"].items()
            },
            "metadata": row["metadata"],
        }
