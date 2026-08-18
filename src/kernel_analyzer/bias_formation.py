"""Version-2 common-state bias-formation measurements.

The v1 scaffold mixed candidate/repair formation measurements with a live
trajectory.  This module is the v2 ground-truth layer: it accepts candidate
and repair residual summaries from *matched common states* and reports where a
complete-vector bias is first observed.  It deliberately has no trajectory
field and cannot consume T1--T4 or SEUP verdicts.

The persisted object is small.  A runner may stream full vectors and pass the
complete Gram/U-statistic summary; toy and CPU tests may pass vectors directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import random
from typing import Any, Dict, Mapping, MutableMapping, Sequence


class FormationLayer(str, Enum):
    LOCAL_ENDPOINT = "LOCAL_ENDPOINT"
    PARAMETER_GRADIENT = "PARAMETER_GRADIENT"
    EFFECTIVE_UPDATE = "EFFECTIVE_UPDATE"


class FormationStatus(str, Enum):
    COMPLETE = "COMPLETE"
    CENTERED = "CENTERED"
    BIASED = "BIASED"
    UNRESOLVED_MISSING_LAYER = "UNRESOLVED_MISSING_LAYER"
    UNRESOLVED_INSUFFICIENT_STATES = "UNRESOLVED_INSUFFICIENT_STATES"
    INVALID_NONFINITE = "INVALID_NONFINITE"
    INVALID_PROJECTION = "INVALID_PROJECTION"
    INVALID_COMMON_STATE = "INVALID_COMMON_STATE"


_LAYER_ORDER = (
    FormationLayer.LOCAL_ENDPOINT,
    FormationLayer.PARAMETER_GRADIENT,
    FormationLayer.EFFECTIVE_UPDATE,
)


@dataclass(frozen=True)
class FormationPolicy:
    """Frozen descriptive gates, not a property definition."""

    min_states: int = 4
    centered_ratio_upper: float = 0.01
    biased_ratio_lower: float = 0.05
    bootstrap_samples: int = 256
    bootstrap_seed: int = 20260818
    energy_floor: float = 1e-30

    def __post_init__(self) -> None:
        if self.min_states < 2:
            raise ValueError("min_states must be at least two")
        if self.centered_ratio_upper < 0 or self.biased_ratio_lower <= self.centered_ratio_upper:
            raise ValueError("centred/bias gates must be ordered and nonnegative")
        if self.bootstrap_samples < 32:
            raise ValueError("bootstrap_samples must be at least 32")
        if self.energy_floor <= 0:
            raise ValueError("energy_floor must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_states": self.min_states,
            "centered_ratio_upper": self.centered_ratio_upper,
            "biased_ratio_lower": self.biased_ratio_lower,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "energy_floor": self.energy_floor,
        }


@dataclass(frozen=True)
class ProjectionCertificate:
    """Provenance required for any auxiliary scalar projection."""

    space: str
    basis_construction_method: str
    calibration_state_ids: Sequence[str]
    basis_digest: str
    orientation_rule: str
    uses_candidate_measurements: bool
    frozen_before_confirmation: bool

    def __post_init__(self) -> None:
        required = {
            "space": self.space,
            "basis_construction_method": self.basis_construction_method,
            "basis_digest": self.basis_digest,
            "orientation_rule": self.orientation_rule,
        }
        if any(not str(value) for value in required.values()):
            raise ValueError("projection provenance is incomplete")
        if not self.calibration_state_ids:
            raise ValueError("projection must name calibration states")
        if not self.frozen_before_confirmation:
            raise ValueError("projection must be frozen before confirmation")

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
class VectorObservation:
    """Complete-vector summary for one common state and one formation layer."""

    coordinate_count: int
    vector_digest: str
    mean_vector_energy: float
    total_error_energy: float
    u_statistic: float
    bootstrap_lower: float
    bootstrap_upper: float
    signed_projection: float | None = None
    projection: ProjectionCertificate | None = None
    vector: tuple[float, ...] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.coordinate_count < 1 or not self.vector_digest:
            raise ValueError("complete-vector identity is missing")
        numeric = (
            self.mean_vector_energy,
            self.total_error_energy,
            self.u_statistic,
            self.bootstrap_lower,
            self.bootstrap_upper,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("vector summary contains a nonfinite value")
        if self.mean_vector_energy < 0 or self.total_error_energy < 0:
            raise ValueError("vector energies must be nonnegative")
        if self.bootstrap_lower > self.bootstrap_upper:
            raise ValueError("bootstrap interval is reversed")
        if self.signed_projection is not None and not math.isfinite(float(self.signed_projection)):
            raise ValueError("signed projection must be finite")
        if self.projection is not None and self.signed_projection is None:
            raise ValueError("a projection certificate requires a signed projection")

    @classmethod
    def from_value(cls, value: Any, policy: FormationPolicy) -> "VectorObservation":
        if isinstance(value, VectorObservation):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("formation observations must be complete-vector objects")
        projection_payload = value.get("projection")
        projection = None
        if projection_payload is not None:
            if not isinstance(projection_payload, Mapping):
                raise ValueError("projection must be an object")
            projection = ProjectionCertificate(
                space=str(projection_payload.get("space", "")),
                basis_construction_method=str(projection_payload.get("basis_construction_method", "")),
                calibration_state_ids=tuple(str(x) for x in projection_payload.get("calibration_state_ids", ())),
                basis_digest=str(projection_payload.get("basis_digest", "")),
                orientation_rule=str(projection_payload.get("orientation_rule", "")),
                uses_candidate_measurements=bool(projection_payload.get("uses_candidate_measurements", False)),
                frozen_before_confirmation=bool(projection_payload.get("frozen_before_confirmation", False)),
            )
        vector = value.get("vector")
        if vector is not None:
            if not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError("vector must be a nonempty numeric sequence")
            values = tuple(float(x) for x in vector)
            if any(not math.isfinite(x) for x in values):
                raise ValueError("vector contains a nonfinite value")
            # A row-level vector is only a state observation.  Cross-state
            # centering is performed by the trace after all rows arrive, so a
            # single row has a degenerate (zero) cross-state U-statistic.
            # Keeping this path makes the CPU/synthetic runner useful without
            # pretending that one state proves a bias.
            energy = _dot(values, values)
            digest = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
            return cls(
                coordinate_count=len(values),
                vector_digest=digest,
                mean_vector_energy=energy,
                total_error_energy=energy,
                u_statistic=0.0,
                bootstrap_lower=0.0,
                bootstrap_upper=0.0,
                signed_projection=(None if value.get("signed_projection") is None
                                   else float(value["signed_projection"])),
                projection=projection,
                vector=values,
            )
        required = (
            "coordinate_count", "vector_digest", "mean_vector_energy",
            "total_error_energy", "u_statistic", "bootstrap_lower", "bootstrap_upper",
        )
        if any(key not in value for key in required):
            raise ValueError("complete Gram/U-statistic summary is required")
        return cls(
            coordinate_count=int(value["coordinate_count"]),
            vector_digest=str(value["vector_digest"]),
            mean_vector_energy=float(value["mean_vector_energy"]),
            total_error_energy=float(value["total_error_energy"]),
            u_statistic=float(value["u_statistic"]),
            bootstrap_lower=float(value["bootstrap_lower"]),
            bootstrap_upper=float(value["bootstrap_upper"]),
            signed_projection=(None if value.get("signed_projection") is None
                               else float(value["signed_projection"])),
            projection=projection,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_count": self.coordinate_count,
            "vector_digest": self.vector_digest,
            "mean_vector_energy": self.mean_vector_energy,
            "total_error_energy": self.total_error_energy,
            "u_statistic": self.u_statistic,
            "bootstrap_lower": self.bootstrap_lower,
            "bootstrap_upper": self.bootstrap_upper,
            "signed_projection": self.signed_projection,
            "projection": None if self.projection is None else self.projection.as_dict(),
        }


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def _mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    n = len(vectors)
    return tuple(math.fsum(row[index] for row in vectors) / n for index in range(len(vectors[0])))


def _mean_energy(vectors: Sequence[Sequence[float]]) -> float:
    mean = _mean_vector(vectors)
    return _dot(mean, mean)


def summarize_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    policy: FormationPolicy | None = None,
) -> Dict[str, Any]:
    """Compute complete Gram, U-statistic and deterministic bootstrap summaries."""

    policy = policy or FormationPolicy()
    if len(vectors) < 2:
        raise ValueError("at least two vectors are required for a cross-state summary")
    rows = [tuple(float(x) for x in row) for row in vectors]
    dimension = len(rows[0])
    if dimension < 1 or any(len(row) != dimension for row in rows):
        raise ValueError("vectors must share one nonzero coordinate count")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("vectors must be finite")
    gram = [[_dot(left, right) for right in rows] for left in rows]
    total_energy = math.fsum(gram[index][index] for index in range(len(rows))) / len(rows)
    mean_energy = _mean_energy(rows)
    pair_count = len(rows) * (len(rows) - 1) // 2
    u_stat = math.fsum(gram[i][j] for i in range(len(rows)) for j in range(i + 1, len(rows))) / pair_count
    rng = random.Random(policy.bootstrap_seed)
    bootstrap: list[float] = []
    for _ in range(policy.bootstrap_samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        bootstrap.append(_mean_energy(sample) / max(
            math.fsum(_dot(row, row) for row in sample) / len(sample),
            policy.energy_floor,
        ))
    bootstrap.sort()
    lo = bootstrap[max(0, int(0.025 * len(bootstrap)))]
    hi = bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))]
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return {
        "coordinate_count": dimension,
        "vector_digest": digest,
        "mean_vector_energy": mean_energy,
        "total_error_energy": total_energy,
        "u_statistic": u_stat,
        "bootstrap_lower": lo,
        "bootstrap_upper": hi,
        "complete_gram": gram,
        "mean_energy_ratio": mean_energy / max(total_energy, policy.energy_floor),
        "u_energy_ratio": u_stat / max(total_energy, policy.energy_floor),
        "bootstrap_seed": policy.bootstrap_seed,
        "bootstrap_samples": policy.bootstrap_samples,
    }


def _layer_status(observations: Sequence[VectorObservation], policy: FormationPolicy) -> FormationStatus:
    if not observations:
        return FormationStatus.UNRESOLVED_MISSING_LAYER
    if len(observations) < policy.min_states:
        return FormationStatus.UNRESOLVED_INSUFFICIENT_STATES
    ratios = [obs.mean_vector_energy / max(obs.total_error_energy, policy.energy_floor)
              for obs in observations]
    lower = min(obs.bootstrap_lower for obs in observations)
    upper = max(obs.bootstrap_upper for obs in observations)
    if upper <= policy.centered_ratio_upper:
        return FormationStatus.CENTERED
    if lower >= policy.biased_ratio_lower:
        return FormationStatus.BIASED
    # Keep the continuous score, but do not convert an ambiguous interval to a
    # positive or negative categorical label.
    del ratios
    return FormationStatus.UNRESOLVED_INSUFFICIENT_STATES


def _layer_summary(layer: FormationLayer, observations: Sequence[VectorObservation], policy: FormationPolicy) -> Dict[str, Any]:
    if not observations:
        return {
            "layer": layer.value,
            "state_count": 0,
            "status": FormationStatus.UNRESOLVED_MISSING_LAYER.value,
            "complete_vector_statistics": False,
        }
    status = _layer_status(observations, policy)
    coordinate_counts = sorted({obs.coordinate_count for obs in observations})
    ratio_values = [obs.mean_vector_energy / max(obs.total_error_energy, policy.energy_floor)
                    for obs in observations]
    projection_ok = all(
        obs.projection is not None for obs in observations if obs.signed_projection is not None
    )
    return {
        "layer": layer.value,
        "state_count": len(observations),
        "coordinate_counts": coordinate_counts,
        "vector_digests": [obs.vector_digest for obs in observations],
        "mean_vector_energy_mean": math.fsum(obs.mean_vector_energy for obs in observations) / len(observations),
        "total_error_energy_mean": math.fsum(obs.total_error_energy for obs in observations) / len(observations),
        "mean_energy_ratio_mean": math.fsum(ratio_values) / len(ratio_values),
        "u_statistic_mean": math.fsum(obs.u_statistic for obs in observations) / len(observations),
        "bootstrap_lower_min": min(obs.bootstrap_lower for obs in observations),
        "bootstrap_upper_max": max(obs.bootstrap_upper for obs in observations),
        "complete_vector_statistics": True,
        "status": status.value,
        "projection_provenance_present": projection_ok,
        "complete_gram_available": all(obs.u_statistic != 0.0 or obs.total_error_energy == 0.0
                                        for obs in observations),
    }


class BiasFormationTrace:
    """Common-state formation trace; trajectory drift is not an accepted field."""

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
            raise ValueError("case_id and both common-state splits are required")
        if set(self.calibration_state_ids) & set(self.confirmation_state_ids):
            raise ValueError("calibration and confirmation states must be disjoint")
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
        common_state_digest: str | None,
        local_endpoint: Any,
        parameter_gradient: Any,
        effective_update: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        state_id = str(state_id)
        if state_id in self._rows:
            raise ValueError("duplicate state_id: %s" % state_id)
        if state_id not in self.expected_state_ids:
            raise ValueError("state_id is outside the frozen common-state split")
        expected = "calibration" if state_id in self.calibration_state_ids else "confirmation"
        if str(partition) != expected:
            raise ValueError("partition does not match the frozen common-state split")
        layers = {
            FormationLayer.LOCAL_ENDPOINT: local_endpoint,
            FormationLayer.PARAMETER_GRADIENT: parameter_gradient,
            FormationLayer.EFFECTIVE_UPDATE: effective_update,
        }
        normalized: Dict[str, Any] = {}
        for layer, value in layers.items():
            if value is None:
                normalized[layer.value] = None
                continue
            try:
                normalized[layer.value] = VectorObservation.from_value(value, self.policy)
            except (TypeError, ValueError, KeyError):
                normalized[layer.value] = "INVALID"
        self._rows[state_id] = {
            "state_id": state_id,
            "partition": str(partition),
            "common_state_digest": None if common_state_digest is None else str(common_state_digest),
            "layers": normalized,
            "metadata": dict(metadata or {}),
        }

    def _collect(self, layer: FormationLayer, partition: str) -> tuple[list[VectorObservation], list[str], bool, bool]:
        state_ids = self.calibration_state_ids if partition == "calibration" else self.confirmation_state_ids
        observations: list[VectorObservation] = []
        missing: list[str] = []
        invalid = False
        common_state_invalid = False
        for state_id in state_ids:
            row = self._rows.get(state_id)
            if row is None or row["layers"].get(layer.value) is None:
                missing.append(state_id)
                continue
            if row["layers"].get(layer.value) == "INVALID":
                invalid = True
                continue
            if not row.get("common_state_digest"):
                common_state_invalid = True
            observations.append(row["layers"][layer.value])
        return observations, missing, invalid, common_state_invalid

    def _make_layers(self, partition: str) -> tuple[Dict[str, Any], list[str], list[str], list[str]]:
        summaries: Dict[str, Any] = {}
        missing_all: list[str] = []
        invalid_layers: list[str] = []
        invalid_kinds: list[str] = []
        for layer in _LAYER_ORDER:
            observations, missing, invalid, common_invalid = self._collect(layer, partition)
            summary = _layer_summary(layer, observations, self.policy)
            if missing:
                summary["missing_state_ids"] = missing
                summary["status"] = FormationStatus.UNRESOLVED_MISSING_LAYER.value
                missing_all.extend(missing)
            if invalid:
                summary["status"] = FormationStatus.INVALID_NONFINITE.value
                invalid_layers.append(layer.value)
                invalid_kinds.append(FormationStatus.INVALID_NONFINITE.value)
            if common_invalid:
                summary["status"] = FormationStatus.INVALID_COMMON_STATE.value
                invalid_layers.append(layer.value)
                invalid_kinds.append(FormationStatus.INVALID_COMMON_STATE.value)
            if not summary.get("projection_provenance_present", True):
                summary["status"] = FormationStatus.INVALID_PROJECTION.value
                invalid_layers.append(layer.value)
                invalid_kinds.append(FormationStatus.INVALID_PROJECTION.value)
            summaries[layer.value] = summary
        return summaries, sorted(set(missing_all)), invalid_layers, invalid_kinds

    def finalize(self) -> Dict[str, Any]:
        missing_rows = [state_id for state_id in self.expected_state_ids if state_id not in self._rows]
        unexpected_rows = [state_id for state_id in self._rows if state_id not in self.expected_state_ids]
        calibration_layers, calibration_missing, calibration_invalid, calibration_invalid_kinds = self._make_layers("calibration")
        confirmation_layers, confirmation_missing, confirmation_invalid, confirmation_invalid_kinds = self._make_layers("confirmation")
        all_layers = list(confirmation_layers.values())
        all_layer_summaries = list(calibration_layers.values()) + list(confirmation_layers.values())
        invalid = calibration_invalid + confirmation_invalid
        invalid_kinds = calibration_invalid_kinds + confirmation_invalid_kinds
        if missing_rows or calibration_missing or confirmation_missing:
            status = FormationStatus.UNRESOLVED_MISSING_LAYER.value
        elif FormationStatus.INVALID_NONFINITE.value in invalid_kinds:
            status = FormationStatus.INVALID_NONFINITE.value
        elif FormationStatus.INVALID_COMMON_STATE.value in invalid_kinds:
            status = FormationStatus.INVALID_COMMON_STATE.value
        elif FormationStatus.INVALID_PROJECTION.value in invalid_kinds:
            status = FormationStatus.INVALID_PROJECTION.value
        elif any(layer["status"] == FormationStatus.UNRESOLVED_INSUFFICIENT_STATES.value for layer in all_layer_summaries):
            status = FormationStatus.UNRESOLVED_INSUFFICIENT_STATES.value
        elif any(layer["status"] not in {FormationStatus.CENTERED.value, FormationStatus.BIASED.value}
                 for layer in all_layer_summaries):
            status = FormationStatus.UNRESOLVED_INSUFFICIENT_STATES.value
        elif unexpected_rows:
            status = FormationStatus.UNRESOLVED_MISSING_LAYER.value
        elif not all(layer["status"] in {FormationStatus.CENTERED.value, FormationStatus.BIASED.value}
                     for layer in all_layer_summaries):
            status = FormationStatus.UNRESOLVED_INSUFFICIENT_STATES.value
        else:
            status = FormationStatus.COMPLETE.value

        first_confirmed = None
        first_observed = None
        prior_centered = True
        for layer in _LAYER_ORDER:
            current = confirmation_layers[layer.value]["status"]
            if first_observed is None and current == FormationStatus.BIASED.value:
                first_observed = layer.value
            if first_confirmed is None and prior_centered and current == FormationStatus.BIASED.value:
                first_confirmed = layer.value
            if current != FormationStatus.CENTERED.value:
                prior_centered = False
        return {
            "schema": "kernel-analyzer-bias-formation-certificate-v2",
            "case_id": self.case_id,
            "status": status,
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
            "calibration_layers": calibration_layers,
            "confirmation_layers": confirmation_layers,
            "first_confirmed_bias_stage": first_confirmed,
            "first_observed_biased_stage": first_observed,
            "formation_point": (
                "CONFIRMED" if first_confirmed is not None and status == FormationStatus.COMPLETE.value
                else "UNRESOLVED"
            ),
            "trajectory_drift_in_formation": False,
            "missing_rows": sorted(set(missing_rows + calibration_missing + confirmation_missing)),
            "unexpected_rows": unexpected_rows,
            "row_digest": hashlib.sha256(json.dumps(
                self._serializable_rows(), sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "rows": self._serializable_rows(),
        }

    def _serializable_rows(self) -> list[Dict[str, Any]]:
        rows = []
        for state_id in self.expected_state_ids:
            if state_id not in self._rows:
                continue
            row = self._rows[state_id]
            layers: Dict[str, Any] = {}
            for key, value in row["layers"].items():
                if isinstance(value, VectorObservation):
                    layers[key] = value.as_dict()
                else:
                    layers[key] = value
            rows.append({
                "state_id": row["state_id"],
                "partition": row["partition"],
                "common_state_digest": row["common_state_digest"],
                "layers": layers,
                "metadata": row["metadata"],
            })
        return rows
