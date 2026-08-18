"""Candidate-blind bias formation traces.

The project studies *where* a local forward/backward difference first stops
being centred.  This module is intentionally a small, tensor-free protocol
layer: runners may compute the summaries from full tensors, but the persisted
trace contains only declared scalar projections, norms, and digests.

No candidate verdict, operator identity, or SEUP result is consumed here.  A
trace can therefore be generated before any property tournament label exists.
The six property names are hypotheses, not a ranking or a classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


class BiasProperty(str, Enum):
    CONDITIONAL_SOURCE_ASYMMETRY = "P1_CONDITIONAL_SOURCE_ASYMMETRY"
    SOURCE_TRANSPORT_ALIGNMENT = "P2_SOURCE_TRANSPORT_ALIGNMENT"
    FB_NUMERICAL_CONSISTENCY = "P3_FB_NUMERICAL_CONSISTENCY"
    NONLINEAR_RECTIFICATION = "P4_NONLINEAR_RECTIFICATION"
    OPTIMIZER_RECTIFICATION = "P5_OPTIMIZER_RECTIFICATION"
    SEMANTIC_ORBIT_CENTERING = "P6_SEMANTIC_ORBIT_CENTERING"


class BiasLayer(str, Enum):
    LOCAL_ENDPOINT = "LOCAL_ENDPOINT"
    PARAMETER_GRADIENT = "PARAMETER_GRADIENT"
    EFFECTIVE_UPDATE = "EFFECTIVE_UPDATE"
    TRAJECTORY_DRIFT = "TRAJECTORY_DRIFT"


class BiasStatus(str, Enum):
    CENTERED = "CENTERED"
    BIASED = "BIASED"
    NONFINITE = "NONFINITE"
    UNRESOLVED = "UNRESOLVED"


class PropertyDecision(str, Enum):
    SUPPORTED_CROSS_MECHANISM_PROPERTY = "SUPPORTED_CROSS_MECHANISM_PROPERTY"
    SUPPORTED_CASE_SPECIFIC_MECHANISM = "SUPPORTED_CASE_SPECIFIC_MECHANISM"
    COUNTEREXAMPLE_FOUND = "COUNTEREXAMPLE_FOUND"
    UNRESOLVED = "UNRESOLVED"


_LAYER_ORDER = (
    BiasLayer.LOCAL_ENDPOINT,
    BiasLayer.PARAMETER_GRADIENT,
    BiasLayer.EFFECTIVE_UPDATE,
    BiasLayer.TRAJECTORY_DRIFT,
)


@dataclass(frozen=True)
class BiasTracePolicy:
    """Frozen engineering gates for descriptive transition certificates.

    These thresholds decide whether a finite trace is *reportable* as centred
    or biased.  They do not define a property and must not be tuned against a
    property outcome.  The default is deliberately conservative and can be
    overridden only by a versioned protocol file.
    """

    min_states: int = 4
    centred_z_abs_max: float = 2.0
    persistence_ratio_min: float = 0.6
    mean_norm_floor: float = 1e-30

    def __post_init__(self) -> None:
        if self.min_states < 2:
            raise ValueError("min_states must be at least two")
        if self.centred_z_abs_max <= 0 or self.persistence_ratio_min <= 0:
            raise ValueError("bias gates must be positive")
        if self.persistence_ratio_min > 1:
            raise ValueError("persistence_ratio_min cannot exceed one")
        if self.mean_norm_floor <= 0:
            raise ValueError("mean_norm_floor must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_states": self.min_states,
            "centred_z_abs_max": self.centred_z_abs_max,
            "persistence_ratio_min": self.persistence_ratio_min,
            "mean_norm_floor": self.mean_norm_floor,
        }


@dataclass(frozen=True)
class LayerValue:
    """A scalar projection and its scale for one state.

    ``signed_value`` is a declared projection (for example an event mean or a
    frozen carrier projection).  It is not silently inferred from a candidate
    tensor.  ``norm`` is the corresponding nonnegative residual/update scale.
    """

    signed_value: float
    norm: float
    support_count: int = 1
    digest: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.signed_value)):
            raise ValueError("signed_value must be finite")
        if not math.isfinite(float(self.norm)) or self.norm < 0:
            raise ValueError("norm must be finite and nonnegative")
        if int(self.support_count) < 1:
            raise ValueError("support_count must be positive")

    @classmethod
    def from_value(cls, value: Any) -> "LayerValue":
        if isinstance(value, LayerValue):
            return value
        if isinstance(value, Mapping):
            return cls(
                signed_value=float(value["signed_value"]),
                norm=float(value.get("norm", abs(float(value["signed_value"]))),),
                support_count=int(value.get("support_count", 1)),
                digest=None if value.get("digest") is None else str(value["digest"]),
            )
        scalar = float(value)
        return cls(signed_value=scalar, norm=abs(scalar))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "signed_value": float(self.signed_value),
            "norm": float(self.norm),
            "support_count": int(self.support_count),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class LayerCertificate:
    layer: BiasLayer
    state_count: int
    signed_mean: float
    signed_variance: float
    signed_skew: float | None
    mean_to_norm: float
    signed_persistence: float
    status: BiasStatus
    missing_state_ids: Sequence[str] = field(default_factory=tuple)
    nonfinite_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "state_count": self.state_count,
            "signed_mean": self.signed_mean,
            "signed_variance": self.signed_variance,
            "signed_skew": self.signed_skew,
            "mean_to_norm": self.mean_to_norm,
            "signed_persistence": self.signed_persistence,
            "status": self.status.value,
            "missing_state_ids": list(self.missing_state_ids),
            "nonfinite_count": self.nonfinite_count,
        }


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _variance(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _skew(values: Sequence[float], mean: float, variance: float) -> float | None:
    if len(values) < 3 or variance <= 0:
        return None
    scale = math.sqrt(variance)
    return math.fsum(((value - mean) / scale) ** 3 for value in values) / len(values)


def _persistence(values: Sequence[float]) -> float:
    denominator = math.fsum(abs(value) for value in values)
    if denominator == 0:
        return 0.0
    return abs(math.fsum(values)) / denominator


def _digest_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class BiasTrace:
    """Collect and certify the four transitions for one case.

    ``calibration_state_ids`` and ``evaluation_state_ids`` are explicit.  A
    duplicate, missing, or out-of-split state fails closed.  Each row must
    provide all four layers; a runner that cannot provide a layer must record
    ``None`` and receives ``UNRESOLVED`` rather than silently dropping it.
    """

    def __init__(
        self,
        case_id: str,
        calibration_state_ids: Sequence[str],
        evaluation_state_ids: Sequence[str],
        policy: BiasTracePolicy | None = None,
    ) -> None:
        self.case_id = str(case_id)
        self.calibration_state_ids = tuple(str(x) for x in calibration_state_ids)
        self.evaluation_state_ids = tuple(str(x) for x in evaluation_state_ids)
        if not self.case_id or not self.calibration_state_ids or not self.evaluation_state_ids:
            raise ValueError("case_id and both state splits are required")
        if set(self.calibration_state_ids) & set(self.evaluation_state_ids):
            raise ValueError("calibration and evaluation states must be disjoint")
        self.policy = policy or BiasTracePolicy()
        self._rows: MutableMapping[str, Dict[str, Any]] = {}

    @property
    def expected_state_ids(self) -> tuple[str, ...]:
        return self.calibration_state_ids + self.evaluation_state_ids

    def add(
        self,
        state_id: str,
        partition: str,
        *,
        local_endpoint: LayerValue | Mapping[str, Any] | float | None,
        parameter_gradient: LayerValue | Mapping[str, Any] | float | None,
        effective_update: LayerValue | Mapping[str, Any] | float | None,
        trajectory_drift: LayerValue | Mapping[str, Any] | float | None,
        feedback: LayerValue | Mapping[str, Any] | float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        state_id = str(state_id)
        if state_id in self._rows:
            raise ValueError("duplicate state_id: %s" % state_id)
        if state_id not in self.expected_state_ids:
            raise ValueError("state_id is outside the frozen split: %s" % state_id)
        partition = str(partition)
        if partition not in {"calibration", "evaluation"}:
            raise ValueError("partition must be calibration or evaluation")
        expected_partition = (
            "calibration" if state_id in self.calibration_state_ids else "evaluation"
        )
        if partition != expected_partition:
            raise ValueError("partition does not match frozen state split")
        values = {
            BiasLayer.LOCAL_ENDPOINT: local_endpoint,
            BiasLayer.PARAMETER_GRADIENT: parameter_gradient,
            BiasLayer.EFFECTIVE_UPDATE: effective_update,
            BiasLayer.TRAJECTORY_DRIFT: trajectory_drift,
        }
        normalized: Dict[str, Any] = {}
        for layer, value in values.items():
            if value is None:
                normalized[layer.value] = None
                continue
            try:
                normalized[layer.value] = LayerValue.from_value(value).as_dict()
            except (TypeError, ValueError, KeyError):
                # Preserve the state slot and fail closed at certification;
                # malformed/nonfinite data must not become an implicit zero.
                normalized[layer.value] = {"invalid": True}
        try:
            normalized_feedback = (
                None if feedback is None else LayerValue.from_value(feedback).as_dict()
            )
        except (TypeError, ValueError, KeyError):
            normalized_feedback = {"invalid": True}
        self._rows[state_id] = {
            "state_id": state_id,
            "partition": partition,
            "layers": normalized,
            "feedback": normalized_feedback,
            "metadata": dict(metadata or {}),
        }

    def _layer_certificate(self, layer: BiasLayer, partition: str) -> LayerCertificate:
        expected = (
            self.calibration_state_ids if partition == "calibration" else self.evaluation_state_ids
        )
        values: list[float] = []
        norms: list[float] = []
        missing: list[str] = []
        invalid_count = 0
        for state_id in expected:
            row = self._rows.get(state_id)
            value = None if row is None else row["layers"].get(layer.value)
            if value is None:
                missing.append(state_id)
                continue
            if isinstance(value, Mapping) and value.get("invalid"):
                invalid_count += 1
                continue
            try:
                signed = float(value["signed_value"])
                norm = float(value["norm"])
                if not math.isfinite(signed) or not math.isfinite(norm):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                missing.append(state_id)
                continue
            values.append(signed)
            norms.append(norm)
        if invalid_count:
            return LayerCertificate(
                layer=layer,
                state_count=len(values),
                signed_mean=0.0,
                signed_variance=0.0,
                signed_skew=None,
                mean_to_norm=0.0,
                signed_persistence=0.0,
                status=BiasStatus.NONFINITE,
                missing_state_ids=tuple(missing),
                nonfinite_count=invalid_count,
            )
        if missing or len(values) < self.policy.min_states:
            return LayerCertificate(
                layer=layer,
                state_count=len(values),
                signed_mean=0.0,
                signed_variance=0.0,
                signed_skew=None,
                mean_to_norm=0.0,
                signed_persistence=0.0,
                status=BiasStatus.UNRESOLVED,
                missing_state_ids=tuple(missing),
            )
        mean = _mean(values)
        variance = _variance(values, mean)
        norm_mean = max(_mean(norms), self.policy.mean_norm_floor)
        ratio = abs(mean) / norm_mean
        z = abs(mean) / max(math.sqrt(variance / len(values)), self.policy.mean_norm_floor)
        status = BiasStatus.BIASED if z > self.policy.centred_z_abs_max else BiasStatus.CENTERED
        return LayerCertificate(
            layer=layer,
            state_count=len(values),
            signed_mean=mean,
            signed_variance=variance,
            signed_skew=_skew(values, mean, variance),
            mean_to_norm=ratio,
            signed_persistence=_persistence(values),
            status=status,
        )

    def finalize(self) -> Dict[str, Any]:
        missing_rows = [state_id for state_id in self.expected_state_ids if state_id not in self._rows]
        unexpected_rows = [state_id for state_id in self._rows if state_id not in self.expected_state_ids]
        calibration_certificates: Dict[str, Dict[str, Any]] = {}
        evaluation_certificates: Dict[str, Dict[str, Any]] = {}
        calibration_first_bias_layer: str | None = None
        first_bias_layer: str | None = None
        for layer in _LAYER_ORDER:
            calibration = self._layer_certificate(layer, "calibration")
            evaluation = self._layer_certificate(layer, "evaluation")
            calibration_certificates[layer.value] = calibration.as_dict()
            evaluation_certificates[layer.value] = evaluation.as_dict()
            if calibration_first_bias_layer is None and calibration.status == BiasStatus.BIASED:
                calibration_first_bias_layer = layer.value
            if first_bias_layer is None and evaluation.status == BiasStatus.BIASED:
                first_bias_layer = layer.value
        # ``layers`` remains an evaluation alias for compact consumers.
        layer_certificates = evaluation_certificates
        drift = evaluation_certificates[BiasLayer.TRAJECTORY_DRIFT.value]
        persistent = (
            drift["status"] == BiasStatus.BIASED.value
            and drift["signed_persistence"] >= self.policy.persistence_ratio_min
        )
        status = "COMPLETE" if not missing_rows and not unexpected_rows else "UNRESOLVED"
        return {
            "schema": "kernel-analyzer-bias-transition-certificate-v1",
            "case_id": self.case_id,
            "status": status,
            "state_split": {
                "calibration_state_ids": list(self.calibration_state_ids),
                "evaluation_state_ids": list(self.evaluation_state_ids),
                "calibration_count": len(self.calibration_state_ids),
                "evaluation_count": len(self.evaluation_state_ids),
                "disjoint": not bool(set(self.calibration_state_ids) & set(self.evaluation_state_ids)),
            },
            "policy": self.policy.as_dict(),
            "missing_rows": missing_rows,
            "unexpected_rows": unexpected_rows,
            "calibration_layers": calibration_certificates,
            "evaluation_layers": evaluation_certificates,
            "layers": layer_certificates,
            "calibration_first_noncentered_layer": calibration_first_bias_layer,
            "first_noncentered_layer": first_bias_layer,
            "temporal": {
                "status": "PERSISTENT" if persistent else "TEMPORALLY_CANCELING_OR_UNRESOLVED",
                "signed_persistence": drift["signed_persistence"],
                "uses_t4_or_final_drift_label": False,
            },
            "feedback": {
                "declared": any(row.get("feedback") is not None for row in self._rows.values()),
                "not_used_to_fit_layer_status": True,
            },
            "row_digest": _digest_rows([self._rows[state_id] for state_id in self._rows]),
            "rows": [self._rows[state_id] for state_id in self.expected_state_ids if state_id in self._rows],
        }


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trace input must contain a JSON object")
    return value
