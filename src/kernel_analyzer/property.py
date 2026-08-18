"""Reference/schedule-derived properties for directional F+B bias.

This module deliberately separates quantities available before a candidate
verdict from measurements that only confirm a mechanism after candidate
values are observed.  It does not encode an operator-name classifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import struct
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .statistics import coherence_certificate


class Hypothesis(str, Enum):
    ARITHMETIC_EVENT_ASYMMETRY = "ARITHMETIC_EVENT_ASYMMETRY"
    CARRIER_GEOMETRY = "CARRIER_GEOMETRY"
    VJP_TRANSPORT_SUSCEPTIBILITY = "VJP_TRANSPORT_SUSCEPTIBILITY"
    ARITHMETIC_CARRIER_COUPLING = "ARITHMETIC_CARRIER_COUPLING"
    TEMPORAL_PERSISTENCE = "TEMPORAL_PERSISTENCE"
    SIGNED_TRANSPORT_COHERENCE = "SIGNED_TRANSPORT_COHERENCE"


class EvidenceStage(str, Enum):
    PREDICTOR = "PREDICTOR"
    CONFIRMATORY = "CONFIRMATORY"
    CONSEQUENCE = "CONSEQUENCE"


class EvidenceStatus(str, Enum):
    SUPPORT = "SUPPORT"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CaseRole(str, Enum):
    COHERENT_FB_BIAS = "COHERENT_FB_BIAS"
    NORMAL_REFERENCE = "NORMAL_REFERENCE"
    SEMANTIC_REGION_SUPPORT = "SEMANTIC_REGION_SUPPORT"


HYPOTHESIS_STAGES = {
    Hypothesis.ARITHMETIC_EVENT_ASYMMETRY: EvidenceStage.PREDICTOR,
    Hypothesis.CARRIER_GEOMETRY: EvidenceStage.PREDICTOR,
    Hypothesis.VJP_TRANSPORT_SUSCEPTIBILITY: EvidenceStage.PREDICTOR,
    Hypothesis.ARITHMETIC_CARRIER_COUPLING: EvidenceStage.CONFIRMATORY,
    Hypothesis.TEMPORAL_PERSISTENCE: EvidenceStage.CONSEQUENCE,
    Hypothesis.SIGNED_TRANSPORT_COHERENCE: EvidenceStage.PREDICTOR,
}

# Predictor features must be candidate-blind and verdict-blind.  These names
# are rejected recursively, including common plural and suffix variants.
FORBIDDEN_PREDICTOR_KEYS = frozenset({
    "candidate_output", "candidate_outputs", "candidate_tensor",
    "candidate_value", "candidate_values", "oracle_verdict", "case_verdict",
    "historical_verdict", "history_verdict", "operator_name", "op_name",
    "module_name", "model_name", "source_path", "candidate_id",
    "t1_verdict", "t2_verdict", "t3_verdict", "t4_verdict",
    "observed_carrier", "observed_gradient_error", "accumulated",
})


def _normalise_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _forbidden_predictor_key(key: str) -> bool:
    if key in FORBIDDEN_PREDICTOR_KEYS:
        return True
    if key.endswith("_verdict"):
        return True
    if key.startswith(("t1_", "t2_", "t3_", "t4_", "observed_t1_",
                       "observed_t2_", "observed_t3_", "observed_t4_")):
        return True
    if key.startswith(("candidate_output", "candidate_tensor", "candidate_value")):
        return True
    return key in {"operator", "operator_family", "model", "module"}


def predictor_leaks(value: Any, prefix: str = "") -> Tuple[str, ...]:
    """Return forbidden paths found in a nested predictor feature object."""
    leaks = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normal = _normalise_key(key)
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            if _forbidden_predictor_key(normal):
                leaks.append(path)
            leaks.extend(predictor_leaks(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            path = "%s[%d]" % (prefix, index)
            leaks.extend(predictor_leaks(nested, path))
    return tuple(leaks)


def validate_predictor_features(features: Mapping[str, Any]) -> None:
    leaks = predictor_leaks(features)
    if leaks:
        raise ValueError("candidate/verdict identity leaked into predictor: %s" %
                         ", ".join(leaks))


@dataclass(frozen=True)
class HypothesisEvidence:
    hypothesis: Hypothesis
    status: EvidenceStatus
    measurements: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    evidence_paths: Sequence[str] = field(default_factory=tuple)
    stage: Optional[EvidenceStage] = None

    def __post_init__(self) -> None:
        expected = HYPOTHESIS_STAGES[self.hypothesis]
        actual = self.stage if self.stage is not None else expected
        if actual != expected:
            raise ValueError("%s evidence must be %s, not %s" %
                             (self.hypothesis.value, expected.value, actual.value))
        object.__setattr__(self, "stage", actual)
        if actual == EvidenceStage.PREDICTOR:
            validate_predictor_features(self.measurements)

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["hypothesis"] = self.hypothesis.value
        row["status"] = self.status.value
        row["stage"] = self.stage.value if self.stage is not None else None
        row["evidence_paths"] = list(self.evidence_paths)
        return row


@dataclass(frozen=True)
class PropertyCase:
    case_id: str
    role: CaseRole
    mechanism_level: str
    arithmetic_mechanism: str
    flash_style_verdict: str
    evidence: Sequence[HypothesisEvidence]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if (self.role == CaseRole.COHERENT_FB_BIAS and
                self.mechanism_level == "CLOSED_SEMANTIC_REGION"):
            raise ValueError("semantic regions cannot count as root property positives")
        hypotheses = [row.hypothesis for row in self.evidence]
        if len(hypotheses) != len(set(hypotheses)):
            raise ValueError("duplicate hypothesis evidence in %s" % self.case_id)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role.value,
            "mechanism_level": self.mechanism_level,
            "arithmetic_mechanism": self.arithmetic_mechanism,
            "flash_style_verdict": self.flash_style_verdict,
            "hypotheses": [row.as_dict() for row in self.evidence],
        }


def effective_rank(eigenvalues: Sequence[float]) -> float:
    """Participation-ratio effective rank for nonnegative spectrum values."""
    values = [float(value) for value in eigenvalues]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("effective rank requires finite nonnegative eigenvalues")
    total = math.fsum(values)
    squared = math.fsum(value * value for value in values)
    if total == 0.0:
        return 0.0
    return total * total / squared


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors have different dimensions")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def accumulation_decomposition(vectors: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Exact finite-sample energy decomposition, used only as a lemma.

    ||sum_t d_t||^2 = sum_t ||d_t||^2 + 2 sum_{s<t} <d_s,d_t>.
    This identity detects coherent accumulation but does not explain its cause.
    """
    if not vectors:
        raise ValueError("at least one vector is required")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(row) != dimension for row in vectors):
        raise ValueError("vectors must have one common nonzero dimension")
    if any(not math.isfinite(float(x)) for row in vectors for x in row):
        raise ValueError("vectors must be finite")
    diagonal = math.fsum(_dot(row, row) for row in vectors)
    cross = 2.0 * math.fsum(
        _dot(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )
    summed = [math.fsum(float(row[index]) for row in vectors)
              for index in range(dimension)]
    total = _dot(summed, summed)
    return {"diagonal_energy": diagonal, "cross_energy": cross,
            "total_energy": total, "residual": total - diagonal - cross}


def deterministic_conditional_rounding_mean(rounding_error: float) -> float:
    """E[epsilon | exact deterministic operands] for deterministic rounding.

    With no stochastic rounding variable, conditioning on the exact operands
    leaves a point mass.  The conditional mean is epsilon itself, so requiring
    it to be zero merely requires exact representability and is not a useful
    discriminator among inexact deterministic operations.
    """
    value = float(rounding_error)
    if not math.isfinite(value):
        raise ValueError("rounding_error must be finite")
    return value


def round_fp32_to_bf16(value: float) -> float:
    """Round one finite/infinite FP32 value to BF16 using ties-to-even.

    The returned Python float is the exact FP32 embedding of the BF16 bit
    pattern.  This is a schedule-derived value, not an observed candidate
    tensor value.
    """
    fp32 = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    exponent = fp32 & 0x7F800000
    fraction = fp32 & 0x007FFFFF
    if exponent == 0x7F800000:
        # Preserve infinities.  Preserve NaNs as NaNs and retain a nonzero
        # payload after truncation even when all high payload bits were zero.
        if fraction and not (fp32 & 0x007F0000):
            fp32 |= 0x00010000
        rounded = fp32 & 0xFFFF0000
    else:
        rounded = (fp32 + 0x7FFF + ((fp32 >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", rounded))[0]


def derive_signed_rounding_error(exact_result: float, declared_dtype: str) -> float:
    """Derive a local rounding residual from reference data and a schedule.

    ``exact_result`` is the reference evaluation of one arithmetic event.
    Only BF16 final rounding is implemented in v1; unsupported arithmetic is
    fail-closed instead of silently approximated.
    """
    value = float(exact_result)
    if not math.isfinite(value):
        raise ValueError("exact_result must be finite")
    dtype = _normalise_key(declared_dtype)
    if dtype in {"bf16", "bfloat16", "torch_bfloat16"}:
        return round_fp32_to_bf16(value) - value
    if dtype in {"fp32", "float32", "torch_float32"}:
        return 0.0
    raise ValueError("unsupported declared arithmetic dtype: %s" % declared_dtype)


def signed_event_transport(
    event_errors: Sequence[float],
    reference_transport_directions: Sequence[Sequence[float]],
) -> Tuple[float, ...]:
    """Compute ``sum_e epsilon_e J_e`` for one state.

    Each transport direction is the complete reference-only F+B response of
    the declared parameter-gradient coordinates to one scalar arithmetic
    event.  Vector arithmetic events are represented by flattened scalar
    events so no coordinate can be selected post hoc.
    """
    if not event_errors or len(event_errors) != len(reference_transport_directions):
        raise ValueError("every arithmetic event needs one transport direction")
    dimension = len(reference_transport_directions[0])
    if dimension == 0 or any(len(row) != dimension
                             for row in reference_transport_directions):
        raise ValueError("transport directions need one common nonzero dimension")
    values = [float(value) for value in event_errors]
    directions = [[float(value) for value in row]
                  for row in reference_transport_directions]
    if any(not math.isfinite(value) for value in values) or any(
            not math.isfinite(value) for row in directions for value in row):
        raise ValueError("event errors and transport directions must be finite")
    return tuple(math.fsum(values[event] * directions[event][coordinate]
                           for event in range(len(values)))
                 for coordinate in range(dimension))


@dataclass(frozen=True)
class SignedTransportState:
    """Compact leading-order F+B transport for one frozen natural state."""

    state_id: str
    transported_error: Sequence[float]
    nonlinear_remainder_bound: float = 0.0

    def __post_init__(self) -> None:
        if not self.state_id or not self.transported_error:
            raise ValueError("state_id and transported_error are required")
        if any(not math.isfinite(float(value)) for value in self.transported_error):
            raise ValueError("transported_error must be finite")
        remainder = float(self.nonlinear_remainder_bound)
        if not math.isfinite(remainder) or remainder < 0.0:
            raise ValueError("nonlinear remainder bound must be finite and nonnegative")


def signed_transport_certificate(
    states: Sequence[SignedTransportState], *, reference_margin: float,
    alpha: float = 0.05, bootstrap_samples: int = 2000, seed: int = 0,
) -> Dict[str, Any]:
    """Certify candidate-value-blind Signed Transport Coherence.

    The leading vectors must be computed solely from reference operands, the
    declared arithmetic schedule and complete analytic F+B transport.  The
    mean remainder bound makes the nonzero-mean claim conservative for
    nonlinear regions.  No endpoint identity or observed T1--T4 value enters
    this calculation.
    """
    margin = float(reference_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("reference_margin must be finite and nonnegative")
    if not states:
        raise ValueError("at least one state is required")
    ids = [row.state_id for row in states]
    if len(ids) != len(set(ids)):
        raise ValueError("state IDs must be unique")
    vectors = [tuple(float(value) for value in row.transported_error)
               for row in states]
    dimension = len(vectors[0])
    if any(len(row) != dimension for row in vectors):
        raise ValueError("transported errors have different dimensions")

    count = len(vectors)
    mean = tuple(math.fsum(row[index] for row in vectors) / count
                 for index in range(dimension))
    amplitude = math.fsum(_dot(row, row) for row in vectors) / count
    directional_energy = _dot(mean, mean)
    concentration = directional_energy / amplitude if amplitude else 0.0
    mean_remainder = math.fsum(float(row.nonlinear_remainder_bound)
                               for row in states) / count
    certified_mean_magnitude = max(
        0.0, math.sqrt(max(0.0, directional_energy)) - mean_remainder
    )
    coherence = coherence_certificate(
        vectors, alpha=alpha, bootstrap_samples=bootstrap_samples, seed=seed
    )
    if coherence["status"].startswith("UNRESOLVED"):
        status = "ABSTAIN_UNRESOLVED_COHERENCE"
    elif (coherence["status"] == "PASS" and
          certified_mean_magnitude > margin):
        status = "PREDICTED_COHERENT_F_B_BIAS"
    else:
        status = "NO_PREDICTED_COHERENT_F_B_BIAS"
    return {
        "schema": "kernel-analyzer-signed-transport-certificate-v1",
        "status": status,
        "state_count": count,
        "coordinate_count": dimension,
        "amplitude": amplitude,
        "directional_energy": directional_energy,
        "concentration": concentration,
        "mean_nonlinear_remainder_bound": mean_remainder,
        "certified_mean_magnitude": certified_mean_magnitude,
        "reference_margin": margin,
        "margin_excess": certified_mean_magnitude - margin,
        "coherence": coherence,
        "candidate_tensor_values_read": False,
        "complete_coordinates_required": True,
    }


def select_property(cases: Sequence[PropertyCase]) -> Dict[str, Any]:
    """Apply conservative, mechanism-level selection criteria.

    A winner needs coherent F+B support from at least two distinct declared
    arithmetic mechanisms, separation of all completed normal references, and
    no unresolved root evidence.  Confirmatory and consequence axes are never
    promoted to predictors.
    """
    positives = [row for row in cases if row.role == CaseRole.COHERENT_FB_BIAS]
    controls = [row for row in cases if row.role == CaseRole.NORMAL_REFERENCE]
    assessments = []
    winners = []
    for hypothesis in Hypothesis:
        stage = HYPOTHESIS_STAGES[hypothesis]
        by_case = {
            row.case_id: next((item for item in row.evidence
                               if item.hypothesis == hypothesis), None)
            for row in positives + controls
        }
        positive_support = [row.case_id for row in positives
                            if by_case[row.case_id] is not None and
                            by_case[row.case_id].status == EvidenceStatus.SUPPORT]
        control_separation = [row.case_id for row in controls
                              if by_case[row.case_id] is not None and
                              by_case[row.case_id].status == EvidenceStatus.COUNTEREXAMPLE]
        unresolved = [case_id for case_id, evidence in by_case.items()
                      if evidence is None or evidence.status == EvidenceStatus.UNRESOLVED]
        mechanisms = sorted({row.arithmetic_mechanism for row in positives
                             if row.case_id in positive_support})
        reasons = []
        if stage != EvidenceStage.PREDICTOR:
            reasons.append("axis_is_%s_not_a_pre_verdict_predictor" % stage.value.lower())
        if len(mechanisms) < 2:
            reasons.append("fewer_than_two_distinct_arithmetic_mechanisms")
        if len(control_separation) != len(controls):
            reasons.append("does_not_separate_all_normal_references")
        if unresolved:
            reasons.append("unresolved_root_case_evidence")
        eligible = not reasons
        if eligible:
            winners.append(hypothesis.value)
        assessments.append({
            "hypothesis": hypothesis.value,
            "stage": stage.value,
            "positive_support": positive_support,
            "normal_reference_separation": control_separation,
            "unresolved_root_cases": unresolved,
            "distinct_supported_mechanisms": mechanisms,
            "eligible": eligible,
            "rejection_reasons": reasons,
        })
    return {
        "status": ("SUPPORTED_PROPERTY_CANDIDATES" if winners else
                   "NO_SINGLE_MATHEMATICAL_PROPERTY_SUPPORTED_YET"),
        "winners": winners,
        "assessments": assessments,
    }
