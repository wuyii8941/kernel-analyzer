"""Reference-only predictor interface for future zero-shot property work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


FORBIDDEN_REFERENCE_KEYS = frozenset({
    "candidate", "candidate_output", "candidate_residual", "repair", "repair_output",
    "t4_verdict", "seup_verdict", "historical_verdict", "oracle_verdict",
    "trajectory_drift", "effective_update_residual", "gradient_residual",
})


def _find_forbidden(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text.lower() in FORBIDDEN_REFERENCE_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_forbidden(child, f"{path}[{index}]") )
    return found


@dataclass(frozen=True)
class ReferenceOnlyInputs:
    reference_operands: Mapping[str, Any]
    declared_arithmetic_schedule: Mapping[str, Any]
    analytic_fb_transport: Mapping[str, Any]
    semantic_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        payload = {
            "reference_operands": self.reference_operands,
            "declared_arithmetic_schedule": self.declared_arithmetic_schedule,
            "analytic_fb_transport": self.analytic_fb_transport,
            "semantic_metadata": self.semantic_metadata,
        }
        forbidden = _find_forbidden(payload)
        if forbidden:
            raise ValueError("reference-only inputs contain forbidden candidate-derived fields: " + ", ".join(forbidden))


class ReferenceOnlyPredictor(Protocol):
    """Implementations may predict a property without seeing candidate tensors."""

    def predict(self, inputs: ReferenceOnlyInputs) -> Mapping[str, Any]:
        ...


def validate_reference_payload(payload: Mapping[str, Any]) -> None:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise ValueError("candidate-derived fields are forbidden in zero-shot predictor payload: " + ", ".join(forbidden))
