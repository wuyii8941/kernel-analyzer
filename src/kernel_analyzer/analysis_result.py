"""Backend-neutral result records for training numerical analysis.

This record complements the historical T1--T4 certificate.  A valid analysis
does not require a positive bias result or a measured training consequence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AnalysisResult:
    case_id: str
    contrast_id: str
    measurement_status: str
    claim_scope: str
    bias_analysis: Mapping[str, Any] = field(default_factory=dict)
    equivalence_decision: str = "NOT_ASSESSED"
    training_outcome: str = "NOT_MEASURED"
    mandatory_endpoints: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if self.measurement_status not in {"VALID", "INVALID", "PARTIAL"}:
            raise ValueError("unknown measurement_status")
        if self.equivalence_decision not in {
            "EQUIVALENT", "NON_EQUIVALENT", "INCONCLUSIVE", "NOT_ASSESSED",
        }:
            raise ValueError("unknown equivalence_decision")
        result = asdict(self)
        result["schema"] = "kernel-analyzer-analysis-result-v1"
        result["mandatory_endpoints"] = list(self.mandatory_endpoints)
        return result
