"""Stable public interfaces for kernel-analyzer plugins and reports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


PASS = "PASS"
FAIL = "FAIL"
UNRESOLVED = "UNRESOLVED"
BLOCKED = "BLOCKED"
VALID_STATUSES = {PASS, FAIL, UNRESOLVED, BLOCKED}
TIERS = ("T1_LOCAL", "T2_CAUSAL", "T3_COHERENT", "T4_ACCUMULATION")


@dataclass(frozen=True)
class ConcreteFBProof:
    """Invocation-specific proof that an executed backward realizes a VJP."""

    saved_tensor_origins_exact: bool
    cotangent_edge_exact: bool
    backward_program_matches_analytic_vjp: bool
    non_tensor_arguments_exact: bool
    output_edges_exact: bool
    forward_program_sha256: str
    backward_program_sha256: str
    analytic_derivation_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "forward_program_sha256", "backward_program_sha256",
            "analytic_derivation_sha256",
        ):
            value = getattr(self, name)
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("%s must be a SHA-256 digest" % name)

    @property
    def complete(self) -> bool:
        return all((
            self.saved_tensor_origins_exact,
            self.cotangent_edge_exact,
            self.backward_program_matches_analytic_vjp,
            self.non_tensor_arguments_exact,
            self.output_edges_exact,
        ))

    def as_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "complete": self.complete}


@dataclass(frozen=True)
class AnalysisState:
    state_id: str
    payload: Any = None
    role: str = "DISCOVERY"

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if self.role not in {"DISCOVERY", "CONFIRMATION", "TRAJECTORY", "CONTROL"}:
            raise ValueError("unsupported state role: %s" % self.role)


@dataclass(frozen=True)
class StepExecution:
    """One concrete scalar-loss forward/backward execution."""

    loss_closure: Callable[[], Any]
    endpoint_closure: Callable[[], Mapping[str, Any]]
    reset: Optional[Callable[[], None]] = None


@dataclass(frozen=True)
class ResourceBudget:
    writable_root: Path
    min_free_bytes: int = 0
    max_artifact_bytes: Optional[int] = None
    max_vram_bytes: Optional[int] = None

    def validate(self, output_dir: Path) -> None:
        root = self.writable_root.resolve()
        output = output_dir.resolve()
        if root != output and root not in output.parents:
            raise ValueError("output_dir is outside the declared writable root")
        import shutil
        if shutil.disk_usage(str(root)).free < self.min_free_bytes:
            raise RuntimeError("resource preflight: insufficient free disk")


@dataclass(frozen=True)
class TierEvidence:
    status: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError("unsupported tier status: %s" % self.status)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseCertificate:
    case_id: str
    proof_unit_id: str
    candidate_id: str
    tiers: Mapping[str, TierEvidence]
    classification: str
    natural: bool = True
    cause_axis: str = "UNATTRIBUTED"

    def __post_init__(self) -> None:
        missing = set(TIERS) - set(self.tiers)
        if missing:
            raise ValueError("case certificate lacks tiers: %s" % sorted(missing))
        ordered_pass = True
        for tier in TIERS:
            status = self.tiers[tier].status
            if status == PASS and not ordered_pass:
                raise ValueError("later tier passes after an earlier non-pass: %s" % tier)
            ordered_pass = ordered_pass and status == PASS
        complete = all(self.tiers[tier].status == PASS for tier in TIERS)
        if self.classification == "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE" and not complete:
            raise ValueError("complete case requires T1-T4 pass")
        if self.cause_axis not in {
            "PRECISION", "OPTIMIZATION", "MIXED", "OTHER", "UNATTRIBUTED"
        }:
            raise ValueError("unsupported case cause axis: %s" % self.cause_axis)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "proof_unit_id": self.proof_unit_id,
            "candidate_id": self.candidate_id,
            "natural": self.natural,
            "cause_axis": self.cause_axis,
            "classification": self.classification,
            "tiers": {tier: self.tiers[tier].as_dict() for tier in TIERS},
        }


@dataclass(frozen=True)
class ReferenceAnalysis:
    subject: str
    proof_units: Sequence[Mapping[str, Any]]
    census: Mapping[str, Any]
    templates: Sequence[Mapping[str, Any]] = ()
    unresolved: Sequence[Mapping[str, Any]] = ()
    case_targets: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class CandidateCensus:
    candidate_id: str
    runtime_regions: Sequence[Mapping[str, Any]]
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisContext:
    spec: "AnalysisSpec"
    reference: ReferenceAnalysis
    run_dir: Path


class ReferenceProvider(ABC):
    @abstractmethod
    def analyze(self, spec: "AnalysisSpec", run_dir: Path) -> ReferenceAnalysis:
        """Return an execution-derived reference F+B atlas."""


class CandidateBackend(ABC):
    candidate_id: str

    @abstractmethod
    def census(self, context: AnalysisContext) -> CandidateCensus:
        """Capture the candidate runtime-region denominator."""

    def predict_signed_transport(
        self, context: AnalysisContext, proof_units: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, TierEvidence]:
        """Optionally predict F+B bias without candidate tensor values.

        Implementations derive signed arithmetic events from reference operands
        plus the declared schedule and transport them through the analytic F+B
        program.  The default accounts for every unit but abstains, so adding
        property prediction does not silently reduce the denominator.
        """
        rows = {}
        for unit in proof_units:
            unit_id = unit.get("unit_id", unit.get("proof_unit_id"))
            if not unit_id:
                raise ValueError("proof unit lacks a stable ID")
            rows[str(unit_id)] = TierEvidence(
                status=UNRESOLVED,
                reason="backend has no reference/schedule factor provider",
            )
        return rows

    @abstractmethod
    def measure_local(
        self, context: AnalysisContext, proof_units: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, TierEvidence]:
        """Return T1 evidence keyed by proof-unit ID."""

    @abstractmethod
    def intervene_causally(
        self, context: AnalysisContext, proof_unit_ids: Sequence[str]
    ) -> Mapping[str, TierEvidence]:
        """Run reference replacement and matched sham for T2."""

    @abstractmethod
    def confirm_coherence(
        self, context: AnalysisContext, proof_unit_ids: Sequence[str]
    ) -> Mapping[str, TierEvidence]:
        """Confirm fixed complete-carrier directions on independent states."""

    @abstractmethod
    def run_trajectory(
        self, context: AnalysisContext, proof_unit_ids: Sequence[str]
    ) -> Mapping[str, TierEvidence]:
        """Run paired live-weight trajectories for T4."""


@dataclass(frozen=True)
class AnalysisSpec:
    subject: str
    reference: ReferenceProvider
    candidates: Sequence[CandidateBackend]
    states: Sequence[AnalysisState]
    output_dir: Path
    model_factory: Optional[Callable[[], Any]] = None
    step_builder: Optional[Callable[[Any, AnalysisState], StepExecution]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    resources: Optional[ResourceBudget] = None

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("subject must be non-empty")
        if not self.candidates:
            raise ValueError("at least one candidate backend is required")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        if any(re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None for value in candidate_ids):
            raise ValueError("candidate IDs must be filesystem-safe")
        state_ids = [state.state_id for state in self.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("state IDs must be unique")
        if self.resources is not None:
            self.resources.validate(self.output_dir)


@dataclass(frozen=True)
class AnalysisReport:
    run_id: str
    status: str
    subject: str
    proof_unit_count: int
    unresolved_proof_units: int
    candidate_summaries: Mapping[str, Mapping[str, Any]]
    case_certificates: Sequence[CaseCertificate]
    artifact_dir: Path

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "subject": self.subject,
            "proof_unit_count": self.proof_unit_count,
            "unresolved_proof_units": self.unresolved_proof_units,
            "candidate_summaries": dict(self.candidate_summaries),
            "case_certificates": [row.as_dict() for row in self.case_certificates],
            "artifact_dir": str(self.artifact_dir),
        }
