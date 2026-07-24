"""Manifest/adapter boundary for the frozen generic localization core.

An arbitrary subject needs an execution adapter to restore state or replace a
region.  The adapter may execute case-specific code, but supplies no candidate
ranking or claim level; those remain in the generic core.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from .localization import StageObservation, ddmin_regions, localization_certificate, screen_stages


class LocalizationAdapter(Protocol):
    def case_identity(self) -> Mapping[str, Any]: ...
    def semantic_contract(self) -> Mapping[str, Any]: ...
    def stage_ids(self) -> Sequence[str]: ...
    def run_stage(self, stage_id: str) -> Mapping[str, Any]: ...
    def region_ids(self) -> Sequence[str]: ...
    def preserves_symptom(self, enabled_regions: tuple[str, ...]) -> bool: ...
    def provenance(self, region_ids: Sequence[str]) -> Mapping[str, Any]: ...
    def evidence(self, candidate_regions: Sequence[str]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RuntimeResult:
    certificate: dict[str, Any]
    stage_screen: dict[str, Any]
    reduction: dict[str, Any]


def run_localization(adapter: LocalizationAdapter) -> RuntimeResult:
    """Run generic screen/reduce/certify; never inspect op/model/fault names."""
    observations = []
    for stage_id in adapter.stage_ids():
        row = dict(adapter.run_stage(stage_id))
        if "contract_holds" not in row:
            raise ValueError(f"stage {stage_id} omitted contract_holds")
        observations.append(StageObservation(stage_id, row["contract_holds"],
            tuple(row.get("artifact_ids", ())), tuple(row.get("notes", ()))))
    stages = screen_stages(observations)
    regions = tuple(adapter.region_ids())
    reduction = ddmin_regions(regions, adapter.preserves_symptom)
    candidates = reduction.get("candidate_regions", [])
    certificate = localization_certificate(case_identity=adapter.case_identity(), semantic_contract=adapter.semantic_contract(),
        stage_screen=stages, reduction=reduction, provenance=adapter.provenance(candidates),
        evidence=adapter.evidence(candidates), manual_decisions=[])
    return RuntimeResult(certificate, stages, reduction)
