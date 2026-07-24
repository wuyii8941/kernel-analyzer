"""Case-agnostic primitives for fail-closed compiler localization.

This module contains neither model names nor operator names nor repair rules.
It is deliberately agnostic about how a caller runs a stage or substitutes a
region.  Callers supply a declared semantic predicate; these routines preserve
the audit trail and refuse to turn an ordered observation into a proof of a
unique first-bad compiler pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class StageObservation:
    stage_id: str
    contract_holds: bool | None
    artifact_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def screen_stages(observations: Iterable[StageObservation]) -> dict[str, Any]:
    """Report supported stage candidates without assuming stages form a chain."""
    rows = list(observations)
    if not rows:
        raise ValueError("stage screening needs at least one observation")
    failed = [row.stage_id for row in rows if row.contract_holds is False]
    unknown = [row.stage_id for row in rows if row.contract_holds is None]
    passed = [row.stage_id for row in rows if row.contract_holds is True]
    if not failed:
        claim = "NO_OBSERVED_STAGE_FAILURE"
    elif unknown:
        claim = "SUPPORTED_STAGE_CANDIDATE_WITH_UNKNOWN_GAPS"
    else:
        claim = "SUPPORTED_STAGE_CANDIDATE"
    return {"observations": [asdict(row) for row in rows], "passing_stages": passed, "failing_stages": failed,
            "unknown_stages": unknown, "allowed_stage_claim": claim,
            "not_claimed": "a unique first-bad compiler pass or a strict stage order"}


def ddmin_regions(regions: Sequence[str], preserves_symptom: Callable[[tuple[str, ...]], bool]) -> dict[str, Any]:
    """Generic symptom-preserving delta reduction over an explicitly supplied set."""
    current = tuple(regions)
    if len(set(current)) != len(current):
        raise ValueError("region IDs must be unique")
    trace: list[dict[str, Any]] = []
    if not current:
        return {"status": "UNINSTANTIATED", "candidate_regions": [], "trace": trace}
    initial = bool(preserves_symptom(current))
    trace.append({"subset": list(current), "preserves_symptom": initial})
    if not initial:
        return {"status": "INVALID_INITIAL_SYMPTOM", "candidate_regions": list(current), "trace": trace}
    # A singleton inventory can establish that the declared symptom is present,
    # but it contains no removable alternative.  Calling that a ``minimal``
    # candidate set would incorrectly turn enumeration into reduction.
    if len(current) == 1:
        return {
            "status": "UNREDUCIBLE_SINGLETON_INVENTORY",
            "candidate_regions": list(current),
            "input_region_count": 1,
            "reduction_ratio": 0.0,
            "trace": trace,
            "not_claimed": "symptom-preserving reduction, unique source region, causal share, or root cause",
        }
    granularity = 2
    while len(current) >= 2:
        chunks = [current[i::granularity] for i in range(granularity) if current[i::granularity]]
        reduced = False
        for chunk in chunks:
            complement = tuple(item for item in current if item not in set(chunk))
            if not complement:
                continue
            outcome = bool(preserves_symptom(complement))
            trace.append({"subset": list(complement), "preserves_symptom": outcome})
            if outcome:
                current = complement; granularity = max(2, granularity - 1); reduced = True; break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return {"status": "ONE_MINIMAL_CANDIDATE_SET", "candidate_regions": list(current),
            "input_region_count": len(regions), "reduction_ratio": 1.0 - len(current) / len(regions),
            "trace": trace, "not_claimed": "unique source region, causal share, or root cause"}


def localization_certificate(*, case_identity: Mapping[str, Any], semantic_contract: Mapping[str, Any],
                             stage_screen: Mapping[str, Any], reduction: Mapping[str, Any],
                             provenance: Mapping[str, Any], evidence: Mapping[str, Any],
                             manual_decisions: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Assemble an auditable certificate without upgrading its evidence level."""
    return {"schema_version": "forkcert.generic-localization-certificate.v0.1", "case_identity": dict(case_identity),
            "semantic_contract": dict(semantic_contract), "stage_screen": dict(stage_screen),
            "region_reduction": dict(reduction), "provenance": dict(provenance), "evidence": dict(evidence),
            "manual_decisions": [dict(row) for row in manual_decisions],
            "stopping_reason": "the certificate stops at the narrowest layer whose provenance and controlled evidence are present; absence of lower-level evidence is not permission to guess an operation or kernel"}
