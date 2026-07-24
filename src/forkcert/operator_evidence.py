"""Fail-closed evidence primitives for relational operator analysis.

The helpers in this module deliberately separate four questions:

* did the complete implementations disagree;
* did a local callable disagree on byte-identical boundary inputs;
* is there auditable provenance from source/module to generated code; and
* did an intervention preserve all declared non-target context.

No numerical magnitude is used to choose the allowed claim level.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


CLAIM_LEVELS = (
    "INVALID",
    "OBSERVATION",
    "LOCAL_INJECTION",
    "INTERVENTION_DEPENDENT_ATTRIBUTION",
    "OPERATOR_LEVEL_EFFECT",
    "CROSS_LEVEL_COMPILER_LOCALIZATION",
)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_fingerprint(tensor: Any) -> dict[str, Any]:
    """Return a content plus layout fingerprint without retaining tensor data."""

    value = tensor.detach()
    cpu = value.cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    storage_offset = int(value.storage_offset()) if hasattr(value, "storage_offset") else 0
    return {
        "content_sha256": digest.hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": list(value.stride()),
        "storage_offset": storage_offset,
        "requires_grad": bool(value.requires_grad),
        "is_contiguous": bool(value.is_contiguous()),
    }


def fingerprint_tree(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Fingerprint every tensor leaf using a stable structural path."""

    rows: list[dict[str, Any]] = []
    if hasattr(value, "detach") and hasattr(value, "shape"):
        rows.append({"path": prefix or "<root>", **tensor_fingerprint(value)})
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(fingerprint_tree(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.extend(fingerprint_tree(item, f"{prefix}[{index}]"))
    return rows


def same_boundary_inputs(left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]) -> bool:
    """Require content *and* layout equality for all captured tensor leaves."""

    return list(left) == list(right)


def context_signature(context: Mapping[str, Any], ignored_target_ids: Iterable[str] = ()) -> dict[str, Any]:
    """Canonicalize graph/kernel context while removing declared target entries only."""

    ignored = set(ignored_target_ids)
    rows = []
    for row in context.get("artifacts", []):
        if str(row.get("target_id", "")) in ignored:
            continue
        rows.append(dict(row))
    rows.sort(key=lambda row: canonical_json_sha256(row))
    return {
        "compiler_config_digest": context.get("compiler_config_digest"),
        "graph_count": context.get("graph_count"),
        "graphs": sorted(context.get("graphs", []), key=canonical_json_sha256),
        "artifacts": rows,
        "shape_layout_contracts": sorted(
            context.get("shape_layout_contracts", []), key=canonical_json_sha256
        ),
        "autotuning": context.get("autotuning", {"status": "UNOBSERVED"}),
    }


def compare_non_target_context(
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
    ignored_target_ids: Iterable[str] = (),
) -> dict[str, Any]:
    before = context_signature(baseline, ignored_target_ids)
    after = context_signature(intervention, ignored_target_ids)
    return {
        "exact": before == after,
        "baseline_sha256": canonical_json_sha256(before),
        "intervention_sha256": canonical_json_sha256(after),
        "ignored_target_ids": sorted(set(ignored_target_ids)),
        "baseline": before,
        "intervention": after,
    }


def multiset_exact(left: Iterable[Any], right: Iterable[Any]) -> bool:
    """Order-insensitive identity check used for non-target kernel inventories."""

    return Counter(canonical_json_sha256(item) for item in left) == Counter(
        canonical_json_sha256(item) for item in right
    )


def production_mediation_interpretation(
    production_observed: bool | None,
    mediation_observed: bool | None,
) -> dict[str, Any]:
    """Interpret the two evidence chains without turning either into root cause.

    ``None`` means that the corresponding experiment was not validly instantiated.
    Production requires a same-boundary-input local replay.  Mediation requires a
    fixed suffix and a boundary-value substitution.  A positive result in both is
    still only a candidate-region result until provenance and context gates pass.
    """

    if production_observed is None or mediation_observed is None:
        interpretation = "one or both evidence chains are uninstantiated"
    elif production_observed and mediation_observed:
        interpretation = (
            "the region produces a local discrepancy and that boundary discrepancy "
            "changes the declared endpoint; this is not by itself a root-cause claim"
        )
    elif production_observed:
        interpretation = (
            "the region produces a local discrepancy with no observed effect on the "
            "declared endpoint"
        )
    elif mediation_observed:
        interpretation = (
            "the boundary carries an endpoint-relevant upstream discrepancy; the region "
            "is not established as its producer"
        )
    else:
        interpretation = "no current production or endpoint-mediation evidence"
    return {
        "production_observed": production_observed,
        "mediation_observed": mediation_observed,
        "interpretation": interpretation,
    }


@dataclass(frozen=True)
class EvidenceGates:
    complete_witness: bool
    same_input_local_replay: bool
    local_discrepancy_reproducible: bool
    provenance_complete: bool
    candidate_realization_preserved: bool
    intervention_executed: bool
    oracle_recomputed: bool
    non_target_context_invariant: bool
    lower_level_replay: bool = False
    first_bad_stage_isolated: bool = False
    null_controls_valid: bool = True


def allowed_claim_level(gates: EvidenceGates) -> str:
    """Return the strongest claim licensed by gates, never by effect magnitude."""

    if not gates.null_controls_valid:
        return "INVALID"
    if not gates.complete_witness:
        return "INVALID"
    level = "OBSERVATION"
    if gates.same_input_local_replay and gates.local_discrepancy_reproducible:
        level = "LOCAL_INJECTION"
    if gates.intervention_executed and gates.oracle_recomputed:
        level = "INTERVENTION_DEPENDENT_ATTRIBUTION"
    if (
        level == "INTERVENTION_DEPENDENT_ATTRIBUTION"
        and gates.provenance_complete
        and gates.candidate_realization_preserved
        and gates.non_target_context_invariant
    ):
        level = "OPERATOR_LEVEL_EFFECT"
    if (
        level == "OPERATOR_LEVEL_EFFECT"
        and gates.lower_level_replay
        and gates.first_bad_stage_isolated
    ):
        level = "CROSS_LEVEL_COMPILER_LOCALIZATION"
    return level


def validate_evidence_report(report: Mapping[str, Any]) -> list[str]:
    """Independently check the minimum machine-readable evidence contract."""

    errors: list[str] = []
    required = {
        "schema_version",
        "case_identity",
        "region_inventory",
        "local_replay",
        "provenance",
        "intervention",
        "oracle",
        "gates",
        "allowed_claim_level",
        "limitations",
    }
    missing = sorted(required - set(report))
    if missing:
        errors.append(f"missing report fields: {missing}")
        return errors
    try:
        gates = EvidenceGates(**report["gates"])
    except (TypeError, ValueError) as error:
        errors.append(f"invalid gates: {error}")
        return errors
    expected = allowed_claim_level(gates)
    if report["allowed_claim_level"] != expected:
        errors.append(
            f"claim level mismatch: reported={report['allowed_claim_level']} expected={expected}"
        )
    if report["allowed_claim_level"] not in CLAIM_LEVELS:
        errors.append("unknown claim level")
    if not report["limitations"]:
        errors.append("limitations must be non-empty")
    return errors
