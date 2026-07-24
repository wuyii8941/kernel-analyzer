"""Fail-closed pre-reveal sealing and post-reveal scoring for Phase 3.

The localization core deliberately knows nothing about historical patches.  A
separate evaluator needs a small, reproducible protocol to enforce that
separation: seal the locator output *before* the evaluator provides the patch
scope, then score only coverage and stopping decisions.  This module does not
pretend that a hash can prove what an analyst has read; its purpose is to make
the required attestations and the exact scored objects explicit.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .operator_evidence import canonical_json_sha256


REQUIRED_TRUTH_FIELDS = {
    "schema_version",
    "case_id",
    "certificate_sha256",
    "revealed_by_independent_evaluator",
    "patch_stage_tags",
    "patch_candidate_ids",
    "correct_stopping_level",
}


def _certificate_payload(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the hash payload, excluding only its self-referential digest."""

    payload = deepcopy(dict(certificate))
    pre_reveal = payload.get("pre_reveal")
    if isinstance(pre_reveal, Mapping):
        pre_reveal = dict(pre_reveal)
        pre_reveal.pop("certificate_sha256", None)
        payload["pre_reveal"] = pre_reveal
    return payload


def seal_pre_reveal_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a pre-reveal attestation and a deterministic certificate digest.

    The caller must run this immediately after the frozen locator.  If a case
    already contains reveal material, sealing fails rather than overwriting it.
    This prevents accidental post-hoc relabelling, while making no claim that a
    local program can prove an analyst's private knowledge.
    """

    sealed = deepcopy(dict(certificate))
    forbidden = {"post_reveal", "external_ground_truth", "patch_truth"} & set(sealed)
    if forbidden:
        raise ValueError(f"cannot seal certificate containing reveal fields: {sorted(forbidden)}")
    if "pre_reveal" in sealed:
        raise ValueError("certificate is already sealed")
    sealed["pre_reveal"] = {
        "fixed_revision_accessed": False,
        "patch_accessed": False,
        "issue_root_cause_accessed": False,
        "certificate_sha256": "PENDING",
        "protocol_limit": (
            "this is an attestation, not a proof of an analyst's private knowledge; "
            "independent withholding is required for an external score"
        ),
    }
    sealed["pre_reveal"]["certificate_sha256"] = canonical_json_sha256(
        _certificate_payload(sealed)
    )
    return sealed


def validate_pre_reveal_certificate(certificate: Mapping[str, Any]) -> list[str]:
    """Return protocol violations; an empty list is the only valid input to scoring."""

    errors: list[str] = []
    pre_reveal = certificate.get("pre_reveal")
    if not isinstance(pre_reveal, Mapping):
        return ["missing pre_reveal seal"]
    for key in ("fixed_revision_accessed", "patch_accessed", "issue_root_cause_accessed"):
        if pre_reveal.get(key) is not False:
            errors.append(f"pre_reveal attestation {key} is not false")
    expected = pre_reveal.get("certificate_sha256")
    actual = canonical_json_sha256(_certificate_payload(certificate))
    if expected != actual:
        errors.append("pre_reveal certificate hash mismatch")
    forbidden = {"post_reveal", "external_ground_truth", "patch_truth"} & set(certificate)
    if forbidden:
        errors.append(f"certificate contains reveal fields: {sorted(forbidden)}")
    return errors


def _strings(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        found: set[str] = set()
        for key in ("id", "region_id", "node_id", "kernel_id", "target", "stage_id", "tag"):
            if value.get(key) is not None:
                found.add(str(value[key]))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        found: set[str] = set()
        for item in value:
            found |= _strings(item)
        return found
    return {str(value)}


def certificate_candidate_ids(certificate: Mapping[str, Any]) -> set[str]:
    """Collect explicitly emitted candidate IDs; never derive names from deltas."""

    reduction = certificate.get("region_reduction", {})
    provenance = certificate.get("provenance", {})
    candidates = _strings(reduction.get("candidate_regions", ()))
    candidates |= _strings(provenance.get("candidate_regions", ()))
    candidates |= _strings(provenance.get("candidate_kernel_ids", ()))
    return candidates


def certificate_stage_tags(certificate: Mapping[str, Any]) -> set[str]:
    screen = certificate.get("stage_screen", {})
    provenance = certificate.get("provenance", {})
    return _strings(screen.get("failing_stages", ())) | _strings(provenance.get("stage_tags", ()))


def score_post_reveal(
    certificate: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, Any]:
    """Score a sealed certificate against evaluator-supplied patch metadata.

    ``truth`` is deliberately a compact label file, not patch text.  It must
    be created and held by an independent evaluator until seal time.  Candidate
    coverage requires explicit identifier overlap, so a generic stage label
    cannot accidentally receive operation-level credit.
    """

    errors = validate_pre_reveal_certificate(certificate)
    missing = sorted(REQUIRED_TRUTH_FIELDS - set(truth))
    if missing:
        errors.append(f"missing truth fields: {missing}")
    case_id = certificate.get("case_identity", {}).get("case_id")
    if truth.get("case_id") != case_id:
        errors.append("truth case_id does not match certificate")
    seal = certificate.get("pre_reveal", {}).get("certificate_sha256")
    if truth.get("certificate_sha256") != seal:
        errors.append("truth is not bound to the sealed certificate")
    if truth.get("revealed_by_independent_evaluator") is not True:
        errors.append("truth lacks independent-evaluator attestation")
    if errors:
        return {
            "schema_version": "forkcert.historical-post-reveal-score.v0.1",
            "valid": False,
            "errors": errors,
            "allowed_claim": "INVALID",
        }

    observed_stages = certificate_stage_tags(certificate)
    expected_stages = _strings(truth["patch_stage_tags"])
    observed_candidates = certificate_candidate_ids(certificate)
    expected_candidates = _strings(truth["patch_candidate_ids"])
    stage_overlap = sorted(observed_stages & expected_stages)
    candidate_overlap = sorted(observed_candidates & expected_candidates)
    initial_count = int(certificate.get("region_reduction", {}).get("input_region_count", 0))
    final_count = len(certificate.get("region_reduction", {}).get("candidate_regions", ()))
    reported_level = str(certificate.get("provenance", {}).get("stopping_level", "UNDECLARED"))
    correct_stopping = reported_level == str(truth["correct_stopping_level"])
    expected_kernel = str(truth["correct_stopping_level"]).lower() == "kernel"
    reported_kernel = reported_level.lower() == "kernel"
    return {
        "schema_version": "forkcert.historical-post-reveal-score.v0.1",
        "valid": True,
        "case_id": case_id,
        "sealed_certificate_sha256": seal,
        "stage_coverage": bool(stage_overlap),
        "stage_overlap": stage_overlap,
        "candidate_mechanism_coverage": bool(candidate_overlap),
        "candidate_overlap": candidate_overlap,
        "candidate_set": {"initial_count": initial_count, "final_count": final_count,
                          "reduction_ratio": (1 - final_count / initial_count) if initial_count else None},
        "reported_stopping_level": reported_level,
        "correct_stopping_level": truth["correct_stopping_level"],
        "stopping_decision_correct": correct_stopping,
        "erroneous_kernel_descent": bool(reported_kernel and not expected_kernel),
        "allowed_claim": "EXTERNAL_PATCH_VALIDATED" if stage_overlap else "EXTERNAL_SCORE_NO_COVERAGE",
        "limitations": [
            "score measures declared patch-scope coverage, not unique causality",
            "an independent evaluator's withholding attestation is required but cannot be proved by this code",
            "candidate coverage is unavailable when the truth has no comparable provenance identifiers",
        ],
    }
