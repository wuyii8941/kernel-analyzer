"""Validation helpers for the eight-case bias-formation audit.

The audit keeps formation, mechanism intervention, and trajectory consequence
as separate evidence layers.  In particular, an unrelated-state ``CENTERED``
result is not a conditional-bias null, and a trajectory cannot manufacture a
formation-stage label.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence


class EvidenceStatus(str, Enum):
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"
    NOT_MEASURED = "NOT_MEASURED"
    UNRESOLVED = "UNRESOLVED"
    INVALID = "INVALID"


class MechanismVerdict(str, Enum):
    SOURCE = "SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM"
    TRANSPORT = "SUPPORTED_CASE_SPECIFIC_TRANSPORT_MECHANISM"
    CONTRACT = "SUPPORTED_CASE_SPECIFIC_CONTRACT_MECHANISM"
    TRANSPORT_CONTRACT = "SUPPORTED_SEMANTIC_REGION_TRANSPORT_CONTRACT_MECHANISM"
    PARTIAL_SOURCE = "PARTIAL_SOURCE_MECHANISM"
    CAUSAL_DIFFERENCE = "CAUSAL_IMPLEMENTATION_DIFFERENCE_FORMATION_UNRESOLVED"
    OPTIMIZER_RESPONSE = "SUPPORTED_CASE_SPECIFIC_OPTIMIZER_RESPONSE_MECHANISM"
    CONTRAST_MISMATCH = "UNRESOLVED_CONTRAST_MISMATCH"
    VARIANCE_ONLY = "VARIANCE_ONLY_UNDER_DECLARED_CONDITION"
    UNRESOLVED = "UNRESOLVED"


ALLOWED_FORMATION = {
    "BIASED", "CENTERED", "CANCELING_STRUCTURE", "UNRESOLVED",
    "NOT_MEASURED",
}


def validate_case(case: Mapping[str, Any]) -> None:
    """Fail closed on the scientific category errors found in older reports."""

    required = {
        "case_id", "model", "semantic_unit", "forward_backward",
        "formation", "mechanism", "trajectory", "evidence",
    }
    missing = sorted(required - set(case))
    if missing:
        raise ValueError(f"{case.get('case_id', '<unknown>')}: missing {missing}")

    formation = case["formation"]
    for level in ("conditional", "global"):
        values = formation.get(level, {})
        for layer in ("local", "gradient", "update"):
            if values.get(layer, "NOT_MEASURED") not in ALLOWED_FORMATION:
                raise ValueError(
                    f"{case['case_id']}: invalid {level} {layer} formation status"
                )

    # A trajectory is a consequence.  It must never be cited as the source of
    # local/gradient/update formation labels.
    if formation.get("label_source") == "TRAJECTORY":
        raise ValueError(f"{case['case_id']}: trajectory cannot label formation")

    verdict = MechanismVerdict(case["mechanism"]["verdict"])
    if verdict == MechanismVerdict.VARIANCE_ONLY:
        conditional = formation.get("conditional", {})
        if not conditional or any(
            conditional.get(layer) != "CENTERED"
            for layer in ("local", "gradient", "update")
        ):
            raise ValueError(
                f"{case['case_id']}: variance-only requires three conditional nulls"
            )

    supported = {
        MechanismVerdict.SOURCE,
        MechanismVerdict.TRANSPORT,
        MechanismVerdict.CONTRACT,
        MechanismVerdict.TRANSPORT_CONTRACT,
        MechanismVerdict.OPTIMIZER_RESPONSE,
    }
    intervention = case["mechanism"].get("intervention", {})
    if verdict in supported and not (
        intervention.get("causal_effect") is True
        and intervention.get("matched_sham_exact") is True
    ):
        raise ValueError(
            f"{case['case_id']}: supported mechanism lacks intervention/sham"
        )

    if case["case_id"] == "qwen128_vproj_mm" and case["mechanism"].get(
        "trajectory_repairs_declared_local_source"
    ) is not False:
        raise ValueError(
            "qwen128_vproj_mm: accumulation trajectory must not be attributed "
            "to the separately identified output-rounding source"
        )

    for path in case["evidence"]:
        if not isinstance(path, str) or not path:
            raise ValueError(f"{case['case_id']}: malformed evidence path")


def validate_audit(cases: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        "liger_fused_ce", "phi4_seq64_lmhead_dx", "qwen64_vproj_mm",
        "qwen128_vproj_mm", "qwen_saved_p_seq128", "qwen3vl_silu_layer0",
        "mamba_seq64_input_proj", "qwen_layer23_attention_state",
    }
    ids = [str(case.get("case_id")) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case in systematic audit")
    if set(ids) != expected:
        raise ValueError(
            f"eight-case denominator changed: missing={sorted(expected-set(ids))}, "
            f"extra={sorted(set(ids)-expected)}"
        )
    for case in cases:
        validate_case(case)


def first_conditional_bias_stage(case: Mapping[str, Any]) -> str:
    """Return a stage only when every preceding conditional layer is centered."""

    values = case["formation"].get("conditional", {})
    preceding: list[str] = []
    for layer in ("local", "gradient", "update"):
        status = values.get(layer, "NOT_MEASURED")
        if status == "BIASED":
            return layer.upper() if all(x == "CENTERED" for x in preceding) else "UNRESOLVED"
        if status not in {"CENTERED", "BIASED"}:
            return "UNRESOLVED"
        preceding.append(status)
    return "NONE" if preceding else "UNRESOLVED"
