"""Fail-closed validation for the sample-completion trace.

The trace is deliberately small: large vectors stay outside the repository and
are represented by a path, coordinate count, and digest.  The validator checks
that all four steps of the same training difference are present for all 32
steps before allowing a final label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


STAGES = ("operator_output", "parameter_gradient", "optimizer_update", "trajectory", "feedback")
PREDICTOR_FORBIDDEN = frozenset({
    "final_label", "final_drift", "seup_verdict", "t4_verdict", "historical_case_name",
})


@dataclass(frozen=True)
class TraceValidation:
    status: str
    reasons: tuple[str, ...]
    case_id: str
    steps: int

    @property
    def valid(self) -> bool:
        return self.status == "VALID_32_STEP_TRACE"


def _finite_number(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return value == value and abs(value) != float("inf")


def validate_trace(payload: Mapping[str, Any]) -> TraceValidation:
    """Validate one uniform trace without reading historical verdicts."""
    case_id = str(payload.get("case_id", ""))
    reasons: list[str] = []
    if not case_id:
        reasons.append("MISSING_CASE_ID")
    if payload.get("schema") != "kernel-analyzer-sample-completion-trace-v1":
        reasons.append("WRONG_SCHEMA")
    state_ids = payload.get("state_ids")
    if not isinstance(state_ids, Sequence) or isinstance(state_ids, (str, bytes)):
        reasons.append("MISSING_STATE_IDS")
        state_ids = []
    if len(state_ids) != 32:
        reasons.append("REQUIRES_EXACTLY_32_STEPS")
    if len(set(map(str, state_ids))) != len(state_ids):
        reasons.append("DUPLICATE_STATE_IDS")
    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        reasons.append("MISSING_STAGES")
        stages = {}
    for stage in STAGES:
        rows = stages.get(stage)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            reasons.append(f"MISSING_STAGE:{stage}")
            continue
        if len(rows) != 32:
            reasons.append(f"STAGE_LENGTH_NOT_32:{stage}")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                reasons.append(f"MALFORMED_ROW:{stage}:{index}")
                continue
            if not _finite_number(row.get("l2")):
                reasons.append(f"NONFINITE_L2:{stage}:{index}")
            if not row.get("vector_digest") and not row.get("vector_path"):
                reasons.append(f"MISSING_VECTOR_PROVENANCE:{stage}:{index}")
    common = payload.get("common_state")
    if not isinstance(common, Mapping) or common.get("all_components_equal") is not True:
        reasons.append("INVALID_COMMON_STATE")
    if payload.get("candidate_repair_bound") is not True:
        reasons.append("MISSING_CANDIDATE_REPAIR_BOUND")
    if payload.get("forward_backward_bound") is not True:
        reasons.append("MISSING_FORWARD_BACKWARD_BOUND")
    if payload.get("difference_reaches_parameter_gradient") is not True:
        reasons.append("PARAMETER_GRADIENT_REACH_NOT_CERTIFIED")
    if any(key in payload for key in PREDICTOR_FORBIDDEN):
        # A final label may exist in a completed trace, but the predictor view
        # below removes it.  It must never be used to validate the trace.
        pass
    status = "VALID_32_STEP_TRACE" if not reasons else "ABSTAIN_MALFORMED_OR_INCOMPLETE"
    return TraceValidation(status, tuple(sorted(set(reasons))), case_id, len(state_ids))


def predictor_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the information a 16-step screen may read."""
    state_ids = list(payload.get("state_ids", []))[:16]
    stages = payload.get("stages", {})
    view: dict[str, Any] = {
        "schema": "kernel-analyzer-sample-completion-predictor-input-v1",
        "case_id": payload.get("case_id"),
        "state_ids": state_ids,
        "stages": {
            stage: list(stages.get(stage, []))[:16]
            for stage in ("operator_output", "parameter_gradient", "optimizer_update")
        },
        "allowed_features": [
            "operator_output_error", "parameter_gradient_error", "optimizer_update_error",
            "lag_1", "lag_2", "lag_4", "local_rms", "dtype", "reduction_length",
        ],
    }
    # Do not copy any unknown payload keys.  In particular this excludes final
    # drift, labels, T4/SEUP verdicts, and the last 16 steps.
    return view

