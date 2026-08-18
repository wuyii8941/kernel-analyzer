"""Closed-loop consequence measurements for the v2 bias protocol.

Formation and consequence are intentionally separate.  A consequence trace
may advance a candidate/repair pair and decompose the resulting parameter
drift, but it can never claim where bias was first generated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, MutableMapping, Sequence


class ConsequenceStatus:
    COMPLETE = "COMPLETE"
    UNRESOLVED_MISSING_STEP = "UNRESOLVED_MISSING_STEP"
    INVALID_NONFINITE = "INVALID_NONFINITE"
    INVALID_RECURRENCE = "INVALID_RECURRENCE"


@dataclass(frozen=True)
class ConsequenceValue:
    """A scalar carrier coordinate plus the norm of the full increment."""

    signed_value: float
    norm: float

    @classmethod
    def from_value(cls, value: Any) -> "ConsequenceValue":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("consequence values must be objects")
        signed = float(value.get("signed_value"))
        norm = float(value.get("norm"))
        if not math.isfinite(signed) or not math.isfinite(norm) or norm < 0:
            raise ValueError("consequence values must be finite with nonnegative norm")
        return cls(signed, norm)

    def as_dict(self) -> Dict[str, float]:
        return {"signed_value": self.signed_value, "norm": self.norm}


class BiasConsequenceTrace:
    """A closed-loop candidate/repair recurrence, never a formation oracle."""

    def __init__(self, case_id: str, step_ids: Sequence[str], recurrence_tolerance: float = 1e-6) -> None:
        self.case_id = str(case_id)
        self.step_ids = tuple(str(x) for x in step_ids)
        if not self.case_id or not self.step_ids or len(set(self.step_ids)) != len(self.step_ids):
            raise ValueError("case_id and unique step_ids are required")
        if recurrence_tolerance < 0 or not math.isfinite(recurrence_tolerance):
            raise ValueError("recurrence_tolerance must be finite and nonnegative")
        self.recurrence_tolerance = float(recurrence_tolerance)
        self._rows: MutableMapping[str, Dict[str, Any]] = {}

    def add(
        self,
        step_id: str,
        *,
        local_increment: Any,
        feedback_increment: Any,
        actual_drift_increment: Any,
        final_drift: Any,
        recurrence_residual: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        step_id = str(step_id)
        if step_id in self._rows or step_id not in self.step_ids:
            raise ValueError("step_id is not a unique frozen consequence step")
        try:
            local = ConsequenceValue.from_value(local_increment)
            feedback = ConsequenceValue.from_value(feedback_increment)
            actual = ConsequenceValue.from_value(actual_drift_increment)
            final = ConsequenceValue.from_value(final_drift)
            residual = 0.0 if recurrence_residual is None else float(recurrence_residual)
            if not math.isfinite(residual):
                raise ValueError("recurrence residual is nonfinite")
        except (TypeError, ValueError, KeyError):
            self._rows[step_id] = {"step_id": step_id, "invalid": True}
            return
        self._rows[step_id] = {
            "step_id": step_id,
            "local_increment": local,
            "feedback_increment": feedback,
            "actual_drift_increment": actual,
            "final_drift": final,
            "recurrence_residual": residual,
            "metadata": dict(metadata or {}),
        }

    def finalize(self) -> Dict[str, Any]:
        missing = [step for step in self.step_ids if step not in self._rows]
        invalid = any(row.get("invalid") for row in self._rows.values())
        nonfinite = False
        recurrence_bad = False
        local_sum = feedback_sum = actual_sum = 0.0
        rows: list[Dict[str, Any]] = []
        for step in self.step_ids:
            row = self._rows.get(step)
            if row is None:
                continue
            if row.get("invalid"):
                continue
            local = row["local_increment"]
            feedback = row["feedback_increment"]
            actual = row["actual_drift_increment"]
            final = row["final_drift"]
            residual = float(row["recurrence_residual"])
            local_sum += local.signed_value
            feedback_sum += feedback.signed_value
            actual_sum += actual.signed_value
            recurrence_bad = recurrence_bad or abs(residual) > self.recurrence_tolerance
            vals = [local.signed_value, local.norm, feedback.signed_value, feedback.norm,
                    actual.signed_value, actual.norm, final.signed_value, final.norm, residual]
            nonfinite = nonfinite or any(not math.isfinite(value) for value in vals)
            rows.append({
                "step_id": step,
                "local_increment": local.as_dict(),
                "feedback_increment": feedback.as_dict(),
                "actual_drift_increment": actual.as_dict(),
                "final_drift": final.as_dict(),
                "recurrence_residual": residual,
                "metadata": row["metadata"],
            })
        if missing:
            status = ConsequenceStatus.UNRESOLVED_MISSING_STEP
        elif invalid or nonfinite:
            status = ConsequenceStatus.INVALID_NONFINITE
        elif recurrence_bad:
            status = ConsequenceStatus.INVALID_RECURRENCE
        else:
            status = ConsequenceStatus.COMPLETE
        return {
            "schema": "kernel-analyzer-bias-consequence-certificate-v2",
            "case_id": self.case_id,
            "status": status,
            "measurement_kind": "candidate_repair_closed_loop_consequence",
            "uses_candidate_measurements": True,
            "uses_historical_verdicts": False,
            "verdict_blind": True,
            "step_ids": list(self.step_ids),
            "step_count": len(self.step_ids),
            "recurrence_tolerance": self.recurrence_tolerance,
            "missing_step_ids": missing,
            "first_confirmed_bias_stage": None,
            "first_observed_biased_stage": None,
            "formation_point": "NOT_APPLICABLE",
            "cumulative": {
                "local_signed_sum": local_sum,
                "feedback_signed_sum": feedback_sum,
                "actual_drift_signed_sum": actual_sum,
            },
            "row_digest": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "rows": rows,
        }
