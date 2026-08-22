"""Four-counterfactual closed-loop consequence certificate (v2.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Mapping, MutableMapping, Sequence


class ConsequenceStatus:
    COMPLETE = "COMPLETE"
    UNRESOLVED_MISSING_STEP = "UNRESOLVED_MISSING_STEP"
    INVALID_NONFINITE = "INVALID_NONFINITE"
    INVALID_MALFORMED = "INVALID_MALFORMED"
    INVALID_RECURRENCE = "INVALID_RECURRENCE"


@dataclass(frozen=True)
class UpdateValue:
    """One actual optimizer update or drift state, with an optional carrier projection."""

    vector: tuple[float, ...]
    signed_value: float
    vector_digest: str

    @classmethod
    def from_value(cls, value: Any) -> "UpdateValue":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("counterfactual arm must be an object")
        raw = value.get("vector")
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("counterfactual arm requires a complete vector")
        vector = tuple(float(x) for x in raw)
        if any(not math.isfinite(x) for x in vector):
            raise ValueError("counterfactual vector is nonfinite")
        signed = float(value.get("signed_value"))
        if not math.isfinite(signed):
            raise ValueError("counterfactual signed projection is nonfinite")
        digest = hashlib.sha256(json.dumps(vector, separators=(",", ":")).encode()).hexdigest()
        supplied = str(value.get("vector_digest", digest))
        if supplied != digest:
            raise ValueError("counterfactual vector digest does not match the vector")
        return cls(vector, signed, digest)

    @classmethod
    def from_vector(cls, vector: Sequence[float], signed_value: float) -> "UpdateValue":
        values = tuple(float(x) for x in vector)
        digest = hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(values, float(signed_value), digest)

    @property
    def norm(self) -> float:
        return math.sqrt(math.fsum(x * x for x in self.vector))

    def difference(self, other: "UpdateValue") -> "UpdateValue":
        if len(self.vector) != len(other.vector):
            raise ValueError("counterfactual coordinate sets differ")
        vector = tuple(a - b for a, b in zip(self.vector, other.vector))
        return UpdateValue.from_vector(
            vector, self.signed_value - other.signed_value
        )

    def plus(self, other: "UpdateValue") -> "UpdateValue":
        if len(self.vector) != len(other.vector):
            raise ValueError("counterfactual coordinate sets differ")
        vector = tuple(a + b for a, b in zip(self.vector, other.vector))
        return UpdateValue.from_vector(
            vector, self.signed_value + other.signed_value
        )

    def as_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        result = {
            "coordinate_count": len(self.vector),
            "vector_digest": self.vector_digest,
            "signed_value": self.signed_value,
            "norm": self.norm,
        }
        if include_vector:
            result["vector"] = list(self.vector)
        return result


@dataclass(frozen=True)
class ConsequenceStep:
    step_id: str
    candidate_at_candidate_state: UpdateValue
    repair_at_candidate_state: UpdateValue
    candidate_at_repair_state: UpdateValue
    repair_at_repair_state: UpdateValue
    drift_before: UpdateValue
    drift_after: UpdateValue
    local_effect: UpdateValue
    feedback_effect: UpdateValue
    expected_drift_increment: UpdateValue
    actual_drift_increment: UpdateValue
    recurrence_residual: UpdateValue
    global_metadata: Mapping[str, Any] = field(default_factory=dict)


class BiasConsequenceTrace:
    """Computes local/feedback recurrence from four real counterfactual arms."""

    def __init__(self, case_id: str, step_ids: Sequence[str], recurrence_tolerance: float = 1e-6) -> None:
        self.case_id = str(case_id)
        self.step_ids = tuple(str(x) for x in step_ids)
        if not self.case_id or not self.step_ids or len(set(self.step_ids)) != len(self.step_ids):
            raise ValueError("case_id and unique step_ids are required")
        if recurrence_tolerance < 0 or not math.isfinite(recurrence_tolerance):
            raise ValueError("recurrence_tolerance must be finite and nonnegative")
        self.recurrence_tolerance = float(recurrence_tolerance)
        self._rows: MutableMapping[str, ConsequenceStep | str] = {}

    def add(
        self,
        step_id: str,
        *,
        candidate_at_candidate_state: Any,
        repair_at_candidate_state: Any,
        candidate_at_repair_state: Any,
        repair_at_repair_state: Any,
        drift_before: Any,
        drift_after: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        step_id = str(step_id)
        if step_id in self._rows or step_id not in self.step_ids:
            raise ValueError("step_id is duplicate or outside frozen consequence split")
        try:
            cc = UpdateValue.from_value(candidate_at_candidate_state)
            rc = UpdateValue.from_value(repair_at_candidate_state)
            cr = UpdateValue.from_value(candidate_at_repair_state)
            rr = UpdateValue.from_value(repair_at_repair_state)
            before = UpdateValue.from_value(drift_before)
            after = UpdateValue.from_value(drift_after)
            local = cc.difference(rc)
            local_repair = cr.difference(rr)
            feedback_candidate = cc.difference(cr)
            feedback_repair = rc.difference(rr)
            # Symmetric decomposition makes the source/feedback convention
            # explicit while preserving D = L + B exactly.
            local = UpdateValue.from_vector(
                tuple((a + b) / 2.0 for a, b in zip(local.vector, local_repair.vector)),
                (local.signed_value + local_repair.signed_value) / 2.0,
            )
            feedback = UpdateValue.from_vector(
                tuple((a + b) / 2.0 for a, b in zip(feedback_candidate.vector, feedback_repair.vector)),
                (feedback_candidate.signed_value + feedback_repair.signed_value) / 2.0,
            )
            expected = cc.difference(rr)
            actual = after.difference(before)
            recurrence = actual.difference(expected)
            row = ConsequenceStep(
                step_id, cc, rc, cr, rr, before, after, local, feedback,
                expected, actual, recurrence, dict(metadata or {}),
            )
        except (TypeError, ValueError, KeyError) as exc:
            text = str(exc).lower()
            self._rows[step_id] = ConsequenceStatus.INVALID_NONFINITE if "nonfinite" in text else ConsequenceStatus.INVALID_MALFORMED
            return
        self._rows[step_id] = row

    def finalize(self) -> Dict[str, Any]:
        missing = [step for step in self.step_ids if step not in self._rows]
        invalid = [value for value in self._rows.values() if isinstance(value, str)]
        rows = [self._rows[step] for step in self.step_ids if isinstance(self._rows.get(step), ConsequenceStep)]
        recurrence_bad = any(row.recurrence_residual.norm > self.recurrence_tolerance for row in rows)
        nonfinite = any(
            not math.isfinite(value)
            for row in rows
            for value in (row.local_effect.signed_value, row.feedback_effect.signed_value,
                          row.actual_drift_increment.signed_value, row.recurrence_residual.norm)
        )
        global_residual = None
        if rows:
            actual_sum = rows[0].actual_drift_increment
            for row in rows[1:]:
                actual_sum = actual_sum.plus(row.actual_drift_increment)
            expected_total = rows[-1].drift_after.difference(rows[0].drift_before)
            global_residual = actual_sum.difference(expected_total)
            recurrence_bad = recurrence_bad or global_residual.norm > self.recurrence_tolerance
        if missing:
            status = ConsequenceStatus.UNRESOLVED_MISSING_STEP
        elif ConsequenceStatus.INVALID_NONFINITE in invalid or nonfinite:
            status = ConsequenceStatus.INVALID_NONFINITE
        elif ConsequenceStatus.INVALID_MALFORMED in invalid:
            status = ConsequenceStatus.INVALID_MALFORMED
        elif recurrence_bad:
            status = ConsequenceStatus.INVALID_RECURRENCE
        else:
            status = ConsequenceStatus.COMPLETE
        serial = [self._row_dict(row) for row in rows]
        return {
            "schema": "kernel-analyzer-bias-consequence-certificate-v2_1",
            "case_id": self.case_id,
            "status": status,
            "measurement_kind": "candidate_repair_closed_loop_consequence",
            "uses_candidate_measurements": True,
            "uses_historical_verdicts": False,
            "verdict_blind": True,
            "step_ids": list(self.step_ids),
            "step_count": len(self.step_ids),
            "recurrence_tolerance": self.recurrence_tolerance,
            "four_counterfactual_arms_required": True,
            "missing_step_ids": missing,
            "first_confirmed_bias_stage": None,
            "first_observed_biased_stage": None,
            "formation_point": "NOT_APPLICABLE",
            "global_recurrence_residual": None if global_residual is None else global_residual.as_dict(),
            "cumulative": {
                "local_signed_sum": math.fsum(row.local_effect.signed_value for row in rows),
                "feedback_signed_sum": math.fsum(row.feedback_effect.signed_value for row in rows),
                "actual_drift_signed_sum": math.fsum(row.actual_drift_increment.signed_value for row in rows),
            },
            "row_digest": hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "rows": serial,
        }

    @staticmethod
    def _row_dict(row: ConsequenceStep) -> Dict[str, Any]:
        return {
            "step_id": row.step_id,
            "candidate_at_candidate_state": row.candidate_at_candidate_state.as_dict(),
            "repair_at_candidate_state": row.repair_at_candidate_state.as_dict(),
            "candidate_at_repair_state": row.candidate_at_repair_state.as_dict(),
            "repair_at_repair_state": row.repair_at_repair_state.as_dict(),
            "drift_before": row.drift_before.as_dict(),
            "drift_after": row.drift_after.as_dict(),
            "local_effect": row.local_effect.as_dict(),
            "feedback_effect": row.feedback_effect.as_dict(),
            "expected_drift_increment": row.expected_drift_increment.as_dict(),
            "actual_drift_increment": row.actual_drift_increment.as_dict(),
            "recurrence_residual": row.recurrence_residual.as_dict(),
            "metadata": dict(row.global_metadata),
        }
