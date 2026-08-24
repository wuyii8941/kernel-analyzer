#!/usr/bin/env python3
"""Derive the tolerance metrics that are actually present in raw replays.

This is deliberately fail-closed.  A raw replay contains differences and the
repair vector, so max error, relative L2 and RMS can be derived.  It does not
contain the original floating-point operands or their bit patterns, so ULP and
an rtol/atol decision are not invented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/direct_persistence_v4/tolerance_comparison.json"
RAW = {
    "phi4_seq64_lmhead_dx": Path(
        "/data1/tzh/cache/kernel_analyzer/direct_persistence_v4/phi_seq64_raw_stage.json"
    ),
    "qwen_seq128_lmhead_dx": Path(
        "/data1/tzh/cache/kernel_analyzer/direct_persistence_v4/qwen_seq128_raw_stage.json"
    ),
}


def metrics(delta: np.ndarray, reference: np.ndarray | None) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    flat = delta.reshape(delta.shape[0], -1)
    row_l2 = np.linalg.norm(flat, axis=1)
    result: dict[str, Any] = {
        "state_count": int(flat.shape[0]),
        "coordinate_count": int(flat.shape[1]),
        "max_abs_error": float(np.max(np.abs(flat))),
        "rms_error_mean": float(np.sqrt(np.mean(flat * flat, axis=1)).mean()),
        "rms_error_max": float(np.sqrt(np.mean(flat * flat, axis=1)).max()),
        "l2_error_mean": float(row_l2.mean()),
    }
    if reference is None:
        result["relative_l2_mean"] = None
        result["relative_l2_status"] = "ABSTAIN_MISSING_REFERENCE_VECTOR"
    else:
        ref = np.asarray(reference, dtype=np.float64).reshape(flat.shape)
        denom = np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        result["relative_l2_mean"] = float(np.mean(row_l2 / denom))
        result["relative_l2_status"] = "DERIVED_FROM_REPAIR_VECTOR"
    return result


def one(case_id: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"case_id": case_id, "status": "ABSTAIN_MISSING_RAW_REPLAY", "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    vectors = data.get("vectors", {})
    required = {
        "operator_output": "operator_output_error",
        "gradient": "candidate_gradient",
        "repair_gradient": "repair_gradient",
        "update": "candidate_update",
        "repair_update": "repair_update",
    }
    missing = [key for key in required.values() if key not in vectors]
    if missing:
        return {
            "case_id": case_id,
            "status": "ABSTAIN_MISSING_RAW_FIELD",
            "missing": missing,
            "path": str(path),
        }
    gradient = np.asarray(vectors[required["gradient"]], dtype=np.float64)
    repair_gradient = np.asarray(vectors[required["repair_gradient"]], dtype=np.float64)
    update = np.asarray(vectors[required["update"]], dtype=np.float64)
    repair_update = np.asarray(vectors[required["repair_update"]], dtype=np.float64)
    output = np.asarray(vectors[required["operator_output"]], dtype=np.float64)
    return {
        "case_id": case_id,
        "status": "PARTIAL_METRICS_DERIVED",
        "raw_stage_path": str(path),
        "sequence_length": data.get("sequence_length"),
        "optimizer": data.get("optimizer"),
        "metrics": {
            "output": metrics(output, None),
            "gradient_difference": metrics(gradient - repair_gradient, repair_gradient),
            "update_difference": metrics(update - repair_update, repair_update),
        },
        "unavailable": {
            "ulp": "ABSTAIN_MISSING_BITWISE_OPERANDS",
            "rtol_atol": "ABSTAIN_MISSING_REFERENCE_OPERANDS_AND_THRESHOLDS",
        },
    }


def main() -> None:
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    payload["raw_stage_reanalysis"] = {
        "status": "PARTIAL_RAW_STAGE_METRICS",
        "rows": [one(case_id, path) for case_id, path in RAW.items()],
        "claim_boundary": (
            "These rows are an offline reanalysis of preserved raw replay vectors. "
            "They do not complete the common held-out tolerance family; ULP and rtol/atol "
            "remain fail-closed until original/reference operands are preserved."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["raw_stage_reanalysis"]["status"], "rows": 2}, sort_keys=True))


if __name__ == "__main__":
    main()
