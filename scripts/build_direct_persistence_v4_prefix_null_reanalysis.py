#!/usr/bin/env python3
"""Recompute prefix sign-flip calibration where raw 32-step vectors exist.

The frozen engineering rule remains A16 > 1.0.  This file only adds an
offline calibration record; it never replaces missing rows with a null or
turns an uncalibrated short-screen result into a safety verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


RAW = {
    "phi4_seq64_lmhead_dx": Path(
        "/data1/tzh/cache/kernel_analyzer/direct_persistence_v4/phi_seq64_raw_stage.json"
    ),
    "qwen_seq128_lmhead_dx": Path(
        "/data1/tzh/cache/kernel_analyzer/direct_persistence_v4/qwen_seq128_raw_stage.json"
    ),
}
OUT = Path("results/property/direct_persistence_v4/prefix_null_reanalysis.json")
SEED = 20260822
DRAWS = 4000


def one(case_id: str, path: Path) -> dict:
    if not path.is_file():
        return {"case_id": case_id, "status": "ABSTAIN_MISSING_RAW_REPLAY"}
    data = json.loads(path.read_text(encoding="utf-8"))
    vectors = data.get("vectors", {})
    if "candidate_update" not in vectors or "repair_update" not in vectors:
        return {"case_id": case_id, "status": "ABSTAIN_MISSING_UPDATE_VECTORS"}
    delta = np.asarray(vectors["candidate_update"], dtype=np.float64) - np.asarray(
        vectors["repair_update"], dtype=np.float64
    )
    rng = np.random.default_rng(SEED)
    rows = {}
    for horizon in (16, 32):
        x = delta[:horizon]
        denominator = float(np.sqrt(np.sum(x * x)))
        observed = float(np.linalg.norm(np.sum(x, axis=0)) / max(denominator, 1e-30))
        signs = rng.integers(0, 2, size=(DRAWS, horizon), dtype=np.int8) * 2 - 1
        null = np.linalg.norm(np.sum(signs[:, :, None] * x[None, :, :], axis=1), axis=1)
        null /= max(denominator, 1e-30)
        median = float(np.median(null))
        upper = float(np.quantile(null, 0.95))
        p = float((1 + int(np.count_nonzero(null >= observed))) / (DRAWS + 1))
        rows[str(horizon)] = {
            "observed_A": observed,
            "null_median": median,
            "null_upper_95": upper,
            "one_sided_p": p,
            "above_null_95": bool(observed > upper),
        }
    return {
        "case_id": case_id,
        "status": "COMPLETE_RAW_PREFIX_CALIBRATION",
        "optimizer": data.get("optimizer"),
        "state_count": len(delta),
        "seed": SEED,
        "draws": DRAWS,
        "levels": rows,
        "claim_boundary": "Offline calibration for two preserved raw replays; it does not alter the frozen A16 screening rule or fill missing cohort rows.",
    }


def main() -> None:
    OUT.write_text(
        json.dumps(
            {
                "schema": "kernel-analyzer-direct-persistence-v4-prefix-null-reanalysis-v1",
                "status": "PARTIAL_TWO_RAW_REPLAYS",
                "seed": SEED,
                "draws": DRAWS,
                "rows": [one(case_id, path) for case_id, path in RAW.items()],
                "claim_boundary": "Prefix sign-flip calibration is available only where complete raw update vectors were preserved; all other rows remain ABSTAIN.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PARTIAL_TWO_RAW_REPLAYS", "rows": len(RAW)}, sort_keys=True))


if __name__ == "__main__":
    main()
