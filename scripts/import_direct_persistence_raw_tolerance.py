#!/usr/bin/env python3
"""Register compact metrics from a raw held-out replay."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/direct_persistence_v4"


def main() -> None:
    tolerance_path = BASE / "tolerance_comparison.json"
    tolerance = json.loads(tolerance_path.read_text())
    # Prefer the v4 replays, which save same-state candidate/repair updates;
    # fall back to the earlier output/gradient-only files only if a v4 row is
    # not present.  This keeps the import deterministic during a partial run.
    raw_paths = [
        next(path for path in (
            BASE / "gemma4_raw_gelu_loss_v4_tolerance.json",
            BASE / "gemma4_raw_gelu_loss_tolerance.json",
        ) if path.is_file()),
        next(path for path in (
            BASE / "gemma4_raw_softmax_v4_tolerance.json",
            BASE / "gemma4_raw_softmax_tolerance.json",
        ) if path.is_file()),
        next(path for path in (
            BASE / "gemma4_raw_gelu1860_v4_tolerance.json",
            BASE / "gemma4_raw_gelu_backward1860_tolerance.json",
        ) if path.is_file()),
    ]
    raw_paths = [path for path in raw_paths if path.is_file()]
    raw_rows = []
    case_ids = []
    for path in raw_paths:
        raw = json.loads(path.read_text())
        case_ids.append(raw["case_id"])
        raw_rows.append({
            "status": raw["status"],
            "case_id": raw["case_id"],
            "target_endpoint": raw.get("target_endpoint"),
            "raw_manifest": raw["raw_manifest"],
            "raw_manifest_sha256": raw["raw_manifest_sha256"],
            "output": raw["output"],
            "gradient": raw["gradient"],
            "update": raw["update"],
            "update_pair": raw.get("update_pair"),
            "persistence": raw.get("persistence"),
            "severity_proxy": raw.get("severity_proxy"),
            "rtol_atol": raw["rtol_atol"],
            "claim_boundary": raw["claim_boundary"],
        })
    tolerance["raw_new_impl_reanalysis"] = {
        "status": "PARTIAL_RAW_TOLERANCE_COMPLETE",
        "rows": raw_rows,
    }
    tolerance["claim_boundary"] = (
        "Three fresh Gemma NEW_IMPL replays now have exact output, gradient, and "
        "same-state effective-update candidate/repair pairs. The full frozen-pool "
        "tolerance family remains partial because older rows lack raw operands, "
        "but the three fresh rows now support update ULP and rtol/atol analysis."
    )
    tolerance["missing_baselines"] = [
        "complete max absolute error across the frozen pool",
        "complete relative L2 across the frozen pool",
        "complete ULP across the frozen pool",
        "complete rtol/atol sweep across the frozen pool",
        "complete output RMS across the frozen pool",
        "complete gradient RMS across the frozen pool",
        "complete raw operands for the remaining frozen rows",
    ]
    tolerance_path.write_text(json.dumps(tolerance, indent=2, sort_keys=True) + "\n")

    status_path = BASE / "execution_status.json"
    status = json.loads(status_path.read_text())
    status.setdefault("tolerance", {})["raw_new_impl_rows"] = len(raw_rows)
    status["tolerance"]["status"] = f"PARTIAL_RAW_TOLERANCE_WITH_{len(raw_rows)}_NEW_IMPL_ROWS"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": tolerance["raw_new_impl_reanalysis"]["status"], "rows": len(raw_rows), "case_ids": case_ids}))


if __name__ == "__main__":
    main()
