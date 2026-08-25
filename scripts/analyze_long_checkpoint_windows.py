#!/usr/bin/env python3
"""Compute late-window persistence from a bound-endpoint checkpoint.

The live runner keeps a compact three-level vector sequence in its checkpoint
but historically exported only the whole-horizon Gram statistics.  This
offline pass makes the persistence claim auditable without rerunning a model:
non-overlapping windows are tested against their own sign-flip null, and the
late half is reported separately from the full 4096-step aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def one_window(values: np.ndarray, draws: int, seed: int) -> dict[str, float | int]:
    if values.ndim != 2 or values.shape[0] < 2:
        return {"steps": int(values.shape[0]), "A": 0.0, "null95": 0.0, "p": 1.0}
    resultant = values.sum(axis=0)
    energy = float(np.sqrt(np.square(values).sum()))
    observed = float(np.linalg.norm(resultant) / max(energy, 1e-30))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, values.shape[0]))
    null_sum = signs @ values
    null = np.linalg.norm(null_sum, axis=1) / max(energy, 1e-30)
    return {
        "steps": int(values.shape[0]),
        "A": observed,
        "null95": float(np.quantile(null, 0.95)),
        "p": float((1.0 + np.count_nonzero(null >= observed - 1e-12)) / (draws + 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError("checkpoint has no vector rows")
    level_keys = {
        "local": "derived_local_vector",
        "feedback": "derived_feedback_vector",
        "actual": "derived_actual_vector",
    }
    result: dict[str, object] = {
        "schema": "kernel-analyzer-long-window-audit-v1",
        "status": "COMPLETE_CHECKPOINT_WINDOW_REANALYSIS",
        "case_id": payload.get("case_id"),
        "steps": len(rows),
        "window": args.window,
        "null_draws": args.draws,
        "levels": {},
        "claim_boundary": "Window persistence is checked from the saved candidate/repair vector sequence; it is not a substitute for a full-model convergence experiment.",
    }
    for level, key in level_keys.items():
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        windows = []
        for start in range(0, len(values) - args.window + 1, args.window):
            windows.append({
                "start": start + 1,
                "end": start + args.window,
                **one_window(values[start:start + args.window], args.draws, 20260820 + start),
            })
        late = windows[len(windows) // 2 :]
        above = [row for row in late if row["p"] <= 0.05 and row["A"] > row["null95"]]
        level_result = {
            "windows": windows,
            "late_windows": len(late),
            "late_windows_above_own_null": len(above),
            "late_fraction_above_own_null": len(above) / max(len(late), 1),
            "long_persistent": bool(late and len(above) / len(late) >= 0.75),
        }
        result["levels"][level] = level_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_id": payload.get("case_id"), "steps": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
