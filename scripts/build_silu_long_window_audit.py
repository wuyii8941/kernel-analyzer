#!/usr/bin/env python3
"""Build the missing late-window audit from SiLU's recorded carrier projections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/declared_persistent_4096/qwen3vl_silu_4096_with_loss.json"
OUTPUT = ROOT / "results/property/declared_persistent_4096/qwen3vl_silu_4096_with_loss_windows.json"


def amplification(values: np.ndarray) -> float:
    denominator = float(np.sqrt(np.square(values, dtype=np.float64).sum()))
    return abs(float(values.sum(dtype=np.float64))) / denominator if denominator else 0.0


def summarize(values: np.ndarray, *, seed: int, draws: int = 4000) -> dict:
    observed = amplification(values)
    denominator = float(np.sqrt(np.square(values, dtype=np.float64).sum()))
    generator = np.random.default_rng(seed)
    null_values: list[float] = []
    batch = 250
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        signs = generator.integers(0, 2, size=(count, values.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null_values.extend((np.abs(signs @ values) / denominator).tolist())
    null = np.asarray(null_values, dtype=np.float64)

    windows = []
    for offset in range(0, values.size, 32):
        part = values[offset : offset + 32]
        windows.append({
            "measured_index": [offset + 1, offset + part.size],
            "coherence_amplification": amplification(part),
        })
    late = windows[len(windows) // 2 :]
    late_above_one = sum(row["coherence_amplification"] > 1.0 for row in late)
    return {
        "measurement_geometry": "FROZEN_CARRIER_PROJECTION",
        "measured_steps": int(values.size),
        "coherence_amplification": observed,
        "sign_flip_null": {
            "draws": draws,
            "seed": seed,
            "median": float(np.quantile(null, 0.50)),
            "upper_95": float(np.quantile(null, 0.95)),
            "upper_99": float(np.quantile(null, 0.99)),
            "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (draws + 1)),
        },
        "window_steps": 32,
        "windows": windows,
        "late_windows": len(late),
        "late_windows_above_one": late_above_one,
        "late_window_fraction_above_one": late_above_one / len(late),
        "long_persistent": bool(
            observed > float(np.quantile(null, 0.95))
            and (1 + np.count_nonzero(null >= observed)) / (draws + 1) <= 0.05
            and late_above_one / len(late) >= 0.75
        ),
    }


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    records = payload.get("records", [])
    levels = {}
    for level, key in (
        ("local", "local_frozen_carrier_projection"),
        ("feedback", "feedback_frozen_carrier_projection"),
    ):
        values = np.asarray(
            [float(row[key]) for row in records if row.get(key) is not None],
            dtype=np.float64,
        )
        if values.size == 0:
            raise RuntimeError(f"no recorded {key} values")
        levels[level] = summarize(values, seed=20260825 + len(level))
    result = {
        "schema": "kernel-analyzer-silu-long-window-audit-v1",
        "status": "COMPLETE_OFFLINE_FROM_RECORDED_PROJECTIONS",
        "source_artifact": str(SOURCE.relative_to(ROOT)),
        "source_steps": len(records),
        "projection_recording_starts_after_step": len(records) - levels["feedback"]["measured_steps"],
        "levels": levels,
        "claim_boundary": (
            "Late-window evidence is computed from the pre-existing frozen-carrier projections. "
            "It verifies persistence on that declared direction, not the full parameter space."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "local_long_persistent": levels["local"]["long_persistent"],
        "feedback_long_persistent": levels["feedback"]["long_persistent"],
    }))


if __name__ == "__main__":
    main()
