#!/usr/bin/env python3
"""Apply the unified 16+16 statistics to saved-P and SiLU response vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
    str(ROOT / "archive/round1_code/src"),
]

from analyze_three_mechanism_profiles import _profile  # noqa: E402
from scripts.targeted_external_intervention import _count_sketch  # noqa: E402


def sketch(value: torch.Tensor | np.ndarray) -> np.ndarray:
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
    return _count_sketch(tensor.detach().float().reshape(-1), dimension=4096).numpy()


def saved_p() -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    base = Path("/data1/tzh/cache/bias_replay/qwen_saved_p_response_v3")
    even, odd, ids = [], [], []
    for step in range(1, 33):
        e = np.memmap(base / f"even_{step:04d}.f32", mode="r", dtype=np.float32)
        o = np.memmap(base / f"odd_{step:04d}.f32", mode="r", dtype=np.float32)
        even.append(sketch(np.asarray(e)))
        odd.append(sketch(np.asarray(o)))
        ids.append(str(step))
    return even, odd, ids


def silu() -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    base = Path("/data1/tzh/cache/bias_replay/vl_silu_response_v3")
    even, odd, ids = [], [], []
    for step in range(1, 33):
        row = torch.load(base / f"step-{step:02d}.pt", map_location="cpu", weights_only=False)
        even.append(sketch(row["response_even"]))
        odd.append(sketch(row["response_odd"]))
        ids.append(str(step))
    return even, odd, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("saved_p", "silu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "results/property/extended_unified_profiles_v1/protocol.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_NEW_RESULTS":
        raise RuntimeError("protocol was not frozen")
    effects, odd_response, ids = saved_p() if args.case == "saved_p" else silu()
    result = _profile(effects, odd_response, seed=20260931 if args.case == "saved_p" else 20260932)
    payload = {
        "schema": "kernel-analyzer-unified-response-profile-v1",
        "status": "COMPLETE",
        "case_id": "qwen_saved_p_seq128" if args.case == "saved_p" else "qwen3vl_silu_backward",
        "contrast_id": "ANTITHETIC_RESPONSE_REMAINDER",
        "effect": "0.5 * (update(+delta) + update(-delta) - 2 * update(0))",
        "repair_signal": "odd response 0.5 * (update(+delta) - update(-delta))",
        "state_ids": ids,
        "split": {"calibration": ids[:16], "confirmation": ids[16:]},
        "profile": result,
        "measurement_geometry": "FIXED_4096_DIMENSION_COUNT_SKETCH_OF_FULL_RESPONSE_VECTOR",
        "claim_boundary": "Trajectory-conditioned response contrast. It is not an ordinary candidate-repair prevalence row.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
