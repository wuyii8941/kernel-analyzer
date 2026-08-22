#!/usr/bin/env python3
"""Compute the two exact empirical response-parity terms of conditional mu."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"


def norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value))


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = norm(left) * norm(right)
    return float(np.dot(left, right) / denominator) if denominator else None


def summarize(case_id: str, even_sum: np.ndarray, odd_sum: np.ndarray,
              natural_sum: np.ndarray, steps: int, source: str,
              closure_mode: str = "INDEPENDENT_NATURAL_VECTOR") -> dict[str, Any]:
    reconstructed = even_sum + odd_sum
    closure = norm(reconstructed - natural_sum)
    natural_norm = norm(natural_sum)
    even_norm, odd_norm = norm(even_sum), norm(odd_sum)
    if even_norm > 1.25 * odd_norm:
        dominant = "EVEN_RESPONSE_RECTIFICATION"
    elif odd_norm > 1.25 * even_norm:
        dominant = "ODD_RESPONSE_TO_SOURCE_ASYMMETRY"
    else:
        dominant = "MIXED_EVEN_AND_ODD_RESPONSE"
    return {
        "case_id": case_id,
        "steps": steps,
        "coordinates": int(natural_sum.size),
        "source": source,
        "mu_even_l2": norm(even_sum / steps),
        "mu_odd_l2": norm(odd_sum / steps),
        "mu_natural_l2": natural_norm / steps,
        "even_over_natural_resultant": norm(even_sum) / max(natural_norm, 1e-30),
        "odd_over_natural_resultant": norm(odd_sum) / max(natural_norm, 1e-30),
        "even_odd_cosine": cosine(even_sum, odd_sum),
        "even_natural_cosine": cosine(even_sum, natural_sum),
        "odd_natural_cosine": cosine(odd_sum, natural_sum),
        "closure_l2": closure,
        "closure_relative": closure / max(natural_norm, 1e-30),
        "closure_mode": closure_mode,
        "dominant_term": dominant,
        "interpretation": {
            "even": "empirical mean of (F(+epsilon)+F(-epsilon))/2",
            "odd": "empirical mean of (F(+epsilon)-F(-epsilon))/2",
            "identity": "E_p[F(epsilon)] = E_p[F_even(epsilon)] + E_p[F_odd(epsilon)]",
        },
    }


def saved_p(spool: Path) -> dict[str, Any]:
    even_paths = sorted(spool.glob("even_*.f32"))
    odd_paths = sorted(spool.glob("odd_*.f32"))
    if len(even_paths) != 32 or len(odd_paths) != 32:
        raise RuntimeError("saved-P requires 32 even and 32 odd vectors")
    coordinates = even_paths[0].stat().st_size // 4
    even_sum = np.zeros(coordinates, dtype=np.float64)
    odd_sum = np.zeros(coordinates, dtype=np.float64)
    for even_path, odd_path in zip(even_paths, odd_paths):
        even_sum += np.fromfile(even_path, dtype=np.float32)
        odd_sum += np.fromfile(odd_path, dtype=np.float32)
    natural_sum = even_sum + odd_sum
    result = summarize(
        "qwen_saved_p_seq128", even_sum, odd_sum, natural_sum, 32, str(spool),
        closure_mode="ALGEBRAIC_RECONSTRUCTION_WITH_REPORTED_NORM_CHECK",
    )
    report = json.loads((BASE / "qwen_saved_p_pairing_response_vectors_v3.json").read_text())
    reported = float(report["aggregate"]["natural_update_resultant_l2"])
    result["reported_natural_resultant_l2"] = reported
    result["reported_norm_relative_error"] = abs(norm(natural_sum) - reported) / max(
        reported, 1e-30
    )
    return result


def silu(spool: Path) -> dict[str, Any]:
    paths = sorted(spool.glob("step-*.pt"))
    if len(paths) != 32:
        raise RuntimeError("SiLU requires 32 response-vector checkpoints")
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    coordinates = int(first["response_even"].numel())
    even_sum = np.zeros(coordinates, dtype=np.float64)
    odd_sum = np.zeros(coordinates, dtype=np.float64)
    natural_sum = np.zeros(coordinates, dtype=np.float64)
    even_energy: list[float] = []
    for path in paths:
        row = torch.load(path, map_location="cpu", weights_only=False)
        even = row["response_even"].reshape(-1).numpy().astype(np.float64, copy=False)
        odd = row["response_odd"].reshape(-1).numpy().astype(np.float64, copy=False)
        natural = row["natural_update"].reshape(-1).numpy().astype(np.float64, copy=False)
        even_sum += even
        odd_sum += odd
        natural_sum += natural
        even_energy.append(float(np.dot(even, even)))
    result = summarize(
        "qwen3vl_silu_seq160", even_sum, odd_sum, natural_sum, 32, str(spool)
    )
    result["even_energy_first_two_fraction"] = sum(even_energy[:2]) / max(
        sum(even_energy), 1e-30
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved-p-spool", type=Path,
        default=Path("/data1/tzh/cache/bias_replay/qwen_saved_p_response_v3"),
    )
    parser.add_argument(
        "--silu-spool", type=Path,
        default=Path("/data1/tzh/cache/bias_replay/vl_silu_response_v3"),
    )
    parser.add_argument("--output", type=Path, default=BASE / "mu_parity_decomposition.json")
    args = parser.parse_args()
    cases = [saved_p(args.saved_p_spool), silu(args.silu_spool)]
    payload = {
        "schema": "kernel-analyzer-mu-parity-decomposition-v1",
        "status": "COMPLETE_EXACT_RESPONSE_PARITY",
        "cases": cases,
        "claim_boundary": (
            "This exactly decomposes the empirical conditional response mean for "
            "two replayable cases. It does not infer an implementation-orbit density "
            "or extrapolate to cases without exact antithetic response vectors."
        ),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": payload["status"],
        "cases": [{"case_id": row["case_id"], "dominant": row["dominant_term"]}
                  for row in cases],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
