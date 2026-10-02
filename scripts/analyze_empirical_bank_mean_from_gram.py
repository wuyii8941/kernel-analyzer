#!/usr/bin/env python3
"""Run scoped projected-mean checks from retained complete Gram matrices.

This is intentionally an offline analysis.  A complete original-coordinate
Gram is treated as a finite empirical bank, then independent indices are drawn
with replacement.  A calibration direction is learned from the first draws
and the confirmation projection is tested with a one-sided Student interval.
The result is never labelled as a natural-training population theorem.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/root_cause_closure_v1/empirical_bank_projected_mean_v1.json"
CALIBRATION_COUNT = 32
CONFIRMATION_COUNT = 64
SEED = 20260920


def load(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text())


def sources() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    softmax = "results/coverage/cases/qwen128_softmax_fb_formal.json"
    for source in ("saved_state_reconstruction", "semantic_total"):
        rows.append({"problem_group": "softmax_saved_state_backward", "source": softmax, "component": source})
    mm_paths = (
        "results/coverage/cases/qwen128_vproj_precision_decomposition.json",
        "results/coverage/cases/qwen64_vproj_precision_decomposition.json",
        "results/coverage/cases/mamba_seq64_input_proj_precision_decomposition.json",
        "results/coverage/cases/phi4_seq64_lmhead_dx_precision_decomposition.json",
    )
    for path in mm_paths:
        data = load(path)
        for component in ("kernel", "output_rounding", "total"):
            rows.append({"problem_group": "mm_gemm_output_and_accumulation", "source": path, "component": component})
    return rows


def gram_for(row: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    data = load(row["source"])
    direction = data["direction"][row["component"]]
    gram = np.asarray(direction["gram"], dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"non-square Gram for {row}")
    if not direction.get("complete_coordinates", False):
        raise ValueError(f"Gram is not declared complete-coordinate for {row}")
    return gram, direction


def one_source(row: dict[str, str], offset: int) -> dict[str, Any]:
    gram, direction = gram_for(row)
    n = gram.shape[0]
    if n < 2:
        raise ValueError("empirical bank needs at least two states")
    rng = np.random.default_rng(SEED + offset)
    indices = rng.integers(0, n, size=CALIBRATION_COUNT + CONFIRMATION_COUNT)
    calibration = indices[:CALIBRATION_COUNT]
    confirmation = indices[CALIBRATION_COUNT:]
    calibration_gram = gram[np.ix_(calibration, calibration)]
    direction_norm_sq = float(calibration_gram.sum() / CALIBRATION_COUNT**2)
    if not math.isfinite(direction_norm_sq) or direction_norm_sq <= 0:
        return {
            **row,
            "status": "DIRECTION_NOT_IDENTIFIABLE",
            "bank_state_count": n,
            "complete_coordinates": direction["complete_coordinates"],
        }
    norm = math.sqrt(direction_norm_sq)
    values = np.asarray([
        float(gram[index, calibration].sum() / (CALIBRATION_COUNT * norm))
        for index in confirmation
    ], dtype=np.float64)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    one_sided_half = float(t.ppf(0.95, CONFIRMATION_COUNT - 1)) * sd / math.sqrt(CONFIRMATION_COUNT)
    two_sided_half = float(t.ppf(0.975, CONFIRMATION_COUNT - 1)) * sd / math.sqrt(CONFIRMATION_COUNT)
    finite_bank_mean_norm_sq = float(gram.sum() / n**2)
    return {
        **row,
        "status": "PROJECTED_MEAN_SUPPORTED" if mean - one_sided_half > 0 else "PROJECTED_MEAN_NOT_CONFIRMED",
        "bank_state_count": n,
        "calibration_count": CALIBRATION_COUNT,
        "confirmation_count": CONFIRMATION_COUNT,
        "draw_seed": SEED + offset,
        "complete_coordinates": bool(direction["complete_coordinates"]),
        "finite_bank_mean_vector_norm_squared": finite_bank_mean_norm_sq,
        "calibration_direction_norm": norm,
        "confirmation_projection_mean": mean,
        "confirmation_projection_sd": sd,
        "confirmation_projection_two_sided_95": [mean - two_sided_half, mean + two_sided_half],
        "confirmation_projection_one_sided_95_lower": mean - one_sided_half,
        "confirmation_positive_count": int(np.count_nonzero(values > 0)),
        "confirmation_negative_count": int(np.count_nonzero(values < 0)),
        "confirmation_zero_count": int(np.count_nonzero(values == 0)),
        "implies_vector_mean_nonzero_if_bound_positive": bool(mean - one_sided_half > 0),
        "scope": "DECLARED_IID_WITH_REPLACEMENT_FROM_FINITE_EMPIRICAL_BANK",
        "assumptions": [
            "complete original-coordinate Gram represents the declared finite bank",
            "with-replacement index draws are independent",
            "calibration direction is fixed before confirmation",
            "finite variance and Student-t approximation for the signed projection",
        ],
    }


def main() -> None:
    results = [one_source(row, index) for index, row in enumerate(sources())]
    payload = {
        "schema": "kernel-analyzer-empirical-bank-projected-mean-v1",
        "status": "COMPLETE_OFFLINE_FROM_RETAINED_COMPLETE_GRAMS",
        "new_gpu_measurements": False,
        "calibration_count": CALIBRATION_COUNT,
        "confirmation_count": CONFIRMATION_COUNT,
        "seed": SEED,
        "claim_boundary": (
            "These are scoped tests for finite empirical banks represented by retained "
            "complete original-coordinate Grams. They do not establish a natural-training "
            "population claim or a training-quality consequence."
        ),
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "supported": sum(row["status"] == "PROJECTED_MEAN_SUPPORTED" for row in results),
        "not_confirmed": sum(row["status"] == "PROJECTED_MEAN_NOT_CONFIRMED" for row in results),
        "not_identifiable": sum(row["status"] == "DIRECTION_NOT_IDENTIFIABLE" for row in results),
    }))


if __name__ == "__main__":
    main()
