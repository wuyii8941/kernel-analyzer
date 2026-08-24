#!/usr/bin/env python3
"""Split a captured same-state AdamW contrast into numerator and denominator effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sequence_stats(rows: list[np.ndarray]) -> dict[str, float]:
    total = np.sum(rows, axis=0, dtype=np.float64)
    energy2 = sum(float(np.dot(row, row)) for row in rows)
    resultant = float(np.linalg.norm(total))
    energy = math.sqrt(energy2)
    return {"resultant_norm": resultant, "path_energy": energy,
            "coherence_amplification": resultant / max(energy, 1e-300)}


def decompose_step(gc: np.ndarray, gr: np.ndarray, m0: np.ndarray, v0: np.ndarray,
                   step: int, *, lr: float, beta1: float, beta2: float,
                   epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mc = beta1 * m0 + (1.0 - beta1) * gc
    mr = beta1 * m0 + (1.0 - beta1) * gr
    vc = beta2 * v0 + (1.0 - beta2) * gc * gc
    vr = beta2 * v0 + (1.0 - beta2) * gr * gr
    bc1, bc2 = 1.0 - beta1**step, 1.0 - beta2**step

    def update(m: np.ndarray, v: np.ndarray) -> np.ndarray:
        return -lr * (m / bc1) / (np.sqrt(v / bc2) + epsilon)

    ucc, ucr, urc, urr = update(mc, vc), update(mc, vr), update(mr, vc), update(mr, vr)
    total = ucc - urr
    numerator = 0.5 * ((ucc - urc) + (ucr - urr))
    denominator = 0.5 * ((ucc - ucr) + (urc - urr))
    return total, numerator, denominator


def analyze(payload: dict) -> dict:
    if payload.get("schema") != "kernel-analyzer-bound-endpoint-raw-stage-v1" or payload.get("status") != "COMPLETE":
        raise ValueError("a complete bound-endpoint raw capture is required")
    vectors = payload["vectors"]
    names = ("candidate_gradient", "repair_gradient", "candidate_first_moment_before_step",
             "candidate_second_moment_before_step")
    rows = {name: [np.asarray(row, dtype=np.float64) for row in vectors[name]] for name in names}
    count = len(payload["state_ids"])
    if count != 32 or any(len(value) != count for value in rows.values()):
        raise ValueError("the frozen analysis requires 32 complete states")
    optimizer = payload["optimizer"]
    beta1, beta2 = map(float, optimizer["betas"])
    kwargs = dict(lr=float(optimizer["learning_rate"]), beta1=beta1, beta2=beta2,
                  epsilon=float(optimizer["epsilon"]))
    components = {"total": [], "first_moment_numerator": [], "second_moment_denominator": []}
    crossing = {name: [] for name in components}
    noncrossing = {name: [] for name in components}
    max_error = 0.0
    crossing_coordinates = 0
    for index in range(count):
        gc, gr = rows["candidate_gradient"][index], rows["repair_gradient"][index]
        total, numerator, denominator = decompose_step(
            gc, gr, rows["candidate_first_moment_before_step"][index],
            rows["candidate_second_moment_before_step"][index], index + 1, **kwargs)
        max_error = max(max_error, float(np.max(np.abs(total - numerator - denominator))))
        mask = (gc * gr <= 0.0) & (gc != gr)
        crossing_coordinates += int(mask.sum())
        for name, value in zip(components, (total, numerator, denominator)):
            components[name].append(value)
            crossing[name].append(np.where(mask, value, 0.0))
            noncrossing[name].append(np.where(mask, 0.0, value))
    resultants = {name: np.sum(value, axis=0) for name, value in components.items()}
    total_norm2 = float(np.dot(resultants["total"], resultants["total"]))
    shares = {
        name: float(np.dot(resultants[name], resultants["total"]) / max(total_norm2, 1e-300))
        for name in ("first_moment_numerator", "second_moment_denominator")
    }
    return {
        "schema": "kernel-analyzer-adamw-response-components-v1", "status": "COMPLETE",
        "case_id": payload["case_id"], "state_ids": payload["state_ids"], "optimizer": optimizer,
        "coordinate_count": int(components["total"][0].size),
        "sign_crossing_coordinate_events": crossing_coordinates,
        "max_additive_reconstruction_error": max_error,
        "components": {name: sequence_stats(value) for name, value in components.items()},
        "signed_share_along_total_resultant": shares,
        "sign_crossing_components": {name: sequence_stats(value) for name, value in crossing.items()},
        "noncrossing_components": {name: sequence_stats(value) for name, value in noncrossing.items()},
        "interpretation": (
            "The first-moment numerator and second-moment denominator terms are a symmetric "
            "two-factor decomposition of the same captured AdamW update contrast. Their signed "
            "shares may be negative or exceed one when the terms oppose each other."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = analyze(payload)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
