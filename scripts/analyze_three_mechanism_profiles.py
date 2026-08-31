#!/usr/bin/env python3
"""Produce one local/gradient/AdamW profile for the three selected mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


MAX_ANALYSIS_COORDINATES = 4096


def _tensor_vector(value: torch.Tensor) -> np.ndarray:
    flat = value.detach().float().reshape(-1).numpy()
    if flat.size <= MAX_ANALYSIS_COORDINATES:
        return flat
    # Deterministic CountSketch: every coordinate contributes, while memory
    # stays bounded for Gemma's large projection carrier.
    result = np.zeros(MAX_ANALYSIS_COORDINATES, dtype=np.float64)
    for start in range(0, flat.size, 1_000_000):
        stop = min(flat.size, start + 1_000_000)
        indices = np.arange(start, stop, dtype=np.uint64)
        buckets = ((indices * np.uint64(11400714819323198485)) >> np.uint64(32)) % MAX_ANALYSIS_COORDINATES
        signs = np.where(((indices * np.uint64(7046029254386353131)) >> np.uint64(63)) == 0, 1.0, -1.0)
        result += np.bincount(buckets.astype(np.int64), weights=flat[start:stop] * signs,
                              minlength=MAX_ANALYSIS_COORDINATES)
    return result.astype(np.float32)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype(np.float32, copy=False).tobytes()).hexdigest()


def _bootstrap_ci(values: np.ndarray, *, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.5, 0.975])]


def _signflip_p(values: np.ndarray, *, draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    count = 0
    for _ in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        count += abs(float((values * signs).mean())) >= observed
    return (count + 1.0) / (draws + 1.0)


def _profile(u: list[np.ndarray], r: list[np.ndarray], *, seed: int) -> dict:
    if len(u) != 32 or len(r) != 32:
        raise ValueError("profile requires exactly 32 states")
    if any(x.shape != u[0].shape for x in u + r):
        raise ValueError("stage vectors do not share one coordinate system")
    calibration_u = np.stack(u[:16]).astype(np.float64)
    confirmation_u = np.stack(u[16:]).astype(np.float64)
    confirmation_r = np.stack(r[16:]).astype(np.float64)
    mean_cal = calibration_u.mean(axis=0)
    norm = float(np.linalg.norm(mean_cal))
    if norm == 0.0:
        raise ValueError("zero calibration direction")
    direction = mean_cal / norm
    repair_scale = math.sqrt(float(np.mean(np.sum(confirmation_r * confirmation_r, axis=1))))
    if repair_scale <= 1e-12:
        raise ValueError("repair-side scale is below the frozen 1e-12 RMS floor")
    signed = confirmation_u @ direction / repair_scale
    repair_energy = np.sum(confirmation_r * confirmation_r, axis=1)
    usable = repair_energy > max(float(np.median(repair_energy)) * 1e-12, 1e-30)
    if int(usable.sum()) < 12:
        raise ValueError("too few states above repair energy floor")
    aligned = np.sum(confirmation_u[usable] * confirmation_r[usable], axis=1) / repair_energy[usable]
    residual = confirmation_u[usable] - aligned[:, None] * confirmation_r[usable]
    cal_r = np.stack(r[:16]).astype(np.float64)
    cal_energy = np.sum(cal_r * cal_r, axis=1)
    cal_aligned = np.sum(calibration_u * cal_r, axis=1) / np.maximum(cal_energy, 1e-30)
    cal_residual = calibration_u - cal_aligned[:, None] * cal_r
    residual_direction = cal_residual.mean(axis=0)
    residual_norm = float(np.linalg.norm(residual_direction))
    residual_signed = (
        residual @ (residual_direction / residual_norm) / repair_scale
        if residual_norm > 0 else np.zeros(len(residual))
    )
    gram_u = np.stack(u).astype(np.float64) @ np.stack(u).astype(np.float64).T
    gram_r = np.stack(r).astype(np.float64) @ np.stack(r).astype(np.float64).T
    gram_ur = np.stack(u).astype(np.float64) @ np.stack(r).astype(np.float64).T
    return {
        "state_count": 32,
        "calibration_count": 16,
        "confirmation_count": 16,
        "coordinate_count": int(u[0].size),
        "effect_vector_digests": [_digest(x) for x in u],
        "repair_vector_digests": [_digest(x) for x in r],
        "total_effect_rms": float(np.sqrt(np.mean(np.sum(confirmation_u ** 2, axis=1)))),
        "repair_rms": repair_scale,
        "mean_effect_over_repair_rms": float(np.linalg.norm(confirmation_u.mean(axis=0)) / repair_scale),
        "additive_heldout_effect": {
            "estimate": float(signed.mean()),
            "bootstrap_95": _bootstrap_ci(signed, draws=4000, seed=seed),
            "signflip_p": _signflip_p(signed, draws=4000, seed=seed + 1),
        },
        "aligned_effect": {
            "estimate": float(aligned.mean()),
            "bootstrap_95": _bootstrap_ci(aligned, draws=4000, seed=seed + 2),
            "signflip_p": _signflip_p(aligned, draws=4000, seed=seed + 5),
            "usable_states": int(usable.sum()),
        },
        "orthogonal_heldout_effect": {
            "estimate": float(residual_signed.mean()),
            "bootstrap_95": _bootstrap_ci(residual_signed, draws=4000, seed=seed + 3),
            "signflip_p": _signflip_p(residual_signed, draws=4000, seed=seed + 4),
        },
        "joint_gram": {"G_uu": gram_u.tolist(), "G_rr": gram_r.tolist(), "G_ur": gram_ur.tolist()},
    }


def _local_vectors(manifest: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    effects, repairs = [], []
    for row in manifest["observer_gradient_records"]:
        records = row["observer_records"]
        target = next(x for x in records if x.get("repaired_endpoints") or "same_dtype_directional_sketch" in x)
        raw = target.get("raw_capture", {})
        repaired = target.get("repaired_endpoints", [])
        if raw and repaired:
            endpoint = repaired[0]
            candidate = torch.load(raw[endpoint]["candidate"], map_location="cpu", weights_only=True)
            reference = torch.load(raw[endpoint]["reference"], map_location="cpu", weights_only=True)
            repair = reference.to(candidate.dtype).float()
            effects.append(_tensor_vector(candidate.float() - repair))
            repairs.append(_tensor_vector(repair))
            continue
        if "same_dtype_count_sketch" in target:
            sketch = target["same_dtype_count_sketch"]
        elif "same_dtype_directional_sketch" in target:
            sketch = target["same_dtype_directional_sketch"]
        else:
            endpoint = target["repaired_endpoints"][0]
            sketch = target["same_dtype_repair_metrics"][endpoint]["directional_error_sketch"]
        effects.append(np.asarray(sketch.get("effect", sketch.get("signed_delta_values")), dtype=np.float32))
        repairs.append(np.asarray(sketch.get("repair", sketch.get("reference_values")), dtype=np.float32))
    return effects, repairs


def _gradient_vectors(manifest: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    effects, repairs = [], []
    for row in manifest["formation_gradient_records"]:
        value = torch.load(row["path"], map_location="cpu", weights_only=True)
        effects.append(_tensor_vector(value["gradient_difference"]))
        repairs.append(_tensor_vector(value["repair_gradient"]))
    return effects, repairs


def _update_vectors(manifest: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    effects, repairs = [], []
    for row in manifest["trajectory_update_records"]:
        value = torch.load(row["path"], map_location="cpu", weights_only=True)
        candidate = _tensor_vector(value["candidate_update_at_candidate_state"])
        repair = _tensor_vector(value["repair_update_at_candidate_state"])
        effects.append(candidate - repair)
        repairs.append(repair)
    return effects, repairs


def _holm(rows: list[tuple[str, str, float]]) -> dict[str, float]:
    ordered = sorted(rows, key=lambda x: x[2])
    adjusted, running = {}, 0.0
    m = len(ordered)
    for rank, (case, stage, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[f"{case}::{stage}"] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case_dirs = sorted(p for p in args.root.iterdir() if p.is_dir() and (p / "raw_manifest.json").exists())
    output = {"schema": "kernel-analyzer-three-mechanism-profiles-v1", "cases": {}}
    p_rows = []
    for case_dir in case_dirs:
        manifest = json.loads((case_dir / "raw_manifest.json").read_text())
        stages = {}
        for index, (name, loader) in enumerate((
            ("LOCAL", _local_vectors), ("GRADIENT", _gradient_vectors), ("ADAMW_UPDATE", _update_vectors)
        )):
            u, r = loader(manifest)
            stages[name] = _profile(u, r, seed=20260831 + index)
            for branch in ("additive_heldout_effect", "aligned_effect", "orthogonal_heldout_effect"):
                p_rows.append((case_dir.name, f"{name}::{branch}", stages[name][branch]["signflip_p"]))
        output["cases"][case_dir.name] = {"stages": stages, "raw_manifest": str(case_dir / "raw_manifest.json")}
    corrected = _holm(p_rows)
    for case, value in output["cases"].items():
        for stage, profile in value["stages"].items():
            for branch in ("additive_heldout_effect", "aligned_effect", "orthogonal_heldout_effect"):
                profile[branch]["holm_p_across_three_cases_stages_and_effect_types"] = corrected[f"{case}::{stage}::{branch}"]
    output["multiple_testing_family"] = "three cases x three stages x additive/aligned/orthogonal tests; Holm family-wise correction"
    output_path = args.output or (args.root / "summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
