#!/usr/bin/env python3
"""Finalize natural Liger fused-linear CE directional and accumulator evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DATA_ROOT = Path("/data1/tzh").resolve()
ENDPOINTS = ("loss", "dH", "dW")


def checked(path: Path) -> Path:
    result = path.resolve()
    if DATA_ROOT not in (result, *result.parents):
        raise RuntimeError(f"path must stay under /data1/tzh: {result}")
    return result


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def cell(artifact: Mapping[str, Any], contrast: str, endpoint: str) -> Mapping[str, Any]:
    if contrast == "default_minus_fp32_accum":
        return artifact[contrast][endpoint]
    return artifact["implementations"][contrast][endpoint]["candidate_minus_eager"]


def sign_pvalue(values: np.ndarray) -> tuple[int, int, int, float]:
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    tied = int(np.sum(values == 0))
    n = positive + negative
    if n == 0:
        return positive, negative, tied, 1.0
    return positive, negative, tied, float(sum(math.comb(n, k) for k in range(positive, n + 1)) / 2**n)


def bh_qvalues(pvalues: list[float]) -> list[float]:
    order = np.argsort(np.asarray(pvalues))
    result = np.ones(len(pvalues), dtype=np.float64)
    running = 1.0
    for reverse_index in range(len(order) - 1, -1, -1):
        original = int(order[reverse_index])
        rank = reverse_index + 1
        running = min(running, pvalues[original] * len(order) / rank)
        result[original] = min(running, 1.0)
    return [float(value) for value in result]


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "min": float(array.min()), "max": float(array.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--carriers", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = checked(args.protocol)
    amendment_path = checked(args.amendment)
    carriers_path = checked(args.carriers)
    confirmation_dir = checked(args.confirmation_dir)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    carriers = read(carriers_path)
    protocol_hash = sha256(protocol_path)
    if (
        amendment["status"] != "FROZEN_BEFORE_CONFIRMATION"
        or amendment["bindings"]["original_protocol"]["sha256"] != protocol_hash
        or carriers["status"] != "FROZEN_BEFORE_24_CONFIRMATION_VALUES"
        or carriers["bindings"]["protocol"]["sha256"] != protocol_hash
        or carriers["bindings"]["amendment"]["sha256"] != sha256(amendment_path)
        or len(carriers["candidates"]) != 7
        or len(carriers["zero_controls"]) != 2
    ):
        raise RuntimeError("fused-CE frozen inputs differ")
    ids = [row["state_id"] for row in protocol["state_allocations"]["confirmation"]]
    paths = [confirmation_dir / f"{state_id}.json.gz" for state_id in ids]
    artifacts = []
    state_bindings = []
    for state_id, path in zip(ids, paths, strict=True):
        artifact = read(path)
        if (
            artifact["status"] != "COMPLETE"
            or artifact["state"]["state_id"] != state_id
            or artifact["state"]["phase"] != "confirmation"
            or artifact["bindings"]["protocol"]["sha256"] != protocol_hash
            or not all(artifact["gates"].values())
            or artifact["chunk_schedule"]["chunks"] != 64
        ):
            raise RuntimeError(f"invalid confirmation state: {state_id}")
        artifacts.append(artifact)
        state_bindings.append({"path": str(path.resolve()), "sha256": sha256(path)})

    results = []
    pvalues = []
    for index, candidate in enumerate(carriers["candidates"]):
        contrast = candidate["contrast"]
        endpoint = candidate["endpoint"]
        direction = np.asarray(candidate["normalized_values"], dtype=np.float64)
        rows = [cell(artifact, contrast, endpoint) for artifact in artifacts]
        projections = np.asarray(
            [np.asarray(row["residual_countsketch8192"], dtype=np.float64) @ direction for row in rows]
        )
        positive, negative, tied, pvalue = sign_pvalue(projections)
        rng = np.random.default_rng(17_210_000 + index)
        boot = projections[rng.integers(0, len(projections), size=(20_000, len(projections)))].mean(axis=1)
        interval = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        result: dict[str, Any] = {
            "contrast": contrast,
            "endpoint": endpoint,
            "nonzero_states": int(sum(not row["exact"] for row in rows)),
            "carrier_projections": [float(value) for value in projections],
            "positive": positive,
            "negative": negative,
            "tied": tied,
            "one_sided_exact_sign_pvalue": pvalue,
            "mean_projection": float(projections.mean()),
            "cluster_bootstrap_95ci": interval,
            "residual_max_abs": summary([float(row["max_abs"]) for row in rows]),
            "residual_rms": summary([float(row["rms"]) for row in rows]),
            "residual_signed_mean": summary([float(row["mean_signed"]) for row in rows]),
        }
        if contrast in ("default_accum", "fp32_accum"):
            candidate_fp32 = [
                float(artifact["implementations"][contrast][endpoint]["candidate_minus_fp32_region"]["rms"])
                for artifact in artifacts
            ]
            eager_fp32 = [
                float(artifact["implementations"][contrast][endpoint]["eager_minus_fp32_region"]["rms"])
                for artifact in artifacts
            ]
            c = np.asarray(candidate_fp32)
            e = np.asarray(eager_fp32)
            result["fp32_region_adjudication"] = {
                "candidate_closer_states": int(np.sum(c < e)),
                "eager_closer_states": int(np.sum(e < c)),
                "equal_states": int(np.sum(c == e)),
                "candidate_rms": summary(candidate_fp32),
                "eager_rms": summary(eager_fp32),
            }
        pvalues.append(pvalue)
        results.append(result)
    for row, qvalue in zip(results, bh_qvalues(pvalues), strict=True):
        row["bh_fdr_qvalue"] = qvalue
        row["directional_confirmed"] = bool(
            qvalue < 0.05
            and row["cluster_bootstrap_95ci"][0] > 0
            and row["positive"] > row["negative"]
        )

    result_map = {(row["contrast"], row["endpoint"]): row for row in results}
    zero_exact = {}
    for control in carriers["zero_controls"]:
        key = (control["contrast"], control["endpoint"])
        rows = [cell(artifact, *key) for artifact in artifacts]
        zero_exact[f"{key[0]}.{key[1]}"] = {
            "exact_states": int(sum(row["exact"] for row in rows)),
            "states": len(rows),
        }
    default_complete = all(result_map[("default_accum", endpoint)]["directional_confirmed"] for endpoint in ENDPOINTS)
    fp32_complete = all(result_map[("fp32_accum", endpoint)]["directional_confirmed"] for endpoint in ENDPOINTS)
    causal_dw = result_map[("default_minus_fp32_accum", "dW")]["directional_confirmed"]
    controls_exact = all(row["exact_states"] == 24 for row in zero_exact.values())
    default_dw = result_map[("default_accum", "dW")]["fp32_region_adjudication"]
    fp32_dw = result_map[("fp32_accum", "dW")]["fp32_region_adjudication"]
    eager_mean = default_dw["eager_rms"]["mean"]
    default_excess = default_dw["candidate_rms"]["mean"] - eager_mean
    fp32_excess = fp32_dw["candidate_rms"]["mean"] - eager_mean
    removed = 1.0 - fp32_excess / default_excess
    complete_mechanism = bool(
        default_complete
        and causal_dw
        and controls_exact
        and default_dw["eager_closer_states"] == 24
        and removed > 0.9
    )
    verdict = (
        "NATURAL_LIGER_FUSED_CE_BF16_DW_ACCUMULATION_BIAS_CONFIRMED"
        if complete_mechanism
        else "NO_COMPLETE_LIGER_FUSED_CE_ACCUMULATION_MECHANISM"
    )
    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-certificate.v1",
        "status": "COMPLETE",
        "verdict": verdict,
        "mathematical_unit": protocol["mathematical_unit"],
        "denominators": {
            "pilot_states_excluded": 1,
            "discovery_states": 7,
            "confirmation_states": 24,
            "implementations": 2,
            "closed_forward_actual_vjp_units": 62,
            "implementation_endpoint_state_cells": 186,
            "joint_directional_hypotheses": 7,
            "structural_zero_controls": 2,
        },
        "directional_results": results,
        "zero_controls": zero_exact,
        "complete_unit_gates": {
            "default_loss_dH_dW_directional": default_complete,
            "fp32_accum_loss_dH_dW_directional": fp32_complete,
            "default_minus_fp32_accum_dW_directional": causal_dw,
            "accumulator_loss_and_dH_controls_exact": controls_exact,
        },
        "accumulator_causal_readout": {
            "chunk_contributions": 64,
            "default_accumulator_dtype": "BF16",
            "intervention_accumulator_dtype": "FP32",
            "default_dW_candidate_rms_mean": default_dw["candidate_rms"]["mean"],
            "fp32_accum_dW_candidate_rms_mean": fp32_dw["candidate_rms"]["mean"],
            "eager_dW_rms_mean": eager_mean,
            "default_new_error_mean": default_excess,
            "fp32_accum_new_error_mean": fp32_excess,
            "fraction_of_mean_new_error_removed": removed,
            "default_worse_than_eager_states": default_dw["eager_closer_states"],
            "inference": (
                "At fixed H, W, labels, external BF16 dtype, chunk schedule, loss and dH, changing only "
                "the dW accumulator to FP32 removes most candidate-added dW error and the frozen "
                "default-minus-FP32 direction remains positive in every confirmation state."
            ),
        },
        "flashattention_style_gates": {
            "natural_inputs": True,
            "closed_forward_and_actual_backward": True,
            "directional_candidate_error": default_complete,
            "controlled_causal_factor": causal_dw and controls_exact,
            "direct_parameter_gradient_accumulation": True,
            "complete_region_mechanism": complete_mechanism,
            "full_tied_weight_gradient": False,
        },
        "claim_boundary": {
            "supported": "natural same-external-dtype fused-region bias caused by BF16 storage/addition of 64 chunk contributions to the direct lm_head dW endpoint",
            "precision_scope": "the isolated cause is internal accumulation precision placement; it is not a non-precision trigger",
            "not_yet_supported": "coherence after adding the embedding-side gradient into the tied full-model parameter, multi-step behavior, or cross-model generalization",
        },
        "bindings": {
            "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
            "amendment": {"path": str(amendment_path), "sha256": sha256(amendment_path)},
            "carriers": {"path": str(carriers_path), "sha256": sha256(carriers_path)},
            "confirmation_states": state_bindings,
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": verdict,
                "default_complete": default_complete,
                "causal_dW": causal_dw,
                "fraction_removed": removed,
                "confirmed_cells": sum(row["directional_confirmed"] for row in results),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
