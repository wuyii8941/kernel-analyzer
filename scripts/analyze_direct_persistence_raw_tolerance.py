#!/usr/bin/env python3
"""Compute tolerance metrics from an external raw Gemma replay.

The replay keeps tensors outside the repository.  This script reads them one
at a time, writes only compact metrics, and marks a level unavailable when the
raw candidate/reference pair was not saved.  It never treats a missing level
as zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


RTOL_VALUES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
ATOL_VALUES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite_pair(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left = candidate.detach().reshape(-1)
    right = reference.detach().reshape(-1)
    mask = torch.isfinite(left.float()) & torch.isfinite(right.float())
    return left[mask].float(), right[mask].float()


def ulp_distance(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Return exact ULP distances for finite values of the stored dtype."""

    if candidate.dtype != reference.dtype:
        # The comparison is still meaningful, but exact ULP is defined in the
        # candidate representation and must not silently compare mixed widths.
        return {"status": "ABSTAIN_MIXED_DTYPES", "candidate_dtype": str(candidate.dtype), "reference_dtype": str(reference.dtype)}
    if candidate.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return {"status": "ABSTAIN_UNSUPPORTED_DTYPE", "dtype": str(candidate.dtype)}
    left = candidate.detach().reshape(-1)
    right = reference.detach().reshape(-1)
    mask = torch.isfinite(left.float()) & torch.isfinite(right.float())
    left = left[mask]
    right = right[mask]
    if not left.numel():
        return {"status": "COMPLETE", "finite_count": 0, "max": 0, "mean": 0.0, "p95": 0.0}
    signed_dtype = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}[candidate.dtype]
    width = {torch.float16: 16, torch.bfloat16: 16, torch.float32: 32, torch.float64: 64}[candidate.dtype]
    mask_bits = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    left_bits = left.view(signed_dtype).to(torch.int64) & mask_bits
    right_bits = right.view(signed_dtype).to(torch.int64) & mask_bits
    left_ordered = torch.where((left_bits & sign_bit) != 0, (~left_bits) & mask_bits, left_bits | sign_bit)
    right_ordered = torch.where((right_bits & sign_bit) != 0, (~right_bits) & mask_bits, right_bits | sign_bit)
    distances = (left_ordered - right_ordered).abs().double()
    return {
        "status": "COMPLETE",
        "finite_count": int(distances.numel()),
        "max": int(distances.max().item()),
        "mean": float(distances.mean().item()),
        "p95": float(torch.quantile(distances, 0.95).item()),
        "dtype": str(candidate.dtype),
    }


def metric_row(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if not isinstance(candidate, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError("raw pair must contain tensors")
    if candidate.shape != reference.shape:
        raise ValueError("raw pair shape mismatch")
    left, right = finite_pair(candidate, reference)
    delta = left - right
    denom = torch.linalg.vector_norm(right).item()
    return {
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
        "coordinate_count": int(candidate.numel()),
        "finite_count": int(delta.numel()),
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(delta * delta)).item()) if delta.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item() / max(denom, 1e-30)),
        "ulp": ulp_distance(candidate, reference),
    }


def aggregate_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        return {"status": "ABSTAIN_MISSING_RAW_PAIRS", "label": label, "states": 0}
    all_ulp = [row["ulp"] for row in rows if row["ulp"].get("status") == "COMPLETE"]
    return {
        "status": "COMPLETE",
        "label": label,
        "states": len(rows),
        "coordinate_count": rows[0]["coordinate_count"],
        "max_abs_max": max(row["max_abs"] for row in rows),
        "rms_mean": sum(row["rms"] for row in rows) / len(rows),
        "l2_mean": sum(row["l2"] for row in rows) / len(rows),
        "relative_l2_mean": (
            sum(row["relative_l2"] for row in rows if row["relative_l2"] is not None)
            / sum(row["relative_l2"] is not None for row in rows)
            if any(row["relative_l2"] is not None for row in rows) else None
        ),
        "ulp_max": max((row["ulp"].get("max", 0) for row in rows), default=None),
        "ulp_mean": sum(row.get("mean", 0.0) for row in all_ulp) / len(all_ulp) if all_ulp else None,
        "per_state": rows,
    }


def magnitude_row(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().reshape(-1).float()
    finite = value[torch.isfinite(value)]
    return {
        "candidate_dtype": str(value.dtype),
        "reference_dtype": None,
        "coordinate_count": int(value.numel()),
        "finite_count": int(finite.numel()),
        "max_abs": float(finite.abs().max().item()) if finite.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(finite * finite)).item()) if finite.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(finite).item()),
        "relative_l2": None,
        "ulp": {"status": "ABSTAIN_NO_REFERENCE"},
    }


def rtol_atol_sweep(pairs: list[tuple[Path, Path]]) -> dict[str, Any]:
    if not pairs:
        return {"status": "ABSTAIN_MISSING_RAW_PAIRS"}
    rows = []
    for rtol in RTOL_VALUES:
        for atol in ATOL_VALUES:
            pass_rates = []
            for candidate_path, reference_path in pairs:
                candidate = torch.load(candidate_path, map_location="cpu", weights_only=True).float()
                reference = torch.load(reference_path, map_location="cpu", weights_only=True).float()
                finite = torch.isfinite(candidate) & torch.isfinite(reference)
                allowed = atol + rtol * reference.abs()
                passed = finite & ((candidate - reference).abs() <= allowed)
                pass_rates.append(float(passed.float().mean().item()))
            rows.append({"rtol": rtol, "atol": atol, "state_pass_rate_mean": sum(pass_rates) / len(pass_rates)})
    return {"status": "COMPLETE", "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_dir = args.raw_dir
    manifest_path = raw_dir / "raw_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing raw manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    target_endpoint = manifest.get("target_endpoint", ["out_ptr3"])[0]
    formation_pairs: list[tuple[Path, Path]] = []
    output_pairs: list[tuple[Path, Path]] = []
    for record in manifest["observer_gradient_records"]:
        if "confirmation" not in record["state_id"]:
            continue
        for observer_record in record["observer_records"]:
            raw = observer_record.get("raw_capture", {}).get(target_endpoint)
            if raw:
                output_pairs.append((Path(raw["candidate"]), Path(raw["reference"])))
    gradient_rows = []
    for record in manifest["formation_gradient_records"]:
        payload = torch.load(record["path"], map_location="cpu", weights_only=True)
        gradient_rows.append(metric_row(payload["candidate_gradient"], payload["repair_gradient"]))
        del payload
    gradient_metrics = aggregate_rows(gradient_rows, "parameter_gradient")
    update_rows = []
    for record in manifest["trajectory_update_records"]:
        payload = torch.load(record["path"], map_location="cpu", weights_only=True)
        candidate = payload["local_update"]
        update_rows.append(magnitude_row(candidate))
        del payload, candidate
    # This is a magnitude baseline only: local_update is not a candidate vs
    # repair bitwise pair, so ULP/rtol are intentionally not reported here.
    update_metrics = aggregate_rows(update_rows, "local_effective_update_magnitude")
    update_metrics["ulp_status"] = "ABSTAIN_NO_CANDIDATE_REPAIR_UPDATE_PAIR"
    result = {
        "schema": "kernel-analyzer-direct-persistence-v4-raw-tolerance-v1",
        "status": "PARTIAL_RAW_TOLERANCE_COMPLETE",
        "case_id": manifest["case_id"],
        "target_endpoint": target_endpoint,
        "raw_manifest": str(manifest_path),
        "raw_manifest_sha256": digest(manifest_path),
        "output": aggregate_rows(
            [metric_row(torch.load(left, map_location="cpu", weights_only=True), torch.load(right, map_location="cpu", weights_only=True)) for left, right in output_pairs],
            "operator_output_candidate_vs_reference",
        ),
        "gradient": gradient_metrics,
        "update": update_metrics,
        "rtol_atol": rtol_atol_sweep(output_pairs),
        "claim_boundary": "Output and gradient have exact stored candidate/reference pairs. Update magnitudes are available, but update ULP/rtol require storing candidate and repair update tensors separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output_states": len(output_pairs), "gradient_states": len(gradient_rows), "update_states": len(update_rows)}))


if __name__ == "__main__":
    main()
