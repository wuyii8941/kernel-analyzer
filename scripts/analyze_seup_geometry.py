#!/usr/bin/env python3
"""Analyze exploratory SEUP geometry spools without changing mainline verdicts.

The input spool contains CPU vectors from one carrier-local protocol.  Every
subspace is fit from the frozen calibration rows and evaluated on disjoint
evaluation rows.  The output is a compact JSON certificate; raw vectors are
not copied into it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.geometry import (  # noqa: E402
    GeometryAnalyzer,
    _basis_from_rows,
    _flatten,
    load_spool,
    projected_persistence,
    subspace_overlap,
)


def _split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    calibration = [row for row in rows if row.get("phase") == "calibration"]
    evaluation = [row for row in rows if row.get("phase") == "evaluation"]
    if not calibration or not evaluation:
        raise ValueError("spool must contain non-empty calibration and evaluation rows")
    return calibration, evaluation


def _aggregate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    vectors = [_flatten(row[field]) for row in rows]
    matrix = torch.stack(vectors)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    summed = torch.linalg.vector_norm(matrix.sum(dim=0))
    return {
        "rows": len(rows),
        "sum_l2": float(summed.item()),
        "sum_abs_l2": float(norms.sum().item()),
        "persistence": float((summed / norms.sum().clamp_min(1e-30)).item()),
        "energy_l2_sum": float(torch.sum(matrix * matrix).item()),
    }


def _field_rows(rows: Sequence[Mapping[str, Any]], field: str, key: str) -> list[dict[str, Any]]:
    return [{"phase": row.get("phase"), field: {key: row[field][key]}} for row in rows]


def _head_rows(rows: Sequence[Mapping[str, Any]], field: str, key: str,
               heads: int) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for head in range(heads):
        result[str(head)] = []
    for row in rows:
        value = row[field][key]
        rows_per_head = value.shape[0] // heads
        for head in range(heads):
            result[str(head)].append({
                "phase": row.get("phase"),
                field: {key: value[head * rows_per_head:(head + 1) * rows_per_head]},
            })
    return result


def _factor_subspace(rows: Sequence[Mapping[str, Any]], field: str, key: str,
                     side: str, rank: int) -> tuple[torch.Tensor, list[float]]:
    matrices = [row[field][key].float() for row in rows]
    factors = []
    for matrix in matrices:
        factor = matrix @ matrix.T if side == "left" else matrix.T @ matrix
        factors.append(factor)
    covariance = torch.stack(factors).mean(dim=0)
    values, vectors = torch.linalg.eigh(covariance)
    indices = torch.argsort(values, descending=True)[:rank]
    basis = vectors[:, indices].T.contiguous()
    return basis, [float(values[index].item()) for index in indices]


def _factor_report(calibration: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]],
                   field: str, key: str, rank: int = 4) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        odd_basis, odd_values = _factor_subspace(calibration[::2], field, key, side, rank)
        even_basis, even_values = _factor_subspace(calibration[1::2], field, key, side, rank)
        all_basis, all_values = _factor_subspace(calibration, field, key, side, rank)
        projected_norms = []
        original_norms = []
        projected_sum = None
        for row in evaluation:
            matrix = row[field][key].float()
            if side == "left":
                projected = all_basis.T @ (all_basis @ matrix)
            else:
                projected = (matrix @ all_basis.T) @ all_basis
            projected_norms.append(torch.linalg.vector_norm(projected))
            original_norms.append(torch.linalg.vector_norm(matrix))
            projected_sum = projected if projected_sum is None else projected_sum + projected
        projected_sum_l2 = float(torch.linalg.vector_norm(projected_sum).item()) if projected_sum is not None else 0.0
        result[side] = {
            "odd_even_subspace_overlap": subspace_overlap(odd_basis, even_basis),
            "odd_eigenvalues": odd_values,
            "even_eigenvalues": even_values,
            "all_eigenvalues": all_values,
            "evaluation_projection": {
                "persistence": projected_sum_l2 / max(float(torch.stack(projected_norms).sum().item()), 1e-30),
                "energy_capture": float((torch.sum(torch.stack(projected_norms) ** 2) /
                                          torch.sum(torch.stack(original_norms) ** 2).clamp_min(1e-30)).item()),
                "projected_sum_l2": projected_sum_l2,
            },
        }
    return result


def _rank_grid(analyzer: GeometryAnalyzer, calibration: Sequence[Mapping[str, Any]],
               evaluation: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for field in fields:
        # Local calibration fixes the carrier for all downstream components.
        reports[field] = analyzer.rank_report(
            calibration, evaluation, field=field,
            calibration_field="local" if field != "local" else None,
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = load_spool(str(args.spool))
    rows = sorted(payload["rows"], key=lambda row: int(row["step"]))
    calibration, evaluation = _split(rows)
    analyzer = GeometryAnalyzer()
    fields = ["local", "local_candidate", "feedback", "actual", "gradient_delta", "effective_update"]
    fields = [field for field in fields if field in rows[0]]
    result: dict[str, Any] = {
        "schema": "kernel-analyzer-seup-geometry-v1",
        "case_id": payload.get("case_id"),
        "task_id": payload.get("task_id"),
        "carrier_parameters": payload.get("carrier_parameters", []),
        "protocol": payload.get("protocol"),
        "source_spool": str(args.spool),
        "rows": {"calibration": len(calibration), "evaluation": len(evaluation), "total": len(rows)},
        "rank_reports": _rank_grid(analyzer, calibration, evaluation, fields),
        "window_reports": {
            field: analyzer.window_report(rows, field=field)
            for field in ("local", "actual") if field in fields
        },
        "aggregates": {
            "calibration": {field: _aggregate(calibration, field) for field in fields},
            "evaluation": {field: _aggregate(evaluation, field) for field in fields},
        },
    }

    carriers = payload.get("carrier_parameters", [])
    if len(carriers) > 1:
        factorized: dict[str, Any] = {}
        for key in carriers:
            if key not in rows[0]["local"] or rows[0]["local"][key].ndim != 2:
                continue
            key_result: dict[str, Any] = {"whole": analyzer.rank_report(
                calibration, evaluation, field="local", order=[key]
            ), "factorized": _factor_report(calibration, evaluation, "local", key)}
            rows_per_head = 128 if rows[0]["local"][key].shape[0] % 128 == 0 else 0
            if rows_per_head:
                heads = rows[0]["local"][key].shape[0] // rows_per_head
                key_result["heads"] = {
                    head: analyzer.rank_report(
                        _head_rows(calibration, "local", key, heads)[head],
                        _head_rows(evaluation, "local", key, heads)[head],
                        field="local", order=[key],
                    ) for head in _head_rows(calibration, "local", key, heads)
                }
            factorized[key] = key_result
        result["saved_p_factorization"] = factorized

    result["source_spool_sha256"] = hashlib.sha256(args.spool.read_bytes()).hexdigest()
    serialized = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["result_sha256"] = hashlib.sha256(serialized).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "GEOMETRY_COMPLETE", "case_id": result["case_id"],
                      "output": str(args.output), "rows": result["rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
