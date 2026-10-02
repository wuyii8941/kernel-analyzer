#!/usr/bin/env python3
"""Audit signed mean evidence for every retained complete-coordinate endpoint.

The retained full-coordinate coverage artifacts retain a complete original
coordinate Gram for each endpoint.  This script uses those records as a
finite, declared empirical bank and performs the same calibration-direction /
with-replacement confirmation calculation used by the case-level audit.

This is deliberately a retrospective endpoint inventory.  It does not turn
the coverage suite into a natural training population, does not make a
family-wise discovery claim, and does not infer a root cause from an endpoint
mean alone.  No vectors are reconstructed or fabricated: rows without a
complete original-coordinate Gram are skipped and counted separately.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "results/coverage/cases/full_coordinate"
OUT = ROOT / "results/property/root_cause_closure_v1/full_coordinate_endpoint_mean_bias_v1.json"


def iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream the top-level rows array without loading multi-gigabyte files."""

    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    row_index = 0
    with gzip.open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if chunk:
                buffer += chunk.decode("utf-8")
            if not started:
                marker = buffer.find('"rows"')
                if marker < 0:
                    if not chunk:
                        return
                    if len(buffer) > 1_000_000:
                        buffer = buffer[-100_000:]
                    continue
                array_start = buffer.find("[", marker)
                if array_start < 0:
                    continue
                buffer = buffer[array_start + 1 :]
                started = True
            while started:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                try:
                    row, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in {path}")
                yield row_index, row
                row_index += 1
                buffer = buffer[end:]
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                elif buffer.startswith("]"):
                    return
            if not chunk:
                return


def iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield row records and targeted top-level Gram certificates."""

    # A few targeted rsqrt captures store one certificate at the top level
    # rather than under the large shard's ``rows`` array.  They are small;
    # loading only these files does not affect the streaming path for the
    # multi-gigabyte shards.
    if path.parent == SOURCE_ROOT:
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        certificate = data.get("certificate")
        if isinstance(certificate, dict):
            yield 0, {
                "_top_level_certificate": certificate,
                "task_id": data.get("task_id"),
                "exact_semantic_endpoint_id": data.get("exact_aot_endpoint_id"),
                "verdict": certificate.get("status"),
            }
            return
    yield from iter_rows(path)


def operator_key(endpoint: Any) -> str:
    value = str(endpoint or "unknown")
    value = value.rsplit(":", 1)[-1]
    value = value.rsplit("__", 1)[-1]
    return re.sub(r"_\d+$", "", value)


def endpoint_mean(row: dict[str, Any], *, seed: int) -> dict[str, Any] | None:
    certificate = row.get("complete_coordinate_certificate") or row.get("_top_level_certificate")
    if not isinstance(certificate, dict):
        return None
    gram_value = certificate.get("gram")
    if not isinstance(gram_value, list) or not gram_value:
        return None
    gram = np.asarray(gram_value, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] < 4:
        return None
    if not np.isfinite(gram).all():
        raise ValueError("complete-coordinate Gram contains a non-finite value")
    n = int(gram.shape[0])
    fixed_mean_norm_sq = max(0.0, float(gram.sum()) / (n * n))
    fixed_cross_u = (float(gram.sum()) - float(np.trace(gram))) / (n * (n - 1))

    # Draws are with replacement from the finite bank, so the calibration
    # sample size is independent of the bank size.  Keeping 32 here matches
    # the case-level empirical-bank audit and makes endpoint rows comparable.
    calibration_count = 32
    confirmation_count = 64
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=calibration_count + confirmation_count)
    calibration = indices[:calibration_count]
    confirmation = indices[calibration_count:]
    calibration_gram = gram[np.ix_(calibration, calibration)]
    direction_norm_sq = float(calibration_gram.sum()) / (calibration_count**2)
    result: dict[str, Any] = {
        "bank_state_count": n,
        "fixed_suite_mean_vector_norm": math.sqrt(fixed_mean_norm_sq),
        # A complete Gram determines the mean vector of this finite,
        # declared bank directly.  This is an exact finite-bank descriptor
        # (up to the stored floating-point Gram), not a natural-population
        # inference and not a root-cause claim.
        "finite_bank_mean_vector_nonzero": bool(fixed_mean_norm_sq > 0.0),
        "fixed_suite_cross_state_u_statistic": fixed_cross_u,
        "calibration_count": calibration_count,
        "confirmation_count": confirmation_count,
        "draw_seed": seed,
        "scope": "RETROSPECTIVE_IID_WITH_REPLACEMENT_FROM_DECLARED_COVERAGE_BANK",
        "assumptions": [
            "the retained complete-coordinate Gram represents the declared finite bank",
            "with-replacement index draws are independent conditional on that bank",
            "the calibration direction is fixed before confirmation",
            "finite variance and Student-t approximation for the signed projection",
        ],
    }
    if not math.isfinite(direction_norm_sq) or direction_norm_sq <= 0.0:
        result.update({
            "status": "DIRECTION_NOT_IDENTIFIABLE",
            "implies_vector_mean_nonzero_if_bound_positive": False,
        })
        return result
    direction_norm = math.sqrt(direction_norm_sq)
    values = np.asarray(
        [float(gram[int(index), calibration].sum()) / (calibration_count * direction_norm)
         for index in confirmation],
        dtype=np.float64,
    )
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    two_sided_half = float(t.ppf(0.975, confirmation_count - 1)) * sd / math.sqrt(confirmation_count)
    one_sided_half = float(t.ppf(0.95, confirmation_count - 1)) * sd / math.sqrt(confirmation_count)
    two_sided = [mean - two_sided_half, mean + two_sided_half]
    one_sided_lower = mean - one_sided_half
    result.update({
        "status": "PROJECTED_MEAN_SUPPORTED" if one_sided_lower > 0.0 else "PROJECTED_MEAN_NOT_CONFIRMED",
        "calibration_direction_norm": direction_norm,
        "confirmation_projection_mean": mean,
        "confirmation_projection_sd": sd,
        "confirmation_projection_two_sided_95": two_sided,
        "confirmation_projection_one_sided_95_lower": one_sided_lower,
        "two_sided_95_nonzero_supported": bool(two_sided[0] > 0.0 or two_sided[1] < 0.0),
        "positive_mean_supported_one_sided": bool(one_sided_lower > 0.0),
        "confirmation_positive_count": int(np.count_nonzero(values > 0.0)),
        "confirmation_negative_count": int(np.count_nonzero(values < 0.0)),
        "confirmation_zero_count": int(np.count_nonzero(values == 0.0)),
        "finite_bank_mean_vector_norm": math.sqrt(fixed_mean_norm_sq),
        "implies_vector_mean_nonzero_if_bound_positive": bool(one_sided_lower > 0.0),
    })
    return result


def build() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    source_files = sorted(SOURCE_ROOT.glob("**/*.json.gz"))
    for file_index, path in enumerate(source_files):
        # The large files contain a sizeable metadata section before ``rows``;
        # derive the stable model/length label from the directory instead of
        # loading that prefix.  Row streaming below remains the source of all
        # numerical quantities.
        source_label = path.parent.name
        match = re.match(r"(?P<architecture>.+)_seq(?P<sequence_length>\d+)_r1$", source_label)
        header: dict[str, Any] = match.groupdict() if match else {
            "architecture": source_label,
            "sequence_length": None,
        }
        for row_index, row in iter_records(path):
            certificate = row.get("complete_coordinate_certificate") or row.get("_top_level_certificate")
            if not isinstance(certificate, dict) or not certificate.get("gram"):
                skipped["no_complete_original_coordinate_gram"] += 1
                continue
            metric = endpoint_mean(row, seed=730000 + len(rows))
            if metric is None:
                skipped["invalid_complete_gram"] += 1
                continue
            quality_status = str(certificate.get("status", "UNKNOWN"))
            # A complete Gram can be numerically valid while the enclosing
            # causal-carrier gate fails.  Keep such rows visible, but do not
            # call their empirical-bank projection a supported result.
            quality_ok = quality_status in {"PASS", "VERIFIED", "COMPLETE"}
            if not quality_ok:
                metric["status"] = "QUALITY_GATE_FAILED"
                metric["implies_vector_mean_nonzero_if_bound_positive"] = False
            metric.update({
                "source": str(path.relative_to(ROOT)),
                "source_file_index": file_index,
                "row_index": row_index,
                "architecture": header.get("architecture"),
                "sequence_length": header.get("sequence_length"),
                "task_id": row.get("task_id"),
                "exact_semantic_endpoint_id": row.get("exact_semantic_endpoint_id"),
                "operator_key": operator_key(row.get("exact_semantic_endpoint_id")),
                "original_verdict": row.get("verdict"),
                "quality_status": quality_status,
                "eligible_for_scoped_mean": quality_ok,
                "data_use": "RETROSPECTIVE_COVERAGE_ENDPOINT_AUDIT",
            })
            rows.append(metric)

    by_operator: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["operator_key"])].append(row)
    for operator, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        counts = Counter(row["status"] for row in group)
        by_operator[operator] = {
            "endpoint_count": len(group),
            "status_counts": dict(counts),
            "projected_mean_supported": counts["PROJECTED_MEAN_SUPPORTED"],
            "projected_mean_not_confirmed": counts["PROJECTED_MEAN_NOT_CONFIRMED"],
            "direction_not_identifiable": counts["DIRECTION_NOT_IDENTIFIABLE"],
            "quality_gate_failed": counts["QUALITY_GATE_FAILED"],
        }
    counts = Counter(row["status"] for row in rows)
    return {
        "schema": "kernel-analyzer-full-coordinate-endpoint-mean-bias-v1",
        "status": "COMPLETE_RETROSPECTIVE_ENDPOINT_AUDIT",
        "source_scope": str(SOURCE_ROOT.relative_to(ROOT)),
        "source_file_count": len(source_files),
        "row_count": len(rows),
        "skipped_counts": dict(skipped),
        "scope": "RETROSPECTIVE_IID_WITH_REPLACEMENT_FROM_DECLARED_COVERAGE_BANK",
        "familywise_error_controlled": False,
        "natural_training_population_claim": False,
        "root_cause_claim": False,
        "summary": {
            "complete_coordinate_endpoint_rows": len(rows),
            "finite_bank_mean_vector_nonzero": sum(
                bool(row.get("finite_bank_mean_vector_nonzero", False)) for row in rows
            ),
            "finite_bank_mean_vector_nonzero_quality_valid": sum(
                bool(row.get("finite_bank_mean_vector_nonzero", False))
                and bool(row.get("eligible_for_scoped_mean", False))
                for row in rows
            ),
            "projected_mean_supported": counts["PROJECTED_MEAN_SUPPORTED"],
            "projected_mean_not_confirmed": counts["PROJECTED_MEAN_NOT_CONFIRMED"],
            "direction_not_identifiable": counts["DIRECTION_NOT_IDENTIFIABLE"],
            "quality_gate_failed": counts["QUALITY_GATE_FAILED"],
            "original_verdict_counts": dict(Counter(row.get("original_verdict") for row in rows)),
        },
        "operator_summary": by_operator,
        "rows": rows,
    }


def main() -> None:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "json": str(OUT),
        "source_files": report["source_file_count"],
        "rows": report["row_count"],
        "summary": report["summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
