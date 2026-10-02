#!/usr/bin/env python3
"""Rank retained natural region candidates by aligned write direction.

The source records are region substitutions and may include upstream effects.
This report is therefore only a search screen: it ranks candidates for a
future same-local-input intervention and never promotes a problem group.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from build_natural_candidate_frontier import _candidate_family


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/case_causal_audit_v1/signature_causal_links.json"
OUTPUT = ROOT / "results/property/case_causal_audit_v1/signature_candidate_aligned_screen.json"


def _normal_interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]] if values else [0.0, 0.0]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1)
    # This is a descriptive search interval, not a population certificate.
    half = 1.96 * math.sqrt(variance / len(values))
    return [mean - half, mean + half]


def _record(row: dict[str, Any]) -> dict[str, Any] | None:
    raw_path = ROOT / str(row.get("raw_artifact") or "")
    if not raw_path.is_file():
        return None
    data = json.loads(raw_path.read_text())
    stats = data.get("original_coordinate_statistics", {}).get("PARAMETER_WRITE")
    if not isinstance(stats, list):
        return None
    ratios: list[float] = []
    rms_values: list[float] = []
    for item in stats:
        if not isinstance(item, dict):
            continue
        repair = float(item.get("repair_energy", 0.0) or 0.0)
        energy = float(item.get("effect_energy", 0.0) or 0.0)
        inner = float(item.get("effect_repair_inner_product", 0.0) or 0.0)
        if repair > 0.0:
            ratios.append(inner / repair)
            rms_values.append(math.sqrt(max(energy, 0.0) / repair))
    if not ratios:
        return None
    mean = math.fsum(ratios) / len(ratios)
    return {
        "candidate_family": _candidate_family(str(row.get("symbol") or ""), str(row.get("carrier") or "")),
        "carrier": row.get("carrier"),
        "symbol": row.get("symbol"),
        "state_count": len(ratios),
        "aligned_mean": mean,
        "aligned_normal_95_interval_descriptive": _normal_interval(ratios),
        "aligned_positive_count": sum(x > 0.0 for x in ratios),
        "aligned_negative_count": sum(x < 0.0 for x in ratios),
        "aligned_zero_count": sum(x == 0.0 for x in ratios),
        "write_rms_mean": math.fsum(rms_values) / len(rms_values),
        "same_local_operands": (row.get("reference_comparison_scope") or {}).get("same_local_operands"),
        "upstream_differences_possible": (row.get("reference_comparison_scope") or {}).get(
            "includes_possible_upstream_differences"
        ),
        "promotion_status": "SCREEN_ONLY_REGION_SUBSTITUTION",
        "required_next_step": "same-local-operand path-preserving single-source intervention",
    }


def build() -> dict[str, Any]:
    source = json.loads(SOURCE.read_text())
    records = [item for row in source.get("rows", []) if (item := _record(row)) is not None]
    records.sort(key=lambda item: abs(float(item["aligned_mean"])), reverse=True)
    return {
        "schema": "signature-candidate-aligned-screen-v1",
        "status": "SCREEN_ONLY_NOT_A_ROOT_CAUSE_COUNT",
        "source_scope": "retained real-model region substitutions",
        "record_count": len(records),
        "all_records_include_possible_upstream_differences": all(
            item["upstream_differences_possible"] is True for item in records
        ),
        "records": records,
        "interpretation": (
            "The aligned ratios are descriptive prioritization signals. They can identify a promising "
            "natural candidate, but cannot establish a kernel root cause or a population bias while the "
            "region substitution permits upstream differences."
        ),
    }


if __name__ == "__main__":
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "record_count": result["record_count"],
        "top": [
            {
                "family": row["candidate_family"],
                "carrier": row["carrier"],
                "aligned_mean": row["aligned_mean"],
                "interval": row["aligned_normal_95_interval_descriptive"],
                "positive": row["aligned_positive_count"],
                "negative": row["aligned_negative_count"],
            }
            for row in result["records"][:10]
        ],
    }, ensure_ascii=False, indent=2))
