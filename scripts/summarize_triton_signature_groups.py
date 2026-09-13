#!/usr/bin/env python3
"""Deduplicate measured Triton signatures without inventing root causes.

The report separates broad catalogue labels, structural signatures, and exact
candidate computations.  Model positions and output pointers belonging to the
same compiled computation do not become independent operator problems.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(catalog: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    catalog_rows = catalog.get("positions", catalog.get("records", []))
    by_identity = {
        (str(Path(row["release"]).resolve()), str(row["task_id"])): row
        for row in catalog_rows
    }
    attempts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if summary.get("schema") not in {
            "triton-signature-runtime-batch-summary-v1",
            "family-first-campaign-summary-v1",
        }:
            raise ValueError("unexpected summary schema")
        for row in summary.get("rows", []):
            identity = (str(Path(row["source_release"]).resolve()), str(row["task_id"]))
            attempts[identity].append(row)
    valid: list[tuple[dict[str, Any], dict[str, Any]]] = []
    status_counts: Counter[str] = Counter()
    for identity, rows in attempts.items():
        valid_rows = [row for row in rows if row.get("measurement_status") == "VALID"]
        if valid_rows:
            digests = {row.get("analysis_sha256") for row in valid_rows}
            if len(digests) != 1:
                raise ValueError("conflicting valid measurements for " + repr(identity))
            if identity not in by_identity:
                raise ValueError("measured task absent from catalogue: " + repr(identity))
            valid.append((by_identity[identity], valid_rows[0]))
            status_counts["VALID"] += 1
        else:
            statuses = {str(row.get("execution_status", "NOT_ASSESSED")) for row in rows}
            status_counts["NO_VALID_MEASUREMENT"] += 1
            for status in statuses:
                status_counts[status] += 1

    exact_groups: dict[tuple[str, str, str, str], list[tuple[dict, dict]]] = defaultdict(list)
    signature_groups: dict[tuple[str, ...], list[tuple[dict, dict]]] = defaultdict(list)
    for catalog_row, evidence in valid:
        exact_key = (
            str(catalog_row.get("release_task_package_sha256")),
            str(catalog_row.get("candidate_region_id")),
            str(catalog_row.get("phase")),
            str(catalog_row.get("symbol")),
        )
        exact_groups[exact_key].append((catalog_row, evidence))
        signature_groups[tuple(evidence.get("signature_key") or ())].append((catalog_row, evidence))

    exact_rows = []
    for key, members in sorted(exact_groups.items()):
        q = [float(e["fixed_suite_parameter_write_total_rms"])
             for _, e in members if e.get("fixed_suite_parameter_write_total_rms") is not None]
        exact_rows.append({
            "release_task_package_sha256": key[0],
            "candidate_region_id": key[1],
            "phase": key[2],
            "symbol": key[3],
            "position_count": len(members),
            "task_ids": sorted({e["task_id"] for _, e in members}),
            "carriers": sorted({str(c.get("carrier")) for c, _ in members}),
            "catalogue_families": sorted({str(c.get("operator_family")) for c, _ in members}),
            "fixed_suite_write_rms_range": [min(q), max(q)] if q else None,
            "root_cause_established": False,
        })
    return {
        "schema": "triton-signature-deduplication-report-v1",
        "scope": "VALID_FIXED_SUITE_SIGNATURE_MEASUREMENTS_ONLY",
        "attempted_release_task_count": len(attempts),
        "status_counts": dict(status_counts),
        "valid_position_count": len(valid),
        "broad_catalogue_family_count": len({c.get("operator_family") for c, _ in valid}),
        "structural_signature_count": len(signature_groups),
        "exact_candidate_computation_count": len(exact_groups),
        "independent_root_cause_count": None,
        "counts_are_not_interchangeable": True,
        "model_layer_shape_or_output_pointer_does_not_create_a_root_cause_group": True,
        "exact_candidate_computations": exact_rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 新 Triton signature 测量的去重结果", "",
        "本表把有效位置、结构 signature、实际编译计算和独立根因分开。",
        "模型、层号、shape 或同一计算的不同输出不会自动增加问题数。", "",
        f"- 有效位置：{report['valid_position_count']}",
        f"- 目录家族：{report['broad_catalogue_family_count']}",
        f"- 结构 signature：{report['structural_signature_count']}",
        f"- 实际编译计算：{report['exact_candidate_computation_count']}",
        "- 独立根因：未由本轮覆盖测量判定", "",
        "| 阶段 | 实际 Triton 计算 | 位置数 | 目录家族 | 参数写入 RMS 范围 |",
        "|---|---|---:|---|---:|",
    ]
    for row in report["exact_candidate_computations"]:
        rms = row["fixed_suite_write_rms_range"]
        rms_text = "N/A" if rms is None else f"{rms[0]:.6g}–{rms[1]:.6g}"
        lines.append(
            f"| {row['phase']} | `{row['symbol']}` | {row['position_count']} | "
            f"{', '.join(row['catalogue_families'])} | {rms_text} |"
        )
    lines += ["", "这些结果只覆盖固定状态集合；不自动证明总体 bias、根因或训练后果。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    outputs = [args.output] + ([args.markdown] if args.markdown else [])
    if any(path.exists() or not path.resolve().is_relative_to(Path("/data1/tzh")) for path in outputs):
        parser.error("Choose new outputs under /data1/tzh")
    result = summarize(_read(args.catalog), [_read(path) for path in args.summary])
    result["input_sha256"] = {
        str(args.catalog.resolve()): _sha(args.catalog),
        **{str(path.resolve()): _sha(path) for path in args.summary},
        str(Path(__file__).resolve()): _sha(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "valid_position_count", "broad_catalogue_family_count",
        "structural_signature_count", "exact_candidate_computation_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
