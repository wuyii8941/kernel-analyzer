"""Reanalyse saved evidence without changing historical decisions or protocols.

This is record verification, not independent GPU reproduction. All outputs must
be new and inside this repository. Direction diagnostics are descriptive only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from kernel_analyzer.training_numerical_analysis import analyze_artifact
from scripts.verify_adamw8bit_error_compensation_training import verify, interval

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/numerical_coverage_v1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(x, b, a):
    if not all(math.isfinite(v) for v in (x, b, a)) or x < 0 or b <= 0:
        raise ValueError("Invalid original-coordinate energies")
    if abs(a) > math.sqrt(x * b) + 1e-12 * max(x, b):
        raise ValueError("Inner product violates energy bound")
    q, beta = x / b, a / b
    c = 1 + q + 2 * beta
    if c < -1e-12:
        raise ValueError("Negative candidate energy")
    ratio = math.sqrt(max(0, c))
    return dict(difference_rms=math.sqrt(q), aligned_coefficient=beta,
                candidate_reference_rms_ratio=ratio,
                aggregate_cosine=(1 + beta) / ratio if ratio > 0 else None)


def build():
    rows, sources, errors = [], {}, []
    seen = set()
    for version in (4, 5, 9):
        summary = BASE / f"triton_signature_batches_v{version}/summary_final.json"
        sources[str(summary.relative_to(ROOT))] = sha(summary)
        for row in read(summary)["rows"]:
            if row["execution_status"] != "VALID":
                continue
            analysis = Path(row["analysis_artifact"])
            if str(analysis) in seen:
                raise ValueError("Duplicate measurement")
            seen.add(str(analysis))
            raw = analysis.parent / "raw" / (row["case_id"] + ".json")
            protocol_path = analysis.parents[2] / "protocol.json"
            payload, saved, protocol = read(raw), read(analysis), read(protocol_path)
            checks = dict(analysis_hash=sha(analysis) == row["analysis_sha256"],
                          raw_hash=sha(raw) == saved["provenance"]["raw_sha256"])
            recomputed = analyze_artifact(payload, protocol)
            recomputed["provenance"]["raw_sha256"] = sha(raw)
            checks["production_reanalysis"] = recomputed == saved
            checks["frozen_sources"] = all(Path(p).is_file() and sha(Path(p)) == h
                                          for p, h in protocol.get("source_sha256", {}).items())
            ids = payload["state_ids"]
            selected = payload["confirmation_state_ids"]
            if len(set(ids)) != len(ids) or not selected or not set(selected) <= set(ids):
                raise ValueError("Invalid state selection")
            stats = payload["original_coordinate_statistics"]["PARAMETER_WRITE"]
            if len(stats) != len(ids):
                raise ValueError("Incomplete coordinate statistics")
            stats = [stats[ids.index(i)] for i in selected]
            values = geometry(*(math.fsum(s[k] for s in stats) for k in
                                ("effect_energy", "repair_energy", "effect_repair_inner_product")))
            checks["reported_rms"] = math.isclose(values["difference_rms"],
                row["fixed_suite_parameter_write_total_rms"], rel_tol=1e-12, abs_tol=1e-15)
            if not all(checks.values()):
                errors.append(dict(case_id=row["case_id"], checks=checks))
            rows.append(dict(case_id=row["case_id"], family=row["operator_family"],
                             raw=str(raw.relative_to(ROOT)), raw_sha256=sha(raw),
                             protocol=str(protocol_path.relative_to(ROOT)),
                             protocol_sha256=sha(protocol_path), state_count=len(ids),
                             comparison_scope=payload.get("reference_comparison_scope"),
                             checks=checks, **values))
    training = verify(BASE / "adamw8bit_error_compensation_training_v1")
    gains = training["recomputed"]["paired_values"][1:]
    catalog = read(BASE / "observed_kernel_catalog_with_signature_measurements_summary_v3.json")
    nonzero = [r for r in rows if r["difference_rms"] > 0]
    return dict(schema="review-evidence-reanalysis-v1", scope="RETROSPECTIVE_FIXED_SUITE_ANALYSIS",
                status="RECORDS_RECOMPUTED" if not errors and training["status"] == "VERIFIED" else "ERROR",
                independent_gpu_reproduction=False, errors=errors, sources=sources,
                analyzer_sha256=sha(Path(__file__)), rows=rows,
                counts=dict(valid=len(rows), states=sum(r["state_count"] for r in rows),
                            nonzero=len(nonzero),
                            nonzero_norm_ratio_within_0_1_percent=sum(abs(r["candidate_reference_rms_ratio"]-1)<0.001 for r in nonzero),
                            measured_catalog_categories=dict(Counter(r["family"] for r in rows)),
                            defined_categories=catalog["operator_family_count"],
                            observed_nonempty_categories=sum(v>0 for v in catalog["distinct_position_family_counts"].values())),
                training_verification=training,
                excluding_pilot_stream_sensitivity=dict(data_use="POST_HOC_SENSITIVITY_NOT_NEW_CONFIRMATION",
                    stream_count=len(gains), mean=math.fsum(gains)/len(gains), interval_95=interval(gains)),
                unfinished=["same_execution_path_compensation_control", "incremental_detection_validation",
                            "single_kernel_source_attribution_for_graph_substitution_results"],
                limits=["Negative aligned coefficient is not necessarily total step shrinkage",
                        "Category counts are not independent operator problems",
                        "Saved-record verification does not recreate original GPU execution",
                        "No new population or loss-causality guarantee"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT) or target.exists():
        parser.error("Use a new output file inside kernel-analyzer")
    result = build()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result["status"], counts=result["counts"], errors=result["errors"])))
    if result["status"] == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
