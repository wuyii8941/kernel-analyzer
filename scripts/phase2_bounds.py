#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from forkcert.bounds import ErrorSource, assemble_logprob_bound, legal_sources_valid, phase2_decision
from forkcert.io import read_jsonl
from forkcert.report import markdown_table, write_phase_report
from forkcert.stats import percentile


def load_source_payload(path: Path) -> tuple[list[ErrorSource], str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_sources = data["sources"] if isinstance(data, dict) else data
    kind = str(data.get("certificate_kind", "unverified")) if isinstance(data, dict) else "unverified"
    metadata = data if isinstance(data, dict) else {}
    return [ErrorSource(**item) for item in raw_sources], kind, metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_measurement_coverage(
    source_payload: dict,
    source_names: set[str],
    measurements_path: Path | None,
) -> tuple[bool, dict]:
    if measurements_path is None:
        return False, {"failures": ["--measurements is required for source coverage validation"]}
    rows = read_jsonl(str(measurements_path))
    measured_levels = sorted({str(row.get("level")) for row in rows})
    nonzero_levels = sorted(
        {
            str(row.get("level"))
            for row in rows
            if max(
                abs(float(row.get("final_logprob_delta", 0.0))),
                abs(float(row.get("max_logprob_delta", 0.0))),
                abs(float(row.get("max_activation_diff_l2", 0.0))),
            )
            > 0.0
        }
    )
    coverage = source_payload.get("coverage") or {}
    level_sources = coverage.get("level_sources") or {}
    failures = []
    expected_hash = coverage.get("measurements_sha256")
    actual_hash = file_sha256(measurements_path)
    if expected_hash != actual_hash:
        failures.append(f"measurement hash mismatch: expected={expected_hash}, actual={actual_hash}")
    declared_levels = sorted(str(level) for level in coverage.get("measured_levels", []))
    if declared_levels != measured_levels:
        failures.append(f"measured level mismatch: declared={declared_levels}, actual={measured_levels}")
    for level in nonzero_levels:
        mapped = [str(name) for name in level_sources.get(level, [])]
        if not mapped:
            failures.append(f"nonzero level {level} has no analytic source mapping")
            continue
        unknown = sorted(set(mapped) - source_names)
        if unknown:
            failures.append(f"level {level} maps unknown sources: {unknown}")
    return not failures, {
        "measurements_sha256": actual_hash,
        "measured_levels": measured_levels,
        "nonzero_levels": nonzero_levels,
        "level_sources": level_sources,
        "failures": failures,
    }


def empirical_delta_stats(logprob_jsonl: str | None) -> tuple[float | None, float | None]:
    if not logprob_jsonl:
        return None, None
    rows = read_jsonl(logprob_jsonl)
    deltas = [float(row.get("logprob_delta", abs(float(row["logp_alt"]) - float(row["logp_ref"])))) for row in rows]
    return (percentile(deltas, 99), max(deltas)) if deltas else (None, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 bound-candidate assembly with legal-source provenance validation.")
    parser.add_argument("--sources", required=True, help="JSON file with a sources list for ErrorSource.")
    parser.add_argument("--logprob-jsonl", default=None, help="Optional Phase 1 JSONL for tightness check.")
    parser.add_argument("--measurements", default=None, help="Phase 1.5 JSONL used to prove complete analytic source coverage.")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--out-json", default="results/phase2_bounds.json")
    parser.add_argument("--report", default="reports/phase2.md")
    parser.add_argument("--fail-on-unusable", action="store_true", help="Exit non-zero after writing outputs if the bound cannot be used as a fragile/bug classifier.")
    args = parser.parse_args()

    sources, certificate_kind, source_payload = load_source_payload(Path(args.sources))
    result = assemble_logprob_bound(sources, delta=args.delta).to_json_dict()
    result["certificate_kind"] = certificate_kind
    legal_valid, legal_failures = legal_sources_valid(sources)
    result["legal_source_validation"] = legal_valid
    result["legal_source_failures"] = legal_failures
    coverage_valid, coverage = validate_measurement_coverage(
        source_payload,
        {source.name for source in sources},
        Path(args.measurements) if args.measurements else None,
    )
    result["source_coverage_validation"] = coverage_valid
    result["source_coverage"] = coverage
    empirical_p99, empirical_max = empirical_delta_stats(args.logprob_jsonl)
    if empirical_p99 is not None and empirical_p99 > 0:
        result["empirical_delta_p99"] = empirical_p99
        result["empirical_delta_max"] = empirical_max
        result["tightness_prob_over_p99"] = result["logprob_bound_prob"] / empirical_p99
        result["tightness_worst_over_p99"] = result["logprob_bound_worst"] / empirical_p99
    else:
        result["empirical_delta_p99"] = None
        result["empirical_delta_max"] = None
        result["tightness_prob_over_p99"] = None
        result["tightness_worst_over_p99"] = None

    if certificate_kind != "analytic_legal":
        decision = (
            "DOWNGRADE: source file is not an analytic legal-error certificate; "
            "observed-delta heuristics cannot classify fragile versus bug."
        )
    elif not legal_valid:
        decision = "DOWNGRADE: analytic_legal label supplied, but source assumptions/provenance validation failed."
    elif not coverage_valid:
        decision = "DOWNGRADE: analytic sources do not cover every nonzero Phase 1.5 measurement from the exact input file."
    elif empirical_max is not None and empirical_max > result["logprob_bound_worst"]:
        decision = "VIOLATION: empirical max(delta) exceeds the deterministic worst-case legal bound."
    elif (
        result.get("tightness_worst_over_p99") is None
        or not math.isfinite(float(result["logprob_bound_worst"]))
        or float(result["tightness_worst_over_p99"]) > 1000.0
    ):
        decision = "DOWNGRADE: deterministic worst-case B is too loose for the primary stable/fragile/bug classifier."
    else:
        decision = phase2_decision(result["logprob_bound_prob"], result["empirical_delta_p99"])
    result["decision"] = decision
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        key: result[key]
        for key in [
            "source_count",
            "certificate_kind",
            "legal_source_validation",
            "source_coverage_validation",
            "logprob_bound_prob",
            "logprob_bound_worst",
            "empirical_delta_p99",
            "empirical_delta_max",
            "tightness_prob_over_p99",
            "tightness_worst_over_p99",
        ]
    }
    write_phase_report(
        args.report,
        title="Phase 2 Bound Candidate Audit",
        confound_checklist={
            "top_sources_from_attribution": "required by input sources file",
            "analytic_legal_sources": certificate_kind == "analytic_legal",
            "all_nonzero_measurements_covered": coverage_valid,
            "activation_norms_measured": "required by input sources file",
            "probability_delta_recorded": args.delta,
            "logsumexp_lipschitz_applied": True,
            "primary_region_bound_is_deterministic_worst": True,
        },
        delta_self_summary="Refer to Phase 1 report; Phase 2 consumes only cross-path empirical deltas.",
        summary=decision,
        sections={
            "Tightness": markdown_table([summary], list(summary.keys())),
            "Per Source": markdown_table(result["per_source"], list(result["per_source"][0].keys()) if result["per_source"] else []),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_unusable and not decision.startswith("GO"):
        print(f"Phase 2 classifier gate failed: {decision}", file=sys.stderr)
        raise SystemExit(22)


if __name__ == "__main__":
    main()
