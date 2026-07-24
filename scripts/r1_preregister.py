#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

from forkcert.detector import clip_boundary
from forkcert.io import read_jsonl
from forkcert.stats import mean, percentile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed(path: str, value_field: str) -> dict[tuple[str, int, int], float]:
    return {
        (str(row["case_id"]), int(row["token_index"]), int(row["token_id"])): float(row[value_field])
        for row in read_jsonl(path)
    }


def poisson_ppf(probability: float, expected: float) -> int:
    if expected <= 0:
        return 0
    if expected > 700:
        # R1 is expected to be sparse; this branch only prevents underflow for unexpected inputs.
        from statistics import NormalDist

        return max(0, int(math.floor(expected + math.sqrt(expected) * NormalDist().inv_cdf(probability))))
    term = math.exp(-expected)
    cumulative = term
    value = 0
    while cumulative < probability:
        value += 1
        term *= expected / value
        cumulative += term
    return value


def predicted_rate(margins: list[float], deltas: list[float]) -> float:
    if not margins or not deltas:
        return 0.0
    ordered = sorted(margins)
    return sum(bisect.bisect_left(ordered, delta) / len(ordered) for delta in deltas) / len(deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind preregistration for one R1 held-out state.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--margin-dump", required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--ref-a", required=True)
    parser.add_argument("--ref-b", required=True)
    parser.add_argument("--alt-a", required=True)
    parser.add_argument("--alt-b", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", default="results/phase3_online_calibration.json")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    margin_rows = [
        row for row in read_jsonl(args.margin_dump)
        if int(row["optimizer_step"]) == args.target_step and int(row["advantage_sign"]) != 0
    ]
    margins = [
        abs(
            (float(row["new_logp"]) - float(row["old_logp"]))
            - clip_boundary(int(row["advantage_sign"]), args.eps)
        )
        for row in margin_rows
    ]
    margin_keys = {
        (str(row["case_id"]), int(row["token_index"]), int(row["token_id"])) for row in margin_rows
    }
    ref_a = keyed(args.ref_a, "logp")
    ref_b = keyed(args.ref_b, "logp")
    alt_a = keyed(args.alt_a, "logp")
    alt_b = keyed(args.alt_b, "logp")
    path_keys = set(ref_a)
    if not (path_keys == set(ref_b) == set(alt_a) == set(alt_b)):
        raise ValueError("held-out path key coverage mismatch")
    if not margin_keys <= path_keys:
        raise ValueError("held-out clipping-applicable keys are missing from path scores")
    all_keys = sorted(path_keys)
    keys = sorted(margin_keys)
    self_ref = [abs(ref_b[key] - ref_a[key]) for key in all_keys]
    self_alt = [abs(alt_b[key] - alt_a[key]) for key in all_keys]
    deltas = [abs(alt_a[key] - ref_a[key]) for key in keys]
    deltas_all = [abs(alt_a[key] - ref_a[key]) for key in all_keys]
    if max(self_ref, default=0.0) != 0.0 or max(self_alt, default=0.0) != 0.0:
        raise ValueError("independent-process self gate failed")
    rate = predicted_rate(margins, deltas)
    expected = rate * len(keys)
    calibration = json.loads(Path(args.calibration).read_text())
    inputs = [
        args.margin_dump,
        args.ref_a,
        args.ref_b,
        args.alt_a,
        args.alt_b,
        args.samples,
        args.checkpoint,
        args.calibration,
    ]
    payload = {
        "schema_version": "forkcert.r1.preregistration.v1",
        "name": args.name,
        "prediction_frozen_before_scan": True,
        "blind_contract": (
            "Uses unsigned margin and absolute delta marginal distributions only; does not compute signed crossing, "
            "clip branches, or actual_fork labels."
        ),
        "calibration_model_kind": calibration["model_kind"],
        "calibration_source": args.calibration,
        "calibration_source_predicted_rate": calibration["predicted_fork_rate_late"],
        "independence_assumption": True,
        "eps": args.eps,
        "scored_tokens": len(all_keys),
        "scored_cases": len({key[0] for key in all_keys}),
        "clipping_applicable_tokens": len(keys),
        "clipping_applicable_cases": len({key[0] for key in keys}),
        "target_step": args.target_step,
        "margin_summary": {
            "mean": mean(margins),
            "p1": percentile(margins, 1),
            "p5": percentile(margins, 5),
            "p50": percentile(margins, 50),
            "near_1e-2": sum(value < 1e-2 for value in margins),
        },
        "delta_summary": {
            "mean": mean(deltas),
            "p50": percentile(deltas, 50),
            "p95": percentile(deltas, 95),
            "p99": percentile(deltas, 99),
            "max": max(deltas, default=0.0),
        },
        "delta_summary_all_scored_tokens": {
            "mean": mean(deltas_all),
            "p50": percentile(deltas_all, 50),
            "p95": percentile(deltas_all, 95),
            "p99": percentile(deltas_all, 99),
            "max": max(deltas_all, default=0.0),
        },
        "self_control": {
            "independent_process_ref_p99": percentile(self_ref, 99),
            "independent_process_alt_p99": percentile(self_alt, 99),
            "passed": True,
        },
        "predicted_fork_rate": rate,
        "predicted_fork_count": expected,
        "poisson_95_interval": [poisson_ppf(0.025, expected), poisson_ppf(0.975, expected)],
        "clipping_applicability": "applicable" if keys else "no_nonzero_advantage_decisions",
        "inputs": {path: file_sha256(Path(path)) for path in inputs},
        "preregister_script_sha256": file_sha256(Path(__file__)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
