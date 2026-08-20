#!/usr/bin/env python3
"""Summarize dynamic-state generated screens without promoting error to bias."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import gzip
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=32)
    args = parser.parse_args()
    if args.top < 1:
        raise ValueError("top must be positive")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    state_rows = []
    observed_mode = None
    for path in sorted(args.inputs):
        payload = load(path)
        schema = payload["schema"]
        if schema == "kernel-analyzer-generated-nontriton-fp32-screen-v1":
            mode, complete = "nontriton", "COMPLETE_SHARD_ALL_NONTRITON_FP32_REPLAY"
            campaign_by_region = {}
        elif schema in {
            "kernel-analyzer-generated-fp32-screen-v1",
            "kernel-analyzer-generated-typed-fp32-screen-v2",
        }:
            mode, complete = "triton", "COMPLETE_SHARD_ALL_TRITON_FP32_REPLAY"
            campaign = load(ROOT / payload["campaign"])
            campaign_by_region = {row["region_id"]: row for row in campaign["rows"]}
        else:
            raise RuntimeError(f"unsupported generated screen schema: {schema}")
        observed_mode = observed_mode or mode
        if observed_mode != mode:
            raise RuntimeError("Triton and non-Triton screens require separate summaries")
        if payload["status"] != complete:
            raise RuntimeError(f"incomplete screen: {path}")
        if len(payload["states"]) != 1:
            raise RuntimeError("state-specific summary requires one state per input")
        state_id, state = next(iter(payload["states"].items()))
        repeats = state["repeats"]
        if len(repeats) != 2:
            raise RuntimeError(f"formal screen lacks two repeats: {state_id}")
        left, right = repeats[0]["summary"], repeats[1]["summary"]
        left_identity = left.get("runtime_identity") or [
            (
                row["region_id"], row["symbol"], row.get("runtime_invocation_ordinal"),
                row.get("callsite_execution_ordinal"), tuple(sorted(row["endpoint_metrics"])),
            ) for row in left["records"]
        ]
        right_identity = right.get("runtime_identity") or [
            (
                row["region_id"], row["symbol"], row.get("runtime_invocation_ordinal"),
                row.get("callsite_execution_ordinal"), tuple(sorted(row["endpoint_metrics"])),
            ) for row in right["records"]
        ]
        if left_identity != right_identity:
            raise RuntimeError(f"runtime identity changed across repeats: {state_id}")
        static_expected = (
            left["denominator"]["static_generated_compute_calls"]
            if mode == "nontriton" else left["denominator"]["expected_triton_invocations"]
        )
        static_missing = (
            left["denominator"]["static_calls_not_executed_in_measured_step"]
        )
        state_rows.append({
            "state_id": state_id,
            "actual_invocations": len(left["records"]),
            "static_calls": static_expected,
            "static_calls_not_executed": static_missing,
        })
        for record in left["records"]:
            campaign_row = campaign_by_region.get(str(record["region_id"]), {})
            signature = (
                str(record["phase"]), str(record.get("function", record.get("symbol"))),
                str(record.get("source_line_sha256", campaign_row.get("source_line_sha256"))),
            )
            for endpoint, metric in record["endpoint_metrics"].items():
                rms = float(metric["rms"])
                reference_rms = float(metric["reference_rms"])
                grouped[signature].append({
                    "state_id": state_id,
                    "region_id": str(record["region_id"]),
                    "endpoint": endpoint,
                    "relative_rms": rms / max(reference_rms, 1e-30),
                    "signed_mean_over_rms": abs(float(metric["signed_mean"])) / max(rms, 1e-30),
                    "rms": rms,
                })
        del payload, state, repeats, left, right
        gc.collect()

    rows = []
    for (phase, function, source_digest), values in grouped.items():
        relative = [row["relative_rms"] for row in values]
        signed = [row["signed_mean_over_rms"] for row in values]
        rows.append({
            "phase": phase,
            "function": function,
            "source_line_sha256": source_digest,
            "actual_invocations": len(values),
            "states_observed": len({row["state_id"] for row in values}),
            "region_ids": sorted({row["region_id"] for row in values}),
            "median_relative_rms": statistics.median(relative),
            "max_relative_rms": max(relative),
            "median_abs_signed_mean_over_rms": statistics.median(signed),
            "max_abs_signed_mean_over_rms": max(signed),
            "max_rms": max(row["rms"] for row in values),
        })
    rows.sort(key=lambda row: (
        row["max_relative_rms"], row["max_abs_signed_mean_over_rms"],
        row["phase"], row["function"], row["source_line_sha256"],
    ), reverse=True)
    eligible = [
        row for row in rows
        if row["states_observed"] == len(state_rows)
        and (
            row["max_relative_rms"] >= 0.002
            or row["max_abs_signed_mean_over_rms"] >= 0.02
        )
    ]
    output = {
        "schema": "kernel-analyzer-state-specific-generated-screen-summary-v1",
        "status": "COMPLETE_SCREENING_SUMMARY_NO_BIAS_VERDICT",
        "implementation_kind": observed_mode,
        "states": state_rows,
        "denominator": {
            "screening_states": len(state_rows),
            "actual_invocations": sum(row["actual_invocations"] for row in state_rows),
            "unique_syntactic_callsites": len(rows),
            "all_actual_invocations_retained": True,
        },
        "shortlist_rule": (
            "Observed in every screening state and max relative RMS >= 0.002 or "
            "max |signed mean|/RMS >= 0.02. This prioritizes orbit replay only; "
            "it is not a directional-bias verdict."
        ),
        "shortlist": eligible,
        "top_callsites": rows[:args.top],
        "claim_boundary": (
            "FP32-storage precision residual screening. Conditional mean, temporal "
            "persistence, F+B transport, and TCMP remain untested here."
        ),
    }
    if len(state_rows) != len({row["state_id"] for row in state_rows}):
        raise RuntimeError("screening state repeated")
    if not all(math.isfinite(value) for row in rows for value in (
        row["median_relative_rms"], row["max_relative_rms"],
        row["median_abs_signed_mean_over_rms"], row["max_abs_signed_mean_over_rms"],
    )):
        raise RuntimeError("screen summary is nonfinite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **output["denominator"], "shortlist": len(eligible)}))


if __name__ == "__main__":
    main()
