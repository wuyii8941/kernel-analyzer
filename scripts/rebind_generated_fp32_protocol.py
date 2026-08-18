#!/usr/bin/env python3
"""Audit completed legacy shards and bind the frozen FP32 protocol."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/coverage/generated_fp32_protocol.json"


def validate_metric(metric: dict[str, Any]) -> None:
    if metric.get("schema_version") != "kernel-analyzer.nonfinite-aware-streaming-error.v1":
        raise RuntimeError("shard does not contain the frozen streaming metric")
    if not metric.get("full_value_scan") or metric.get("metric_accumulation_dtype") != "torch.float64":
        raise RuntimeError("metric scan/accumulation differs from protocol")
    sketch = metric["directional_error_sketch"]
    if sketch["selection_rule"] not in {
        "EVENLY_SPACED_FLAT_POSITIONS_FIXED_BEFORE_READING_VALUES", "EMPTY_ENDPOINT",
    } or sketch["candidate_values_used_to_select_coordinates"]:
        raise RuntimeError("directional coordinate selection differs from protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    complete = {
        "kernel-analyzer-generated-fp32-screen-v1": "COMPLETE_SHARD_ALL_TRITON_FP32_REPLAY",
        "kernel-analyzer-generated-nontriton-fp32-screen-v1": "COMPLETE_SHARD_ALL_NONTRITON_FP32_REPLAY",
    }
    for path in args.inputs:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("status") != complete.get(data.get("schema")):
            raise RuntimeError(f"only completed generated FP32 shards can be rebound: {path}")
        if data.get("repeat") != 2:
            raise RuntimeError(f"formal shard repeat count differs: {path}")
        records = 0
        for state in data["states"].values():
            if len(state["repeats"]) != 2:
                raise RuntimeError(f"state repeat denominator differs: {path}")
            for repeat in state["repeats"]:
                summary = repeat["summary"]
                for record in summary["records"]:
                    records += 1
                    for metric in record["endpoint_metrics"].values():
                        validate_metric(metric)
        data["protocol_sha256"] = protocol["protocol_sha256"]
        data["protocol_rebinding_audit"] = {
            "status": "COMPLETE_ARTIFACT_LEVEL_PROTOCOL_AUDIT",
            "records_audited": records,
            "all_metrics_full_streaming_scan": True,
            "all_coordinates_candidate_independent": True,
            "formal_repeat_denominator_exact": True,
            "numerical_values_changed_by_rebinding": False,
        }
        temporary = path.with_name(f".{path.name}.rebind.tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
            json.dump(data, handle, sort_keys=True, separators=(",", ":"))
        with gzip.open(temporary, "rt", encoding="utf-8") as handle:
            written = json.load(handle)
        if written["protocol_sha256"] != protocol["protocol_sha256"]:
            raise RuntimeError("protocol rebind post-write validation failed")
        temporary.replace(path)
        print(json.dumps({"path": str(path), "records_audited": records}))


if __name__ == "__main__":
    main()
