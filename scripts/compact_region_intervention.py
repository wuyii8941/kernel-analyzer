#!/usr/bin/env python3
"""Pack one or more region-intervention runs without retaining tensor values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [json.loads(path.read_text()) for path in args.input]
    if not runs:
        raise ValueError("at least one run is required")
    first = runs[0]
    keys = ("seq_len", "dtype", "tf32")
    for run in runs[1:]:
        if any(run.get(key) != first.get(key) for key in keys):
            raise ValueError("all input runs must have the same intervention configuration")
    rows = []
    all_region_ids: set[str] = set()
    for source_path, run in zip(args.input, runs):
        selected = set(run["intervention_region_ids"])
        if not selected:
            raise ValueError("input run has no explicit intervention region")
        all_region_ids.update(selected)
        for repeat in run["repeats"]:
            records = [record for record in repeat["records"] if record["region_id"] in selected]
            if len(records) != len(selected):
                raise ValueError("selected intervention records are incomplete")
            rows.append({
                "input": source_path.name,
                "requested_region_ids": sorted(selected),
                "repeat_id": repeat["repeat_id"],
                "loss": repeat["loss"],
                "intervention_observed_region_ids": repeat["intervention_observed_region_ids"],
                "carrier": repeat["intervention_carrier_parameter_stats"],
                "carrier_sketch": repeat.get("intervention_carrier_sketch"),
                "regions": [
                    {
                        "region_id": record["region_id"],
                        "phase": record["phase"],
                        "symbol": record["symbol"],
                        "reference_symbol": record["reference_symbol"],
                        "intervened_endpoints": record.get("intervened_endpoints", []),
                        "endpoint_metrics": {
                            name: {
                                key: metric.get(key)
                                for key in (
                                    "exact", "candidate_finite", "reference_finite",
                                    "rms", "max_abs", "signed_mean", "nonzero_elements",
                                )
                            }
                            for name, metric in record["endpoint_metrics"].items()
                        },
                    }
                    for record in records
                ],
            })
    parameter = first.get("carrier_parameter_names", ["model.embed_tokens.weight"])[0]
    fixed_values = [
        float(row["carrier_sketch"][parameter]["pilot_cosine"])
        for row in rows
        if row.get("carrier_sketch", {}).get(parameter, {}).get("pilot_cosine") is not None
    ]
    fixed_threshold = 0.05
    output = {
        "schema": "kernel-analyzer-region-intervention-pilot-v1",
        "subject": first.get("subject"),
        "configuration": {
            key: first.get(key) for key in ("seq_len", "dtype", "tf32")
        },
        "checkpoint_steps": sorted({run.get("checkpoint_step") for run in runs}),
        "checkpoint_parameter_sha256": {
            str(run.get("checkpoint_step")): run.get("checkpoint_parameter_sha256")
            for run in runs
        },
        "intervention_region_ids": sorted(all_region_ids),
        "carrier_parameter_names": first.get("carrier_parameter_names", []),
        "candidate_blind": True,
        "reference_replacement": "copy exact same-input reference outputs at generated boundary before downstream kernels",
        "repeat_count": len(rows),
        "fixed_projection_threshold": fixed_threshold,
        "fixed_projection_values": fixed_values,
        "projection_signs": sorted({
            -1
            if row["carrier_sketch"][parameter].get("pilot_cosine", 0.0) < 0
            else 1
            for row in rows
            if row.get("carrier_sketch", {}).get(parameter, {}).get("pilot_cosine") is not None
            and abs(row["carrier_sketch"][parameter].get("pilot_cosine", 0.0)) >= fixed_threshold
        }),
        "rows": rows,
        "gates": {
            "all_requested_regions_observed": all(
                set(row["intervention_observed_region_ids"]) == set(row["requested_region_ids"])
                for row in rows
            ),
            "tensor_values_retained": False,
            "natural_bias_case_added": False,
            "multi_checkpoint_coherence_tested": False,
            "fixed_direction_screened": any(
                row.get("carrier_sketch", {}).get(parameter, {}).get("pilot_cosine") is not None
                for row in rows
            ),
            "fixed_direction_near_zero_count": sum(
                abs(value) < fixed_threshold for value in fixed_values
            ),
        },
        "boundary": "A region-level causal pilot only. It measures local-to-carrier transmission at one checkpoint and does not certify directional bias or a complete natural case.",
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
