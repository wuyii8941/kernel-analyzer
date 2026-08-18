#!/usr/bin/env python3
"""Add compact value-bearing semantic observations to the dtype boundary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def main() -> None:
    path = FINAL / "dtype_dynamic_boundary.json"
    boundary = json.loads(path.read_text())
    observations = []
    for dtype, tf32 in (("fp32", False), ("tf32", True)):
        for seq in (64, 128, 256):
            name = f"dtype_semantic_{dtype}_seq{seq}.json"
            data = json.loads((FINAL / name).read_text())
            rows = data["rows"]
            endpoint_count = 0
            persistent = 0
            for row in rows:
                endpoints = {}
                for item in row.get("state_repeat_metrics", []):
                    endpoints.setdefault(str(item["endpoint"]), []).append(
                        float(item["metric"].get("signed_mean", 0.0))
                    )
                for values in endpoints.values():
                    endpoint_count += 1
                    positive = sum(value > 0.0 for value in values)
                    negative = sum(value < 0.0 for value in values)
                    if values and (positive == len(values) or negative == len(values)):
                        persistent += 1
            observations.append({
                "dtype": "fp32",
                "tf32": tf32,
                "seq_len": seq,
                "file": name,
                "checkpoint_steps": data["checkpoint_steps"],
                "repeats_per_checkpoint": 2,
                "runtime_invocations": data["denominator"]["runtime_invocations"],
                "mapped_invocations": data["denominator"]["mapped_invocations"],
                "unresolved_invocations": data["denominator"]["unresolved_invocations"],
                "mapped_regions_observed_at_all_steps_and_repeats": data["denominator"]["mapped_regions_observed_at_all_steps_and_repeats"],
                "rows_with_any_nonexact_endpoint": sum(row["nonexact_record_count"] > 0 for row in rows),
                "endpoint_count": endpoint_count,
                "sign_persistent_endpoint_count": persistent,
                "max_abs_max": max((float(row["max_abs_max"]) for row in rows), default=0.0),
                "candidate_values_used_to_select_or_classify": data["gates"]["candidate_values_used_to_select_or_classify"],
                "natural_bias_case_added": False,
            })
    boundary["semantic_mapping_observations"] = {
        "status": "COMPLETE_FOR_EXACT_MAPPINGS_ALL_FP32_TF32_SHAPES",
        "entries": observations,
        "interpretation": "All exact candidate-blind mappings are observed across the natural checkpoint bank. Unresolved topology rows remain in the denominator and are not safety-certified. Finite residuals are common but directional signs are not a carrier verdict.",
    }
    boundary["topology_census"]["interpretation"] = "All strict-FP32 and TF32 shapes now have generated-topology and exact-mapping records. The mapped subset has value-bearing F+B observations; unresolved topology rows remain an explicit boundary."
    boundary["claim_boundary"]["next_required_artifact"] = "Carrier-level attribution for mapped regions and exact semantic mapping of unresolved FP32/TF32 topology rows"
    path.write_text(json.dumps(boundary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "entries": len(observations)}))


if __name__ == "__main__":
    main()
