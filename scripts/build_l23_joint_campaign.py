#!/usr/bin/env python3
"""Combine exact seq1024 layer-23 forward and backward endpoint bindings."""

import hashlib
import json
from pathlib import Path


def main() -> None:
    canonical_path = Path(
        "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/"
        "triton_online_reference_campaign_v1.json"
    )
    query_path = Path("results/final/seq1024_query_campaign.json")
    output = Path("results/final/l23_tile_joint_campaign.json")
    canonical = json.loads(canonical_path.read_text())
    query = json.loads(query_path.read_text())
    forward = next(row for row in canonical["rows"] if row["region_id"] == "forward:1356")
    backward = next(row for row in query["rows"] if row["region_id"] == "backward:157")
    # The runtime observer assigns repeated invocations of one generated symbol
    # by ordinal. Preserve the complete repeated-symbol census around each
    # selected invocation; otherwise ordinary earlier layers would be outside
    # the campaign even though only layer 23 is intervened.
    forward_rows = [row for row in canonical["rows"] if row["symbol"] == forward["symbol"]]
    backward_rows = [row for row in query["rows"] if row["symbol"] == backward["symbol"]]
    result = {
        "schema": "kernel-analyzer-l23-tile-joint-campaign-v1",
        "status": "COMPLETE",
        "rows": forward_rows + backward_rows,
        "sources": {
            str(canonical_path): hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
            str(query_path): hashlib.sha256(query_path.read_bytes()).hexdigest(),
        },
        "binding": "exact forward canonical anchors plus exact seq1024 backward anchors",
        "candidate_values_used_to_bind": False,
        "selected_region_ids": ["forward:1356", "backward:157"],
        "repeated_symbol_census": {
            "forward": len(forward_rows),
            "backward": len(backward_rows),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
