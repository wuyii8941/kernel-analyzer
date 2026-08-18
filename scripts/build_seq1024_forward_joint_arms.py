#!/usr/bin/env python3
"""Select every independently closed query/key forward invocation as one group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ENTRYPOINTS = {
    "forkcert.attention_query_rotary_norm_forward_reference:replay_query_rotary_norm_forward",
    "forkcert.attention_key_rotary_norm_reference:replay_key_rotary_norm_forward",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path(
        "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/"
        "triton_online_reference_campaign_v1.json"
    ))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/final/long_trigger_forward_joint_arms.json"),
    )
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    rows = [
        {
            "region_id": row["region_id"],
            "reference_entrypoint": row["reference_entrypoint"],
            "source_nodes": row["source_nodes"],
        }
        for row in campaign["rows"]
        if row.get("reference_entrypoint") in ENTRYPOINTS
    ]
    # The canonical atlas has 28 key closures and 27 query closures.  Layer-0
    # query is part of a distinct fused boundary and is deliberately not
    # smuggled into this independently proved query/key group.
    if len(rows) != 55 or len({row["region_id"] for row in rows}) != 55:
        raise RuntimeError(f"expected 55 unique closed query/key forward regions, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "regions": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
