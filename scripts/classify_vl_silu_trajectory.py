#!/usr/bin/env python3
"""Recompute the SiLU trajectory verdict from retained scalar records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    records = payload["records"]
    steps = [step for step in (1, 8, 16, 32) if step <= len(records)]
    projections = [records[step - 1]["fp32_master_projection"] for step in steps]
    grows = len(records) == 32 and all(
        right > left for left, right in zip(projections, projections[1:])
    )
    payload["directional_projection_checkpoints"] = steps
    payload["directional_projection_strictly_grows"] = grows
    payload["gates"]["directional_live_weight_accumulation"] = grows
    payload["status"] = (
        "PASS_STRICT_FLASH_STYLE_CASE" if grows else
        "FAIL_DIRECTIONAL_ACCUMULATION"
    )
    payload["claim_boundary"] = (
        "One concrete layer-0 SiLU forward and its actual decomposed AOT backward are "
        "isolated while the other 27 SiLU backward programs remain identical between arms. "
        "The repair is nonzero and live weights diverge, but a strict case additionally "
        "requires growth along the step-1 frozen direction at steps 8, 16 and 32."
    )
    payload.pop("result_sha256", None)
    payload["result_sha256"] = canonical(payload)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "steps": steps, "projections": projections,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
