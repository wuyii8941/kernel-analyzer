#!/usr/bin/env python3
"""Recompute one declared state-population exceedance analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernel_analyzer.training_numerical_analysis import (
    analyze_population_exceedance_artifact,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.output.exists() or not output.is_relative_to(Path("/data1/tzh")):
        parser.error("use a new output below /data1/tzh")
    payload = json.loads(args.raw.read_text())
    protocol = json.loads(args.protocol.read_text())
    result = analyze_population_exceedance_artifact(
        payload, protocol, endpoint=args.endpoint
    )
    result["provenance"].update({
        "raw_artifact": str(args.raw.resolve()),
        "raw_artifact_sha256": _sha(args.raw),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha(args.protocol),
        "recompute_script_sha256": _sha(Path(__file__)),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
