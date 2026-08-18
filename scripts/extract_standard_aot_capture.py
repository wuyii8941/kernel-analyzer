#!/usr/bin/env python3
"""Materialize the rich standard-decomposition AOT graph from a proof capture."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load(args.proof_capture)
    capture = source.get("standard_aot_capture")
    if not capture or capture.get("capture_sha256") != digest({
        key: value for key, value in capture.items() if key != "capture_sha256"
    }):
        raise RuntimeError("proof capture lacks a valid rich standard AOT graph")
    payload = {
        "schema": "kernel-analyzer-standard-aot-capture-release-v1",
        "status": "COMPLETE_STANDARD_AOT_FORWARD_BACKWARD_CAPTURE",
        "architecture": source["architecture"],
        "model": source["model"],
        "input": source["input"],
        "preserve_aot_aten": False,
        "allow_graph_breaks": source.get("allow_graph_breaks", False),
        "capture": capture,
        "proof_capture_result_sha256": source["result_sha256"],
        "claim_boundary": (
            "This is the actual default-decomposition AOT F+B graph supplied to Inductor. "
            "Generated-kernel identity and mathematical proof are separate gates."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "result_sha256": payload["result_sha256"]}))


if __name__ == "__main__":
    main()
