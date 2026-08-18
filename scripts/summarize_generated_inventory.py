#!/usr/bin/env python3
"""Create bounded status sidecars for complete generated inventories."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.inputs:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        if data["status"] != "COMPLETE_GENERATED_SCHEDULE_AND_POINTER_DATAFLOW":
            raise RuntimeError(f"inventory is incomplete: {path}")
        payload = {
            "schema": "kernel-analyzer-executed-generated-inventory-summary-v1",
            "status": data["status"], "architecture": data.get("architecture", "qwen"),
            "result_sha256": data["result_sha256"],
            "denominator": data["runtime_call_audit"]["denominator"],
            "source_inventory": str(path.resolve()),
            "source_inventory_bytes": path.stat().st_size,
        }
        output = path.with_name(path.name.removesuffix(".json.gz") + ".summary.json")
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        temporary.replace(output)
        print(json.dumps({"output": str(output), "bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
