#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a compatible CUDA device for ForkCert.")
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--require-flash", action="store_true")
    parser.add_argument("--fallback-any", action="store_true", help="Select the freest CUDA device when no preferred device exists.")
    args = parser.parse_args()

    import torch

    candidates = []
    inventory = []
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        compatible = (not args.require_bf16 or capability[0] >= 8) and (
            not args.require_flash or capability[0] >= 8
        )
        row = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(capability),
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "compatible": compatible,
        }
        inventory.append(row)
        if compatible:
            candidates.append(row)
    if not candidates:
        if args.fallback_any and inventory:
            selected = max(inventory, key=lambda row: int(row["free_bytes"]))
            print(
                json.dumps(
                    {"selected": selected, "preferred_match": False, "devices": inventory},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            print(selected["index"])
            return
        print(json.dumps({"selected": None, "devices": inventory}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
    selected = max(candidates, key=lambda row: int(row["free_bytes"]))
    print(
        json.dumps({"selected": selected, "compatible_count": len(candidates), "preferred_match": True}, sort_keys=True),
        file=sys.stderr,
    )
    print(selected["index"])


if __name__ == "__main__":
    main()
