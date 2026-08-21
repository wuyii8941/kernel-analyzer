#!/usr/bin/env python3
"""Freeze 26 distinct natural images for a multimodal TCMP cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/data1/tzh").resolve()
    if not args.output.resolve().is_relative_to(root) or not args.image_dir.resolve().is_relative_to(root):
        raise ValueError("image bank must remain under /data1/tzh")
    dataset = load_dataset("uoft-cs/cifar10", split="train")
    roles = ["ENGINEERING"] * 2 + ["SCREENING"] * 8 + ["CONFIRMATION"] * 16
    args.image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, role in enumerate(roles):
        image = dataset[index]["img"].convert("RGB")
        path = args.image_dir / f"image_{index:02d}.png"
        image.save(path, format="PNG")
        rows.append({
            "state_id": f"{args.cell_id}-{role.lower()}-{index:02d}",
            "role": role,
            "order_within_role": sum(previous == role for previous in roles[:index]),
            "image_path": str(path.resolve()),
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt": "<start_of_image> Describe the image in one sentence.",
            "dataset_index": index,
        })
    payload = {
        "schema": "kernel-analyzer-tcmp-image-input-bank-v1",
        "cell_id": args.cell_id,
        "dataset": "uoft-cs/cifar10:train:first-26-before-measurement",
        "processor_policy": "OFFICIAL_PROCESSOR_DEFAULT_RESOLUTION_FROZEN_AT_PREFLIGHT",
        "states": rows,
        "splits": {"ENGINEERING": 2, "SCREENING": 8, "CONFIRMATION": 16},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "states": len(rows)}))


if __name__ == "__main__":
    main()
