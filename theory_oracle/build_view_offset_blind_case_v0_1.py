#!/usr/bin/env python
"""Build an opaque two-shape case package for the view-region replay slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def view_chain(value):
    dout, din = value.shape
    return value.view(-1, 1).view(dout, din)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    warm = torch.randn(4, 128, 16, device="cuda", dtype=torch.float32)
    target = torch.randn(4, 171, 6, device="cuda", dtype=torch.float32)
    reference = torch.stack([view_chain(row) for row in target])
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "input.pt"
    reference_path = out / "reference.pt"
    torch.save({"warm": warm.cpu(), "target": target.cpu()}, input_path)
    torch.save(reference.cpu(), reference_path)
    manifest = {
        "schema_version": "forkcert.blind_case_package.v0.1",
        "case_id": "case_002",
        "visibility": "patch_free_opaque_case",
        "contract": {
            "reference_role": "declared_reference_execution",
            "endpoint": "target per-row output tensor relation after prescribed warm calls",
            "compiled_output_must_match_reference": True,
            "warm_calls_are_part_of_the_execution_context": True,
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "warm_shape": list(warm.shape),
            "target_shape": list(target.shape),
            "dtype": str(target.dtype),
            "device_at_capture": str(target.device),
            "seed": 0,
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "locator_exclusions": [
            "issue identifier",
            "fixed revision",
            "patch",
            "pull-request discussion",
            "root-cause notes",
        ],
    }
    (out / "case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
