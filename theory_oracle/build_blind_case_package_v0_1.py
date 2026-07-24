#!/usr/bin/env python
"""Create an opaque, patch-free historical case package from a fixed input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def program(torch: Any, value: Any) -> Any:
    pooled = torch.nn.functional.adaptive_avg_pool2d(value, 7)
    return pooled.flatten(1).sum(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    import torch

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    value = torch.randn(4, 2049, 8, 8, dtype=torch.float32, device="cuda")
    reference = program(torch, value.clone())
    torch.cuda.synchronize()
    input_path = out / "input.pt"
    reference_path = out / "reference.pt"
    torch.save(value.detach().cpu(), input_path)
    torch.save(reference.detach().cpu(), reference_path)
    manifest = {
        "schema_version": "forkcert.blind_case_package.v0.1",
        "case_id": "case_001",
        "visibility": "patch_free_opaque_case",
        "contract": {
            "reference_role": "declared_reference_execution",
            "endpoint": "exact per-batch output tensor relation",
            "compiled_output_must_match_reference": True,
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device_at_capture": str(value.device),
            "seed": 0,
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
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
    (out / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
