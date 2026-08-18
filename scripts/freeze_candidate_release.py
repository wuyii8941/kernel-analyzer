#!/usr/bin/env python3
"""Freeze the small identity manifest for one reproducible candidate release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers
    import triton

    capture, inventory, campaign = map(read, (args.capture, args.inventory, args.campaign))
    if not capture["repeat_stable"] or capture["proof_tag_summary"]["tags_not_observed"]:
        raise RuntimeError("candidate capture is not releaseable")
    if inventory["runtime_call_audit"]["denominator"]["compute_invocations"] != 1447:
        raise RuntimeError("generated compute denominator changed")
    if campaign["denominator"] != {
        "triton_invocations": 686,
        "reference_adapter_exact": 686,
        "reference_adapter_unresolved": 0,
    }:
        raise RuntimeError("Triton reference denominator is not closed")

    model_files = [
        path for path in (
            args.model / "config.json",
            args.model / "model.safetensors.index.json",
            args.model / "generation_config.json",
        ) if path.exists()
    ]
    payload = {
        "schema": "kernel-analyzer-candidate-release-v1",
        "release": "qwen3-1.7b-seq64-bf16-inductor-v1",
        "status": "STATIC_DENOMINATOR_FROZEN_RUNTIME_NUMERICS_PENDING",
        "configuration": {
            "dtype": "torch.bfloat16",
            "backend": "torch.compile/Inductor",
            "fullgraph": True,
            "dynamic": False,
            "preserve_aot_aten": capture["preserve_aot_aten"],
            "sequence_length": capture["input"]["sequence_length"],
            "discovery_state": capture["input"]["state"],
            "token_ids_sha256": capture["input"]["token_ids_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
            "cuda_build": torch.version.cuda,
        },
        "model": {
            "path": str(args.model.resolve()),
            "identity_files": {
                path.name: file_digest(path) for path in model_files
            },
        },
        "state_design": {
            "path": str(args.state_design.resolve()),
            "file_sha256": file_digest(args.state_design),
        },
        "artifacts": {
            name: {
                "path": str(path.resolve().relative_to(ROOT)),
                "file_sha256": file_digest(path),
                "result_sha256": value["result_sha256"],
            }
            for name, path, value in (
                ("capture", args.capture, capture),
                ("inventory", args.inventory, inventory),
                ("triton_campaign", args.campaign, campaign),
            )
        },
        "denominator": {
            "generated_compute_invocations": 1447,
            "triton": 686,
            "external": 760,
            "direct_aten": 1,
            "triton_exact_static_adapters": 686,
        },
        "claim_boundary": (
            "This freezes one candidate binary-program population and its static "
            "reference adapters. It does not assert runtime numerical correctness."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "result_sha256": payload["result_sha256"]}))


if __name__ == "__main__":
    main()
