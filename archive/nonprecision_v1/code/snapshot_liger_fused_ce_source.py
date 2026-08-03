#!/usr/bin/env python3
"""Snapshot the exact installed Liger fused-CE forward/backward sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DATA_ROOT = Path("/data1/tzh").resolve()


def checked(path: Path) -> Path:
    result = path.resolve()
    if DATA_ROOT not in (result, *result.parents):
        raise RuntimeError(f"path must stay under /data1/tzh: {result}")
    return result


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    certificate_path = checked(args.certificate)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    certificate = json.loads(certificate_path.read_text())
    if certificate["verdict"] != "NATURAL_LIGER_FUSED_CE_BF16_DW_ACCUMULATION_BIAS_CONFIRMED":
        raise RuntimeError("fused-CE certificate differs")

    import liger_kernel.ops.cross_entropy as cross_entropy
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    import liger_kernel.transformers.fused_linear_cross_entropy as wrapper
    import torch
    import triton

    sources = []
    for role, module in (
        ("fused_forward_backward", fused),
        ("cross_entropy_triton", cross_entropy),
        ("module_wrapper", wrapper),
    ):
        path = Path(module.__file__).resolve()
        text = path.read_text()
        sources.append(
            {
                "role": role,
                "installed_path": str(path),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-source-snapshot.v1",
        "status": "COMPLETE",
        "environment": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "liger_kernel_root": str(Path(fused.__file__).resolve().parents[1]),
        },
        "bound_program_facts": {
            "chunk_schedule_expression": "inc_factor=cdiv(V,H); chunk_size=next_power_of_2(cdiv(BT,inc_factor)); loop over num_chunks",
            "default_accumulator": "torch.zeros_like(weight) followed by grad_weight += mm(chunk).float(), hence BF16 storage when weight is BF16",
            "intervention": "accum_dtype=torch.float32 changes only grad_weight storage/addition before one final cast to weight.dtype",
            "actual_backward": "the custom forward saves grad_input and grad_weight; the custom backward returns these saved tensors after optional scalar grad_output multiplication",
        },
        "sources": sources,
        "bindings": {
            "certificate": {
                "path": str(certificate_path),
                "sha256": hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
            }
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sources": len(sources), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
