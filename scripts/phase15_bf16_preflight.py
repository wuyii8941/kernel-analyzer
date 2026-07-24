#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def classify_device(
    *,
    device_count: int,
    capability: tuple[int, int] | None,
    bf16_supported: bool,
    require_flash: bool,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if device_count != 1:
        errors.append(f"expected exactly one visible CUDA device, got {device_count}")
    if capability is None:
        errors.append("no CUDA device capability is available")
    elif capability[0] < 8:
        errors.append(f"native BF16 requires SM80 or newer, got SM{capability[0]}{capability[1]}")
    if not bf16_supported:
        errors.append("torch reports native BF16 unsupported")
    if require_flash and (capability is None or capability[0] < 8):
        errors.append("Flash/SDPA validation requires SM80 or newer")
    return not errors, errors


def inspect_runtime(require_flash: bool) -> dict[str, Any]:
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    capability = torch.cuda.get_device_capability(0) if count == 1 else None
    bf16_supported = bool(torch.cuda.is_bf16_supported()) if count == 1 else False
    passed, errors = classify_device(
        device_count=count,
        capability=capability,
        bf16_supported=bf16_supported,
        require_flash=require_flash,
    )
    return {
        "schema_version": "forkcert.bf16-preflight.v1",
        "passed": passed,
        "errors": errors,
        "require_flash": require_flash,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_device_count": count,
        "device": (
            {
                "name": torch.cuda.get_device_name(0),
                "capability": list(capability),
                "bf16_supported": bf16_supported,
            }
            if count == 1 and capability is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed native BF16 hardware gate for ForkCert.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-flash", action="store_true")
    args = parser.parse_args()

    payload = inspect_runtime(args.require_flash)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
