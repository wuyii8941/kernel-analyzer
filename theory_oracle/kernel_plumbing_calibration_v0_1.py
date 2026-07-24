#!/usr/bin/env python
"""Seeded generated-kernel plumbing calibration, not a bug-localization result.

The case deliberately has two layers of information:

* the public opaque manifest contains only program/input/reference/contract;
* a calibration-only seed record is written separately after capture.

The future generic locator consumes only the former plus an executable candidate.
This script is intentionally limited to capturing a stable, small Inductor
realization and describing a mutation *slot*.  It never treats that slot as a
discovered compiler bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> dict[str, Any]:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "device": str(value.device),
    }


def program(value: Any) -> Any:
    """A deliberately tiny view/reduction program with one natural boundary."""

    return value.reshape(value.shape[0], -1).sum(dim=-1)


def discover_generated_code(cache_root: Path) -> list[Path]:
    """Return generated Python wrappers containing Triton code, deterministically."""

    candidates = []
    for path in cache_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "@triton.jit" in text and "def triton_" in text:
            candidates.append(path)
    return sorted(candidates, key=lambda item: (item.stat().st_mtime_ns, str(item)))


def provenance_row(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "kernel_names": sorted(set(re.findall(r"def (triton_[A-Za-z0-9_]+)", text))),
        "source_nodes": re.findall(r"# Topologically Sorted Source Nodes: (.+)", text),
        "original_aten": re.findall(r"Original ATen: (.+)", text),
        "has_sum": "tl.sum(" in text,
        "has_view": "view" in text.lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    # Must precede torch import for compiler debug artifacts on supported versions.
    out = args.out_dir.resolve()
    cache_root = out / "inductor_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root)
    os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

    import torch

    out.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # Integer-valued FP32 inputs keep the unmutated reduction exactly
    # representable (the total is far below 2**24).  This is intentional: the
    # plumbing control must start from an exact baseline rather than quietly
    # calibrating against ordinary reduction-order rounding.
    x = torch.randint(-8, 9, (4, 257), device="cuda", dtype=torch.int32).to(torch.float32)
    eager = program(x.clone())
    torch._dynamo.reset()
    compiled = torch.compile(program, backend="inductor")
    candidate = compiled(x.clone())
    candidate_repeat = compiled(x.clone())
    torch.cuda.synchronize()

    input_path = out / "input.pt"
    reference_path = out / "reference.pt"
    torch.save(x.detach().cpu(), input_path)
    torch.save(eager.detach().cpu(), reference_path)
    generated = discover_generated_code(cache_root)
    inventory = [provenance_row(path) for path in generated]
    manifest = {
        "schema_version": "forkcert.kernel-plumbing-opaque-case.v0.1",
        "case_id": "kernel_plumbing_microcase_v0_1",
        "role": "seeded_calibration_only",
        "contract": {
            "endpoint": "exact output tensor equality",
            "reference_role": "declared reference execution",
            "candidate_must_match_reference": True,
        },
        "input": {"path": str(input_path), "sha256": sha256_file(input_path), **fingerprint(x)},
        "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path), **fingerprint(eager)},
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "seed": args.seed,
        },
        "candidate_baseline": {
            "output": fingerprint(candidate),
            "matches_reference": bool(torch.equal(eager, candidate)),
            "repeat_exact": bool(torch.equal(candidate, candidate_repeat)),
        },
        "provenance_inventory": inventory,
        "locator_exclusions": [
            "seeded mutation recipe",
            "target kernel identity",
            "target expression",
            "expected localization answer",
        ],
    }
    (out / "opaque_case_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "case_id": manifest["case_id"],
        "baseline_matches_reference": manifest["candidate_baseline"]["matches_reference"],
        "baseline_repeat_exact": manifest["candidate_baseline"]["repeat_exact"],
        "generated_wrapper_count": len(inventory),
        "sum_capable_wrappers": sum(row["has_sum"] for row in inventory),
        "manifest": str((out / "opaque_case_manifest.json").resolve()),
    }
    (out / "capture_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
