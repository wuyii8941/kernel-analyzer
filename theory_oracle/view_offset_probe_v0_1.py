#!/usr/bin/env python
"""Probe the view/storage-offset case at coarse region boundaries.

This is an exploratory, patch-free probe.  It deliberately reports whether a
discrepancy is produced by compiling the complete view chain or by compiling
one isolated view region.  It does not assign a root cause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable


def hash_tensor(value: Any) -> dict[str, Any]:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": int(value.storage_offset()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def delta(left: Any, right: Any) -> dict[str, Any]:
    value = (left.detach().float() - right.detach().float()).abs()
    return {
        "max_abs": float(value.max().item()),
        "mean_abs": float(value.mean().item()),
        "nonzero": int((value != 0).sum().item()),
    }


def exact(left: Any, right: Any) -> bool:
    return bool(torch_equal(left, right))


def torch_equal(left: Any, right: Any) -> Any:
    import torch

    return torch.equal(left.detach(), right.detach())


def view_chain(value: Any) -> Any:
    dout, din = value.shape
    return value.view(-1, 1).view(dout, din)


def view_first(value: Any) -> Any:
    return value.view(-1, 1)


def view_second(value: Any) -> Any:
    dout, din = value.shape
    return value.view(dout, din)


def compile_once(torch: Any, fn: Callable[..., Any], cache: Path, value: Any) -> tuple[Any, Any]:
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend="inductor")
    result = compiled(value)
    torch.cuda.synchronize()
    return result, compiled


def compile_callable(torch: Any, fn: Callable[..., Any], cache: Path) -> Any:
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True, backend="inductor")


def provenance(cache: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    # Torch 2.7 emits generated wrappers directly below the cache namespace;
    # newer releases may put them under ``*.debug/output_code.py``.  Scan both
    # layouts, but only retain Python wrappers that contain the generated call
    # contract so the artifact is auditable rather than a cache file count.
    candidates = sorted(set(cache.glob("**/*.py")) | set(cache.glob("**/*.debug/output_code.py")))
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "def call(args)" not in text:
            continue
        rows.append(
            {
                "output_code": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kernel_paths": re.findall(r"# kernel path: (.+)", text),
                "source_nodes": re.findall(r"# Topologically Sorted Source Nodes: (.+)", text),
                "original_aten": re.findall(r"Original ATen: (.+)", text),
                "wrapper_operations": sorted(set(re.findall(r"\b(reinterpret_tensor|assert_size_stride|view|as_strided)\b", text))),
            }
        )
    return {"artifact_count": len(rows), "artifacts": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # The two shapes are intentionally kept in one process: the shape change
    # is part of the observed trigger, not a hidden global between runs.
    warm_values = torch.randn(4, 128, 16, device="cuda", dtype=torch.float32)
    values = torch.randn(4, 171, 6, device="cuda", dtype=torch.float32)
    root = Path(args.cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    compiled_chain = compile_callable(torch, view_chain, root / "chain")
    compiled_first = compile_callable(torch, view_first, root / "first")
    compiled_second = compile_callable(torch, view_second, root / "second")
    # Preserve the original trigger: all three callables see the warm shape
    # before the target shape.  Recompiling a region per input would test a
    # different compiler program and can hide cache/context-dependent faults.
    for warm in warm_values:
        compiled_chain(warm)
        compiled_first(warm)
        compiled_second(view_first(warm))
    torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(values):
        eager_first = view_first(row)
        eager_chain = view_chain(row)
        compiled_chain_value = compiled_chain(row)
        compiled_first_value = compiled_first(row)
        # The isolated second-view replay receives exactly the eager first-view
        # boundary tensor, so this is a same-input local replay test.
        compiled_second_value = compiled_second(eager_first)
        eager_second = view_second(eager_first)
        rows.append(
            {
                "index": index,
                "input": hash_tensor(row),
                "eager_first": hash_tensor(eager_first),
                "compiled_first": hash_tensor(compiled_first_value),
                "first_exact": exact(eager_first, compiled_first_value),
                "eager_chain": hash_tensor(eager_chain),
                "compiled_chain": hash_tensor(compiled_chain_value),
                "chain_exact": exact(eager_chain, compiled_chain_value),
                "eager_second": hash_tensor(eager_second),
                "compiled_second_on_eager_boundary": hash_tensor(compiled_second_value),
                "second_local_exact": exact(eager_second, compiled_second_value),
                "chain_delta": delta(eager_chain, compiled_chain_value),
                "second_local_delta": delta(eager_second, compiled_second_value),
            }
        )
    report = {
        "schema_version": "forkcert.view_offset_probe.v0.1",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "rows": rows,
        "provenance": {name: provenance(root / name) for name in ("chain", "first", "second")},
        "claim_scope": "exploratory region production probe; no root-cause claim",
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
