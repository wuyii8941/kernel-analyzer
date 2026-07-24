#!/usr/bin/env python
"""Force-materialized trace wrapper for the original Qwen3 candidate inventory."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import qwen3_original_compiled_inventory_v0_1 as base


REQUIRED_BASENAMES = {
    "fx_graph_readable.py",
    "fx_graph_transformed.py",
    "ir_pre_fusion.txt",
    "ir_post_fusion.txt",
    "output_code.py",
}


def scan_trace(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    files = []
    counts = {name: 0 for name in REQUIRED_BASENAMES}
    trace_dir = out_dir / "inductor_trace"
    for path in sorted(item for item in trace_dir.rglob("*") if item.is_file()):
        if path.name in counts and "forward" in str(path.parent):
            counts[path.name] += 1
        files.append(
            {
                "path": str(path.relative_to(out_dir)),
                "size": path.stat().st_size,
                "sha256": base.file_sha256(path),
            }
        )
    return files, counts


def main() -> None:
    if "--out-dir" not in sys.argv:
        raise ValueError("--out-dir is required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    cache_dir = Path(tempfile.mkdtemp(prefix="forkcert-qwen3-inventory-v02-", dir="/tmp"))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    try:
        base.main()
        result_path = out_dir / "result.json"
        result = json.loads(result_path.read_text())
        files, counts = scan_trace(out_dir)
        trace_complete = all(count >= 2 for count in counts.values())
        real_tensor_like = [row for row in files if "real_tensor" in row["path"].lower()]
        result["schema_version"] = "forkcert.qwen3-original-compiled-inventory.v0.2"
        result["fresh_compile_cache_forced"] = True
        result["temporary_compile_cache_deleted_after_trace"] = True
        result["trace"] = {
            "save_real_tensors": False,
            "file_count": len(files),
            "total_bytes": sum(row["size"] for row in files),
            "required_forward_artifact_counts": counts,
            "files": files,
        }
        result["gates"]["trace_artifacts_complete"] = trace_complete
        result["gates"]["real_tensor_trace_absent"] = not real_tensor_like
        result["status"] = (
            "VALID_ORIGINAL_CANDIDATE_KERNEL_INVENTORY"
            if all(result["gates"].values())
            else "INVALID_OR_INCOMPLETE_KERNEL_INVENTORY"
        )
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "schema_version": result["schema_version"],
                    "status": result["status"],
                    "gates": result["gates"],
                    "actual_graph_family": result["actual_graph_family"],
                    "observed_scorer_sha256": result["observed_scorer_sha256"],
                    "trace_file_count": len(files),
                    "trace_total_bytes": result["trace"]["total_bytes"],
                    "required_forward_artifact_counts": counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
