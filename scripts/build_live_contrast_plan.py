#!/usr/bin/env python3
"""Freeze the 11-cell execution plan for all pending non-Triton candidates."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
RUNTIME_PYTHON = Path("/data1/tzh/miniconda3/envs/pt_nightly/bin/python")
CONFIG = {
    "qwen3_1p7b": ("qwen", ROOT.parent / "models/Qwen/Qwen3-1.7B", "qwen"),
    "mamba_130m": ("mamba", ROOT.parent / "models/state-spaces/mamba-130m-hf", "mamba"),
    "phi4_mini_3p8b": ("phi", ROOT.parent / "models/microsoft/Phi-4-mini-instruct", "phi4"),
    "deepseek_r1_0528_qwen3_8b": (
        "deepseek8", ROOT.parent / "models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek8b"
    ),
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    queue = json.loads((COVERAGE / "bias_candidate_queue.json").read_text())
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in queue["candidates"]:
        if row["claim"] == "PENDING_EXHAUSTIVE_FULL_COORDINATE_AND_FB_BINDING":
            grouped.setdefault((row["architecture"], int(row["sequence_length"])), []).append(row)
    cells = []
    for (architecture, seq), candidates in sorted(grouped.items()):
        cli_arch, model, prefix = CONFIG[architecture]
        release = COVERAGE / "runtime_releases" / f"{prefix}_seq{seq}_r1"
        capture = json.loads((release / "capture.json").read_text())
        inventory_path = release / "inventory.json.gz"
        with gzip.open(inventory_path, "rt") as handle:
            inventory = json.load(handle)
        identities = {
            str(row.get("source_line_sha256"))
            for row in inventory["runtime_call_audit"]["rows"]
            if row.get("category") == "COMPUTE"
        }
        missing = sorted(
            row["candidate_id"] for row in candidates
            if row["exact_generated_call"]["source_line_sha256"] not in identities
        )
        if missing:
            raise RuntimeError(f"candidate source identity drift in {architecture}/seq{seq}: {missing}")
        input_bank = (
            ROOT / "results/mamba_scan/input_bank.json"
            if architecture == "mamba_130m" and seq == 64 else
            COVERAGE / f"{prefix}_seq{seq}_input_bank.json"
        )
        output = COVERAGE / "live_contrasts" / f"{prefix}_seq{seq}.json"
        executed = Path(capture["modules"][0]["executed_source"])
        historical_cache = executed.parents[1]
        # Absolute kernel paths are embedded in generated wrapper bytes.  The
        # original cache root must therefore be reused even when it was cleaned
        # after capture and needs to be recreated.
        cache_dir = historical_cache
        args = [
            str(RUNTIME_PYTHON), "scripts/run_live_contrast_cell.py",
            "--architecture", cli_arch, "--sequence-length", str(seq),
            "--model", str(model), "--input-bank", str(input_bank),
            "--release-dir", str(release), "--output", str(output),
            "--states", "32", "--repeat", "2",
        ]
        cells.append({
            "architecture": architecture, "sequence_length": seq,
            "candidate_count": len(candidates),
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "model": str(model), "input_bank": str(input_bank),
            "release_dir": str(release), "output": str(output),
            "torchinductor_cache_dir": str(cache_dir),
            "source_identities_valid": True, "argv": args,
            "runtime_environment_policy": {
                "python_executable": str(RUNTIME_PYTHON),
                "bind_only_after_exact_wrapper_source_validation": True,
            },
            "resource_policy": {
                "one_cell_process_at_a_time": True,
                "minimum_data1_free_gib": 50,
                "temporary_root": "/data1/tzh/cache/kernel_analyzer_contrasts",
                "delete_full_vectors_after_gram": True,
            },
        })
    payload = {
        "schema": "kernel-analyzer-live-contrast-plan-v1",
        "status": "READY_FOR_GPU_PREFLIGHT",
        "queue_sha256": queue["result_sha256"],
        "cell_count": len(cells), "candidate_count": sum(row["candidate_count"] for row in cells),
        "execution_order": sorted(cells, key=lambda row: (
            1 if row["architecture"] == "deepseek_r1_0528_qwen3_8b" else 0,
            row["candidate_count"], row["sequence_length"],
        )),
        "claim_boundary": "The plan runs missing contrasts only; it does not rerun execution census or old ABI-invalid data.",
    }
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "live_contrast_plan.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "cells": len(cells),
                      "candidates": payload["candidate_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
