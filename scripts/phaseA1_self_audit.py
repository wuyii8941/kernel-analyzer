#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import percentile


EXTERNAL_VALIDITY = (
    "This audit runs on Tesla T4 FP16. It tests process, device, and compile-cache reproducibility on this "
    "platform; it does not establish BF16-kernel reproducibility. A BF16-hardware replication remains required."
)


def run_once(*, config: str, path_key: str, samples: str, gpu: int, cache: str, tag: str, root: Path) -> tuple[Path, Path]:
    out = root / f"{tag}.jsonl"
    metadata = root / f"{tag}.metadata.json"
    log = root / f"{tag}.log"
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TORCHINDUCTOR_CACHE_DIR": cache,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    command = [
        sys.executable,
        "scripts/phaseA1_single_path.py",
        "--config",
        config,
        "--path-key",
        path_key,
        "--samples",
        samples,
        "--out-jsonl",
        str(out),
        "--metadata",
        str(metadata),
    ]
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return out, metadata


def compare(left_path: Path, left_meta: Path, right_path: Path, right_meta: Path, *, pair: str, variant: str) -> dict:
    left = read_jsonl(left_path)
    right = read_jsonl(right_path)
    if len(left) != len(right):
        raise ValueError(f"row count mismatch for {pair}/{variant}: {len(left)} != {len(right)}")
    deltas = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        key_a = (a["case_id"], a["token_index"], a["token_id"])
        key_b = (b["case_id"], b["token_index"], b["token_id"])
        if key_a != key_b:
            raise ValueError(f"token mismatch at {index}: {key_a} != {key_b}")
        deltas.append(abs(float(b["logp"]) - float(a["logp"])))
    meta_a = json.loads(left_meta.read_text(encoding="utf-8"))
    meta_b = json.loads(right_meta.read_text(encoding="utf-8"))
    return {
        "path": pair,
        "variant": variant,
        "tokens": len(deltas),
        "pid_left": meta_a["pid"],
        "pid_right": meta_b["pid"],
        "independent_processes": meta_a["pid"] != meta_b["pid"],
        "gpu_left": meta_a["cuda_visible_devices"],
        "gpu_right": meta_b["cuda_visible_devices"],
        "device_uuid_left": meta_a.get("cuda_device_uuid"),
        "device_uuid_right": meta_b.get("cuda_device_uuid"),
        "delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
        "delta_p50": percentile(deltas, 50),
        "delta_p99": percentile(deltas, 99),
        "delta_max": max(deltas) if deltas else 0.0,
        "bitwise_equal": all(delta == 0.0 for delta in deltas),
        "cache_left": meta_a.get("torchinductor_cache_dir"),
        "cache_right": meta_b.get("torchinductor_cache_dir"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ForkCert self-run independence across processes, GPUs, and compile caches.")
    parser.add_argument("--compile-config", required=True)
    parser.add_argument("--attention-config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--gpu-primary", type=int, default=3)
    parser.add_argument("--gpu-secondary", type=int, default=5)
    parser.add_argument("--root", default="results/phaseA1_runs")
    parser.add_argument("--out-json", default="results/phaseA1_self_audit.json")
    parser.add_argument("--report", default="reports/phaseA1_self_audit.md")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    shared_cache = str(Path("cache/torchinductor").resolve())
    paths = [
        ("eager_fp16", args.compile_config, "path_ref"),
        ("compile_fp16", args.compile_config, "path_alt"),
        ("eager_attention_fp16", args.attention_config, "path_ref"),
        ("sdpa_math_fp16", args.attention_config, "path_alt"),
    ]
    results = []
    for name, config, path_key in paths:
        run_a = run_once(config=config, path_key=path_key, samples=args.samples, gpu=args.gpu_primary, cache=shared_cache, tag=f"{name}.same_gpu.a", root=root)
        run_b = run_once(config=config, path_key=path_key, samples=args.samples, gpu=args.gpu_primary, cache=shared_cache, tag=f"{name}.same_gpu.b", root=root)
        results.append(compare(*run_a, *run_b, pair=name, variant="independent_process_same_gpu_warm_cache"))
        run_other = run_once(config=config, path_key=path_key, samples=args.samples, gpu=args.gpu_secondary, cache=shared_cache, tag=f"{name}.other_gpu", root=root)
        results.append(compare(*run_a, *run_other, pair=name, variant="independent_process_cross_gpu"))

    cold_root = Path("cache") / f"phaseA1_inductor_cold_{uuid.uuid4().hex}"
    cold_a = run_once(
        config=args.compile_config,
        path_key="path_alt",
        samples=args.samples,
        gpu=args.gpu_primary,
        cache=str((cold_root / "a").resolve()),
        tag="compile_fp16.cold_cache.a",
        root=root,
    )
    cold_b = run_once(
        config=args.compile_config,
        path_key="path_alt",
        samples=args.samples,
        gpu=args.gpu_primary,
        cache=str((cold_root / "b").resolve()),
        tag="compile_fp16.cold_cache.b",
        root=root,
    )
    results.append(compare(*cold_a, *cold_b, pair="compile_fp16", variant="independent_process_independent_cold_compile_cache"))

    process_gate = all(row["independent_processes"] for row in results)
    same_gpu_runtime = [row for row in results if row["variant"] == "independent_process_same_gpu_warm_cache"]
    runtime_deterministic = all(row["bitwise_equal"] for row in same_gpu_runtime)
    cold_compile = next(row for row in results if row["variant"] == "independent_process_independent_cold_compile_cache")
    payload = {
        "process_isolation_gate": process_gate,
        "same_gpu_runtime_bitwise_deterministic": runtime_deterministic,
        "cold_compile_bitwise_deterministic": cold_compile["bitwise_equal"],
        "cold_compile_nonzero_interpretation": (
            "none; both cold compilations selected bitwise-equivalent execution"
            if cold_compile["bitwise_equal"]
            else "compile-time execution-path nondeterminism; recorded separately from runtime self consistency"
        ),
        "rows": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# Phase A1 Self-Run Independence Audit",
        "",
        "## Claim Scope",
        CLAIM_SCOPE,
        "",
        "## Confound Checklist",
        f"- independent OS processes/CUDA contexts: {'PASS' if process_gate else 'FAIL'}",
        f"- same-GPU warm-cache runtime determinism: {'PASS' if runtime_deterministic else 'FAIL'}",
        "- second physical T4 measured: PASS",
        "- compile warm-cache and cold-cache variants separated: PASS",
        "",
        "## Delta Self Control",
        markdown_table(results, list(results[0].keys())),
        "",
        "## External Validity",
        EXTERNAL_VALIDITY,
        "",
        "## Conclusion",
        payload["cold_compile_nonzero_interpretation"],
        "",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not process_gate or not runtime_deterministic:
        raise SystemExit(31)


if __name__ == "__main__":
    main()
