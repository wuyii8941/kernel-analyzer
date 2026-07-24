#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from forkcert.config import load_config
from forkcert.detector import clip_active
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from forkcert.stats import mean, percentile
from scripts.phase6_twin_training import batch_response_logps_with_grad, path_config
from scripts.phase8_matched_step import select_fork_batch


def run_path(cfg, samples, cache_dir: Path, max_fusion_size: int | None):
    import torch
    from torch._inductor import config as inductor_config, metrics

    torch._dynamo.reset()
    metrics.reset()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
    load_cfg = replace(cfg, compile_model=False) if cfg.compile_model else cfg
    tokenizer, model = load_hf_path(load_cfg)
    patch = {"max_fusion_size": max_fusion_size} if max_fusion_size is not None else {}
    with inductor_config.patch(patch):
        if cfg.compile_model:
            model = torch.compile(model)
        with torch.no_grad():
            if cfg.compile_model:
                batch_response_logps_with_grad(tokenizer, model, cfg, samples)
            logps, token_ids = batch_response_logps_with_grad(tokenizer, model, cfg, samples)
    result = logps.detach().float().cpu().tolist(), token_ids, int(metrics.generated_kernel_count)
    del model, tokenizer
    cleanup_memory()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-batch semantic audit for one Inductor fusion partition.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--states", default="data/phase6_step5_replay_dump.jsonl")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--max-fusion-size", type=int)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_determinism(0)
    cert = next(row for row in read_jsonl(args.certificates) if row.get("actual_fork") and row["case_id"] == args.case_id and int(row["token_index"]) == args.token_index)
    samples, states, target = select_fork_batch(read_jsonl(args.samples), read_jsonl(args.states), cert)
    config = load_config(args.config)
    ref_cfg, alt_cfg = path_config(config, "path_ref"), path_config(config, "path_alt")
    root = Path(args.cache_root)
    root.mkdir(parents=True, exist_ok=True)
    ref, ref_ids, _ = run_path(ref_cfg, samples, root / "ref", None)
    alt, alt_ids, kernels = run_path(alt_cfg, samples, root / "alt", args.max_fusion_size)
    expected_ids = [int(row["token_id"]) for row in states]
    if ref_ids != alt_ids or ref_ids != expected_ids:
        raise ValueError("token alignment mismatch")
    rows = []
    for index, (r, a, state) in enumerate(zip(ref, alt, states, strict=True)):
        sign = int(state["advantage_sign"])
        old = float(state["old_logp"])
        ref_clip = clip_active(r, old, sign, float(cert["eps"]))
        alt_clip = clip_active(a, old, sign, float(cert["eps"]))
        boundary = math.log1p(float(cert["eps"])) if sign > 0 else math.log1p(-float(cert["eps"]))
        rows.append({
            "flat_index": index, "case_id": state["case_id"], "token_index": state["token_index"],
            "logp_ref": r, "logp_alt": a, "signed_delta": a - r,
            "signed_margin_ref": r - old - boundary, "clip_ref": ref_clip, "clip_alt": alt_clip,
            "actual_branch_fork": ref_clip != alt_clip,
        })
    deltas = [abs(row["signed_delta"]) for row in rows]
    summary = {
        "schema_version": "forkcert.fusion_batch.v1",
        "fork_id": f"clip-step{cert['metadata']['phase1_metadata']['online_state']['optimizer_step']}-{args.case_id}-t{args.token_index}",
        "max_fusion_size": args.max_fusion_size,
        "tokens": len(rows),
        "generated_kernel_count": kernels,
        "branch_forks_vs_eager": sum(row["actual_branch_fork"] for row in rows),
        "target": rows[target],
        "delta_mean": mean(deltas), "delta_p50": percentile(deltas, 50),
        "delta_p95": percentile(deltas, 95), "delta_p99": percentile(deltas, 99), "delta_max": max(deltas),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
