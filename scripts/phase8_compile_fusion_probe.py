#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from scripts.phase6_twin_training import path_config
from scripts.phase8_case_attribution import run_target, target_batch


SETTINGS = {
    "compile_baseline": {},
    "no_epilogue_fusion": {"epilogue_fusion": False},
    "no_pattern_matcher": {"pattern_matcher": False},
    "max_fusion_size_1": {"max_fusion_size": 1},
    "max_fusion_size_2": {"max_fusion_size": 2},
    "max_fusion_size_4": {"max_fusion_size": 4},
    "max_fusion_size_8": {"max_fusion_size": 8},
    "max_fusion_size_16": {"max_fusion_size": 16},
    "max_fusion_size_32": {"max_fusion_size": 32},
    "no_persistent_reductions": {"triton.persistent_reductions": False},
    "no_split_reductions": {"split_reductions": False},
    "no_reorder_for_locality": {"reorder_for_locality": False},
    "aggressive_fusion": {"aggressive_fusion": True},
    # --- newly added: reduction order and loop scheduling ---
    "no_mix_order_reduction": {"triton.mix_order_reduction": False},
    "no_pick_loop_orders": {"pick_loop_orders": False},
    "no_loop_ordering_after_fusion": {"loop_ordering_after_fusion": False},
    "no_loop_reindexing_after_fusion": {"loop_reindexing_after_fusion": False},
    "no_prologue_fusion": {"prologue_fusion": False},
    "no_batch_fusion": {"batch_fusion": False},
    "deterministic": {"deterministic": True},
    # --- combinations ---
    "deterministic_no_mix_reduction": {"deterministic": True, "triton.mix_order_reduction": False},
    "max_fusion_2_no_mix_reduction": {"max_fusion_size": 2, "triton.mix_order_reduction": False},
    "max_fusion_2_deterministic": {"max_fusion_size": 2, "deterministic": True},
    "full_stabilize": {"max_fusion_size": 2, "deterministic": True, "triton.mix_order_reduction": False, "split_reductions": False},
}


def code_inventory(cache: Path) -> tuple[int, str]:
    files = sorted(cache.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(cache).as_posix().encode())
        digest.update(path.read_bytes())
    return len(files), digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-hook Inductor fusion-class interventions for one natural fork.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--cache-root", default="results/attribution/fusion_probe_cache")
    parser.add_argument("--out", default="results/attribution/step5_compile_fusion_probe.json")
    parser.add_argument("--setting", action="append", choices=sorted(SETTINGS), help="Restrict to selected settings; repeatable.")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_determinism(0)
    config = load_config(args.config)
    cert = next(row for row in read_jsonl(args.certificates) if row.get("actual_fork") and row["case_id"] == args.case_id and int(row["token_index"]) == args.token_index)
    samples = target_batch(read_jsonl(args.samples), cert)
    base_cfg = path_config(config, "path_alt")
    base_cfg = type(base_cfg)(**{**base_cfg.__dict__, "compile_model": False})
    rows = []
    import torch
    from torch._dynamo.utils import counters
    from torch._inductor import config as inductor_config, metrics

    selected_settings = args.setting or list(SETTINGS)
    if "compile_baseline" not in selected_settings:
        selected_settings.insert(0, "compile_baseline")
    for name in selected_settings:
        patch = SETTINGS[name]
        cache = Path(args.cache_root) / name
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache.resolve())
        torch._dynamo.reset()
        counters.clear()
        metrics.reset()
        tokenizer, model = load_hf_path(base_cfg)
        with inductor_config.patch(patch):
            model = torch.compile(model)
            run_target(model, tokenizer, base_cfg, samples, cert)
            value, _ = run_target(model, tokenizer, base_cfg, samples, cert)
        file_count, code_hash = code_inventory(cache)
        logp = float(value.item())
        ratio_log = logp - float(cert["old_logp"])
        boundary = float(cert["clip_boundary"])
        clip = ratio_log > boundary if int(cert["advantage_sign"]) > 0 else ratio_log < boundary
        rows.append(
            {
                "intervention": name,
                "inductor_patch": patch,
                "logp": logp,
                "signed_delta_vs_ref": logp - float(cert["logp_ref"]),
                "signed_margin": ratio_log - boundary,
                "clip_active": clip,
                "fork_vs_reference": clip != bool(cert["clip_ref"]),
                "generated_kernel_count": int(metrics.generated_kernel_count),
                "generated_cpp_vec_kernel_count": int(metrics.generated_cpp_vec_kernel_count),
                "dynamo_unique_graphs": int(counters["stats"]["unique_graphs"]),
                "cache_python_files": file_count,
                "generated_code_sha256": code_hash,
            }
        )
        del model, tokenizer
        cleanup_memory()
    baseline = rows[0]
    for row in rows:
        row["canary_code_changed"] = row["generated_code_sha256"] != baseline["generated_code_sha256"] if row is not baseline else True
        row["valid_intervention"] = row["intervention"] == "compile_baseline" or row["canary_code_changed"]
    payload = {
        "schema_version": "forkcert.fusion_probe.v1",
        "fork_id": f"clip-step{cert['metadata']['phase1_metadata']['online_state']['optimizer_step']}-{args.case_id}-t{args.token_index}",
        "contract": "Same full graph, checkpoint, tokens and backend; only one documented Inductor optimization setting changes.",
        "measurements": rows,
        "claim_scope": "Fusion-class attribution only; a successful setting does not identify a unique source operator.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
