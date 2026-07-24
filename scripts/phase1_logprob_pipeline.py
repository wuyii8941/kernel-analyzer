#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from forkcert.config import load_config
from forkcert.env import audit_environment
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    PathConfig,
    cleanup_memory,
    configure_determinism,
    merge_pair_outputs,
    model_artifact_fingerprint,
    run_path_twice,
)
from forkcert.report import markdown_table, write_phase_report
from forkcert.stats import mean, percentile


def path_config(data: dict, key: str) -> PathConfig:
    item = data[key]
    return PathConfig(
        name=item["name"],
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False),
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
        rmsnorm_reference=item.get("rmsnorm_reference", False),
        rmsnorm_no_upcast=item.get("rmsnorm_no_upcast", False),
        rmsnorm_compile=item.get("rmsnorm_compile", False),
        materialize_bf16_outputs=item.get("materialize_bf16_outputs", False),
        materialization_dtype=item.get("materialization_dtype"),
        allow_bf16_reduced_precision_reduction=item.get("allow_bf16_reduced_precision_reduction"),
        allow_fp16_reduced_precision_reduction=item.get("allow_fp16_reduced_precision_reduction"),
    )


def summarize(rows: list[dict]) -> dict[str, float]:
    deltas = [r["logprob_delta"] for r in rows]
    self_ref = [r["delta_self_ref"] for r in rows]
    self_alt = [r["delta_self_alt"] for r in rows]
    return {
        "n_samples": len({str(row["case_id"]) for row in rows}),
        "n_tokens": len(deltas),
        "delta_mean": mean(deltas),
        "delta_p50": percentile(deltas, 50),
        "delta_p95": percentile(deltas, 95),
        "delta_p99": percentile(deltas, 99),
        "delta_max": max(deltas) if deltas else 0.0,
        "self_ref_p99": percentile(self_ref, 99),
        "self_alt_p99": percentile(self_alt, 99),
    }


def summarize_by_position(rows: list[dict], bucket_size: int = 32) -> list[dict]:
    buckets: dict[int, list[float]] = {}
    for row in rows:
        start = (int(row["token_index"]) // bucket_size) * bucket_size
        buckets.setdefault(start, []).append(float(row["logprob_delta"]))
    output = []
    for start, values in sorted(buckets.items()):
        output.append(
            {
                "token_positions": f"{start}-{start + bucket_size - 1}",
                "n": len(values),
                "mean": mean(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "max": max(values),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 token-level logprob pipeline for two HF paths.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True, help="JSONL with case_id, prompt, response.")
    parser.add_argument("--out-jsonl", default="results/phase1_logprobs.jsonl")
    parser.add_argument("--report", default="reports/phase1.md")
    parser.add_argument("--enforce-self-gate", action="store_true", help="Exit non-zero after writing outputs if either delta_self p99 gate fails.")
    parser.add_argument("--enforce-scale-gate", action="store_true", help="Require 100-500 fixed pairs and at least 50,000 response tokens.")
    parser.add_argument("--warmup-passes", type=int, default=0, help="Discard this many full ordered passes before the two measured self runs.")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    cfg = load_config(args.config)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    ref = path_config(cfg, "path_ref")
    alt = path_config(cfg, "path_alt")
    samples = read_jsonl(args.samples)
    env = audit_environment().to_json_dict()
    ref_fingerprint = model_artifact_fingerprint(ref.model_name_or_path)
    alt_fingerprint = model_artifact_fingerprint(alt.model_name_or_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ref_runs = run_path_twice(
            ref, samples, seed=int(cfg.get("seed", 0)), warmup_passes=args.warmup_passes
        )
        cleanup_memory()
        alt_runs = run_path_twice(
            alt, samples, seed=int(cfg.get("seed", 0)), warmup_passes=args.warmup_passes
        )
        cleanup_memory()
    warn_messages = sorted({str(item.message) for item in caught})
    metadata_path = Path(args.out_jsonl).with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "env": env,
                "config": cfg,
                "warnings": warn_messages,
                "model_artifact_fingerprint_ref": ref_fingerprint,
                "model_artifact_fingerprint_alt": alt_fingerprint,
                "warmup_passes": args.warmup_passes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    compact_env = {
        "torch": env.get("torch"),
        "packages": env.get("packages"),
        "deterministic_env": env.get("deterministic_env"),
    }
    compact_ref_fingerprint = {key: value for key, value in ref_fingerprint.items() if key != "files"}
    compact_alt_fingerprint = {key: value for key, value in alt_fingerprint.items() if key != "files"}
    rows = merge_pair_outputs(
        ref_runs=ref_runs,
        alt_runs=alt_runs,
        path_ref=ref.name,
        path_alt=alt.name,
        metadata={
            "env": compact_env,
            "config": cfg,
            "warnings": warn_messages,
            "metadata_sidecar": str(metadata_path),
            "model_artifact_fingerprint_ref": compact_ref_fingerprint,
            "model_artifact_fingerprint_alt": compact_alt_fingerprint,
                "execution_invariants": {
                "model_eval_called": True,
                "dropout_disabled_by_eval": True,
                "fixed_response_tokens": True,
                "default_position_ids_both_paths": True,
                    "default_causal_attention_mask_both_paths": True,
                    "discarded_full_warmup_passes": args.warmup_passes,
            },
        },
    )
    stats = summarize(rows)
    position_stats = summarize_by_position(rows)
    self_ok_ref = stats["self_ref_p99"] < 0.1 * stats["delta_p50"] if stats["delta_p50"] > 0 else False
    self_ok_alt = stats["self_alt_p99"] < 0.1 * stats["delta_p50"] if stats["delta_p50"] > 0 else False
    scale_ok = 100 <= stats["n_samples"] <= 500 and stats["n_tokens"] >= 50_000
    for row in rows:
        row["metadata"]["phase1_gates"] = {
            "delta_self_ref_gate": self_ok_ref,
            "delta_self_alt_gate": self_ok_alt,
            "sample_and_token_scale_gate": scale_ok,
        }
    write_jsonl(args.out_jsonl, rows)
    write_phase_report(
        args.report,
        title="Phase 1 Logprob Pipeline",
        confound_checklist={
            "fixed_response_tokens": True,
            "token_alignment_checked": True,
            "same_weights_config_expected": cfg.get("same_weights_config_expected", True),
            "model_weight_fingerprint_match": (
                ref_fingerprint.get("verified_local_files") is True
                and alt_fingerprint.get("verified_local_files") is True
                and ref_fingerprint.get("aggregate_sha256") == alt_fingerprint.get("aggregate_sha256")
            ),
            "deterministic_env_recorded": True,
            "warn_only_messages_recorded": True,
            "delta_self_ref_gate": self_ok_ref,
            "delta_self_alt_gate": self_ok_alt,
            "sample_and_token_scale_gate": scale_ok,
        },
        delta_self_summary=(
            f"ref p99={stats['self_ref_p99']:.6g}, alt p99={stats['self_alt_p99']:.6g}, "
            f"cross p50={stats['delta_p50']:.6g}."
        ),
        summary="Phase 1 produced token-level logprob deltas and self-consistency controls.",
        sections={
            "Delta Distribution": markdown_table([stats], list(stats.keys())),
            "Delta By Token Position": markdown_table(
                position_stats,
                list(position_stats[0].keys()) if position_stats else [],
            ),
            "Warn Only Messages": "\n".join(f"- {message}" for message in warn_messages) if warn_messages else "_None captured._",
        },
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    if args.enforce_self_gate and not (self_ok_ref and self_ok_alt):
        print(
            "Phase 1 self-consistency gate failed: require p99(delta_self_ref/alt) < 0.1 * p50(delta_cross).",
            file=sys.stderr,
        )
        raise SystemExit(21)
    if args.enforce_scale_gate and not scale_ok:
        print(
            f"Phase 1 scale gate failed: samples={stats['n_samples']}, tokens={stats['n_tokens']}; "
            "require 100-500 samples and >=50,000 response tokens.",
            file=sys.stderr,
        )
        raise SystemExit(21)


if __name__ == "__main__":
    main()
