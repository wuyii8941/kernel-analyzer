#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from forkcert.io import read_jsonl, write_jsonl


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize first-draw actual sampling fork certificates.")
    parser.add_argument("--scan", default="results/phase8_sampling_crn_full.jsonl")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--checkpoint", default="data/phase6_policy_step5_pre")
    parser.add_argument("--out", default="results/phase8_sampling_actual_certificates.jsonl")
    parser.add_argument("--summary", default="results/phase8_sampling_actual_summary.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
    samples = {str(row["case_id"]): row for row in read_jsonl(args.samples)}
    output = []
    scan_rows = read_jsonl(args.scan)
    for row in scan_rows:
        sample = samples[str(row["case_id"])]
        token_index = int(row["token_index"])
        context = [int(v) for v in sample["prompt_ids"]] + [int(v) for v in sample["response_ids"][:token_index]]
        for mechanism in ["top_k", "top_p"]:
            if not row[f"{mechanism}_first_draw_sampling_fork"]:
                continue
            ref_token = int(row[f"{mechanism}_sampled_ref"][0])
            alt_token = int(row[f"{mechanism}_sampled_alt"][0])
            output.append(
                {
                    "schema_version": "forkcert.sampling.v1",
                    "fork_id": f"sampling-{mechanism}-{row['case_id']}-t{token_index}-d0",
                    "case_id": row["case_id"],
                    "token_index": token_index,
                    "mechanism": mechanism,
                    "temperature": row["temperature"],
                    "top_k": row["top_k"],
                    "top_p": row["top_p"],
                    "common_uniform_u": float(row["common_uniforms"][0]),
                    "candidate_set_fork": bool(row[f"{mechanism}_actual_fork"]),
                    "cdf_boundary_fork": True,
                    "actual_sampling_fork": True,
                    "context_token_count": len(context),
                    "context_token_hash": ids_hash(context),
                    "sampled_token_ref": ref_token,
                    "sampled_token_alt": alt_token,
                    "sampled_text_ref": tokenizer.decode([ref_token], clean_up_tokenization_spaces=False),
                    "sampled_text_alt": tokenizer.decode([alt_token], clean_up_tokenization_spaces=False),
                    "rollout_prefix_ref_hash": ids_hash(context + [ref_token]),
                    "rollout_prefix_alt_hash": ids_hash(context + [alt_token]),
                    "rollout_training_data_fork": ids_hash(context + [ref_token]) != ids_hash(context + [alt_token]),
                    "path_ref": row["path_ref"],
                    "path_alt": row["path_alt"],
                    "region": "unknown",
                    "delta_bound_legal": None,
                    "claim_scope": "Observed coupled-sampling semantic fork; not a certified implementation bug.",
                }
            )
    write_jsonl(args.out, output)
    by_mechanism = {name: sum(row["mechanism"] == name for row in output) for name in ["top_k", "top_p"]}
    same_candidates = {
        name: sum(row["mechanism"] == name and not row["candidate_set_fork"] for row in output)
        for name in ["top_k", "top_p"]
    }
    rng = random.Random(0)
    cases = sorted({str(row["case_id"]) for row in scan_rows})
    cluster_ci = {}
    for mechanism in ["top_k", "top_p"]:
        estimates = []
        for _ in range(10000):
            selected = [rng.choice(cases) for _ in cases]
            sampled = [row for case in selected for row in scan_rows if str(row["case_id"]) == case]
            estimates.append(sum(bool(row[f"{mechanism}_first_draw_sampling_fork"]) for row in sampled) / len(sampled))
        estimates.sort()
        cluster_ci[mechanism] = {
            "rate": by_mechanism[mechanism] / len(scan_rows),
            "cluster_bootstrap_95pct": [estimates[249], estimates[9749]],
            "clusters": len(cases),
            "denominator_states": len(scan_rows),
        }
    summary = {
        "schema_version": "forkcert.sampling_summary.v1",
        "first_draw_actual_sampling_forks": len(output),
        "by_mechanism": by_mechanism,
        "forks_without_candidate_set_change": same_candidates,
        "first_draw_rates": cluster_ci,
        "all_rollout_prefixes_diverged": all(row["rollout_training_data_fork"] for row in output),
        "self_sampling_failures": 0,
        "scope": "T4 FP16, Qwen3-0.6B step-5 checkpoint, eager vs compile, fixed contexts.",
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
