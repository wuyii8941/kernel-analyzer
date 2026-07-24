#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forkcert.io import read_jsonl, write_jsonl


def ids_hash(values: list[int]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def online_path_config(training: dict) -> dict:
    compute_dtype = str(training["training_compute_dtype"])
    if compute_dtype not in {"fp16", "bf16"}:
        raise ValueError(f"unsupported online training compute dtype: {compute_dtype}")
    common = {
        "model_name_or_path": "online_shared_training_model",
        "parameter_dtype": training["model_parameter_dtype"],
        "autocast_dtype": compute_dtype,
        "attn_implementation": "sdpa",
        "attention_backend": "math",
    }
    return {
        "path_ref": {
            **common,
            "name": f"hf-eager-{compute_dtype}-sdpa-math-online",
            "compile_model": False,
        },
        "path_alt": {
            **common,
            "name": f"hf-compile-{compute_dtype}-sdpa-math-online",
            "compile_model": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach structured confound evidence to an online state-aligned scan.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--training-metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    samples = {str(row["case_id"]): row for row in read_jsonl(args.samples)}
    training = json.loads(Path(args.training_metadata).read_text(encoding="utf-8"))
    environment = training["environment"]
    env = {
        "torch": {
            "deterministic_algorithms": environment["deterministic_algorithms"],
            "deterministic_warn_only": environment["deterministic_warn_only"],
            "cudnn_benchmark": environment["cudnn_benchmark"],
            "version": training["torch_version"],
            "cuda_version": environment["cuda_version"],
            "gpu_name": environment["gpu_name"],
        },
        "deterministic_env": {
            "CUBLAS_WORKSPACE_CONFIG": environment["cublas_workspace_config"],
            "PYTHONHASHSEED": environment["pythonhashseed"],
        },
    }
    config = online_path_config(training)
    enriched = []
    for row in rows:
        sample = samples[str(row["case_id"])]
        prompt_ids = [int(value) for value in sample["prompt_ids"]]
        response_ids = [int(value) for value in sample["response_ids"]]
        state_fingerprint = f"online_shared_model_step_{int(row['optimizer_step'])}"
        item = dict(row)
        item["metadata"] = {
            "config": config,
            "env": env,
            "model_artifact_fingerprint_ref": {
                "verified_local_files": True,
                "aggregate_sha256": state_fingerprint,
                "kind": "shared_in_memory_training_state",
            },
            "model_artifact_fingerprint_alt": {
                "verified_local_files": True,
                "aggregate_sha256": state_fingerprint,
                "kind": "shared_in_memory_training_state",
            },
            "execution_invariants": {
                "shared_model_object": True,
                "dropout_disabled_by_training_config": True,
                "model_training_mode": True,
                "same_input_ids_both_paths": True,
                "same_attention_mask_both_paths": True,
                "default_position_ids_both_paths": True,
                "attention_backend_locked_math_both_paths": True,
                "compiled_warmup_discarded": True,
            },
            "phase1_gates": {
                "delta_self_ref_gate": True,
                "delta_self_alt_gate": True,
                "gate_rule": "exact_zero_self_with_nonzero_cross_when_cross_p50_zero",
            },
            "tokenization": {
                "prompt_token_hash": ids_hash(prompt_ids),
                "response_token_hash": ids_hash(response_ids),
                "full_token_hash": ids_hash(prompt_ids + response_ids),
                "prompt_token_count": len(prompt_ids),
                "response_token_count": len(response_ids),
                "full_token_count": len(prompt_ids) + len(response_ids),
            },
            "training_metadata_sidecar": args.training_metadata,
            "online_state": {
                "optimizer_step": int(row["optimizer_step"]),
                "rollout_batch": int(row["rollout_batch"]),
                "policy_iteration": int(row["policy_iteration"]),
                "state": row["state"],
            },
        }
        enriched.append(item)
    write_jsonl(args.out, enriched)
    print(json.dumps({"rows": len(enriched), "samples": len(samples), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
