#!/usr/bin/env python
"""Build a rule-constrained T1a/T1b manifest for one calibration state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from theory_oracle.qwen3_grpo_natural_transition_v0_2 import json_sha256, sha256_file


SCHEMA_VERSION = "forkcert.qwen3-calibration-state-endpoint-manifest.v0.1"


def derive_seed(payload: str) -> tuple[int, str]:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF, digest.hex()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_spec(path: Path, expected_arm: str, repeat: int) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = path / "result.json"
    result = load_json(result_path)
    if not result.get("valid") or result.get("arm") != expected_arm or result.get("repeat") != repeat:
        raise ValueError(f"invalid arm/repeat result: {result_path}")
    update = result.get("vector_artifacts", {}).get("parameter_updates")
    if not update or sha256_file(Path(update["path"])) != update["sha256"]:
        raise ValueError(f"invalid parameter update artifact: {result_path}")
    return (
        {
            "transition_repeat": repeat,
            "transition_result": str(result_path.resolve()),
            "transition_result_sha256": sha256_file(result_path),
            "parameter_updates": str(Path(update["path"]).resolve()),
            "parameter_updates_sha256": update["sha256"],
            "post_parameter_digest": result["post_state"]["parameter_digest"],
            "post_buffer_digest": result["post_state"]["buffer_digest"],
        },
        result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--transition-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    transition_root = Path(args.transition_root).resolve()
    metadata_path = snapshot_dir / "forkcert_transition_snapshot.json"
    metadata = load_json(metadata_path)
    target_identity = metadata["capture_target_identity"]
    arms: dict[str, dict[str, Any]] = {}
    raw_results: list[dict[str, Any]] = []
    for label, directory_prefix, expected_arm in (
        ("reference", "eager", "eager"),
        ("candidate", "compiled", "compiled"),
    ):
        specs = []
        for repeat in (1, 2):
            spec, result = arm_spec(
                transition_root / f"{directory_prefix}_{repeat}", expected_arm, repeat
            )
            specs.append(spec)
            raw_results.append(result)
        arms[label] = {"transition_repeats": specs}

    pre_fields = ("parameter_digest", "buffer_digest", "optimizer_digest", "scheduler_digest", "scaler_digest", "rng")
    if not all(
        all(result["pre_state"][field] == raw_results[0]["pre_state"][field] for field in pre_fields)
        for result in raw_results[1:]
    ):
        raise ValueError("transition arms do not share one exact pre-state")
    contracts = {
        result["anchors"].get("realization_contract_sha256") for result in raw_results
    }
    if None in contracts or len(contracts) != 1:
        raise ValueError("transition arms do not share one prospective realization contract")

    query_id = str(target_identity["query_id"])
    trajectory_id = str(target_identity["trajectory_id"])
    state_id = str(metadata["state_id"])
    seed_payload = f"qwen3-bias-oracle-t1a-v0.1/{query_id}/{trajectory_id}/{state_id}"
    seed, seed_digest = derive_seed(seed_payload)
    pre = raw_results[0]["pre_state"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "RULE_CONSTRAINED_CALIBRATION_STATE_ENDPOINT_MANIFEST",
        "state_scope": {
            **target_identity,
            "optimizer_step": int(metadata["optimizer_step"]),
            "history_selection": metadata["history_selection"],
            "population_role": "CALIBRATION_CONSTRUCTION_AND_SCALE_ONLY",
            "snapshot_dir": str(snapshot_dir),
            "snapshot_metadata_sha256": sha256_file(metadata_path),
            "pre_parameter_digest": pre["parameter_digest"],
            "pre_buffer_digest": pre["buffer_digest"],
            "realization_contract_sha256": next(iter(contracts)),
        },
        "arms": arms,
        "bank": {
            "prompt_source": "forkcert_builtin_arithmetic",
            "prompt_indices": list(range(9000, 9008)),
            "prompt_template": "Solve the problem. Show concise reasoning and end with the numeric answer.\n\nA box starts with {start} items, receives {added}, then gives away {removed}. How many remain?",
            "num_generations_per_prompt": 4,
            "generation_order": "prompt index ascending; four repeated rows per prompt; one group generated at a time",
            "prompt_encoding": "raw tokenizer text encoding with tokenizer defaults; left padded within group",
            "generation": {
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "min_new_tokens": 128,
                "max_new_tokens": 128,
                "use_cache": False,
                "autocast": "CUDA float16",
                "attention": "SDPA MATH",
            },
            "seed_payload": seed_payload,
            "seed_payload_sha256": seed_digest,
            "seed_derivation": "int.from_bytes(digest[0:4], 'big') & 0x7fffffff",
            "seed": seed,
            "reward": "forkcert numeric reward using the last parsed number",
            "advantage": "float32 (reward-group_mean)/(sample_std_Bessel+1e-4) within each group of four",
            "all_tied_rule": "retain tied groups with zero advantages; all groups tied makes this state endpoint UNINSTANTIATED, with no resampling for signal",
            "old_logprob": "common eager pre-state scorer, FP16 autocast, SDPA MATH, temperature 1.0",
            "artifact_rule": "one frozen bank shared by every transition/evaluator repeat; two fresh generations must hash exactly",
        },
        "T1a": {
            "epsilon": 0.2,
            "importance_sampling_level": "token",
            "loss_type": "grpo",
            "aggregation": "within-completion valid-token mean, then equal mean over all 32 completions",
        },
        "T1b": {
            "start_index_inclusive": 9008,
            "stop_index_exclusive": 9072,
            "prompt_template": "Solve the problem. Show concise reasoning and end with the numeric answer.\n\nA box starts with {start} items, receives {added}, then gives away {removed}. How many remain?",
            "prompt_completion_separator": "\n\n",
            "completion_template": "Compute {start} + {added} - {removed}. The answer is {result}.",
            "require_prompt_token_prefix_stability": True,
            "include_eos_in_target": False,
            "batch_size": 8,
        },
        "evaluation": {
            "post_state_evaluator": "same common eager FP16 SDPA-math scorer for every post-state",
            "evaluator_repeats_per_transition_repeat": 2,
            "nesting_rule": "average evaluator repeats e within each transition repeat r before candidate-minus-reference",
        },
        "rule_provenance": {
            "global_contract": "QWEN3_BIAS_ORACLE_C0_MANIFEST_DRAFT_V0_1.json",
            "manifest_content_digest": None,
        },
        "nonclaims": [
            "one calibration state is not population B",
            "common eager scoring is a shared ruler, not truth",
            "T shift is not long-run training harm or operator attribution",
        ],
    }
    manifest["rule_provenance"]["manifest_content_digest"] = json_sha256(
        {key: value for key, value in manifest.items() if key != "rule_provenance"}
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "state_id": state_id,
                "optimizer_step": metadata["optimizer_step"],
                "seed": seed,
                "transition_repeats_per_arm": 2,
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
