#!/usr/bin/env python
"""Freeze the corrected, prompt-disjoint Qwen3 GRPO v0.2 confirmation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "theory_oracle/QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_CONTRACT_V0_2_2026-07-17.md"
EXECUTOR = ROOT / "scripts/phase0_grpo_train.py"
EVALUATOR = ROOT / "theory_oracle/evaluate_qwen3_grpo_training_control_confirmation_v0_2.py"
SPECS = [
    ("A2", ROOT / "configs/oracle_qwen3_grpo_confirm_v0_2_a.yaml"),
    ("B2", ROOT / "configs/oracle_qwen3_grpo_confirm_v0_2_b.yaml"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_phase0():
    spec = importlib.util.spec_from_file_location("phase0_oracle_v02", EXECUTOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from forkcert.config import load_config

    phase0 = load_phase0()
    prior_files = [
        ROOT / "data/phase0_grpo_samples.jsonl",
        ROOT / "data/phase4_online_full_samples.jsonl",
        ROOT / "results/r1/from240_samples.jsonl",
        ROOT / "results/r1/from270_samples.jsonl",
        ROOT / "data/external_datasets/qwen3_impact_confirmation_32/data.jsonl",
        ROOT / "results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_1/a_samples.jsonl",
        ROOT / "results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_1/b_samples.jsonl",
    ]
    prior = {
        prompt_hash(str(item["prompt"]))
        for path in prior_files
        for item in rows(path)
        if "prompt" in item
    }
    records = []
    new_sets = []
    for name, config_path in SPECS:
        cfg = load_config(config_path)
        dataset, source = phase0.prepare_dataset(cfg)
        prompts = [str(item["prompt"]) for item in dataset]
        hashes = {prompt_hash(value) for value in prompts}
        model_path = ROOT / cfg["model"]["model_name_or_path"]
        model_hash = sha256_file(model_path / "model.safetensors")
        records.append(
            {
                "trajectory": name,
                "config": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "dataset_source": source,
                "prompt_count": len(prompts),
                "unique_prompt_hashes": len(hashes),
                "prompt_hash_set_sha256": hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
                "overlap_with_all_prior_banks": len(hashes & prior),
                "model_path": str(model_path.resolve()),
                "model_sha256": model_hash,
                "expected_rollout_states": 10,
                "expected_token_rows": 5120,
                "optimizer_initialization": "new empty state",
            }
        )
        new_sets.append(hashes)
    passed = (
        all(path.is_file() for path in (CONTRACT, EXECUTOR, EVALUATOR))
        and all(item["overlap_with_all_prior_banks"] == 0 for item in records)
        and all(item["prompt_count"] == item["unique_prompt_hashes"] == 64 for item in records)
        and len(new_sets[0] & new_sets[1]) == 0
        and len({item["model_sha256"] for item in records}) == 2
        and sum(item["expected_rollout_states"] for item in records) == 20
    )
    payload = {
        "schema_version": "forkcert.qwen3-grpo-training-control-confirmation-manifest.v0.2",
        "contract": str(CONTRACT.resolve()),
        "contract_sha256": sha256_file(CONTRACT),
        "executor": str(EXECUTOR.resolve()),
        "executor_sha256": sha256_file(EXECUTOR),
        "scoring_logic": str((ROOT / "theory_oracle/evaluate_qwen3_grpo_training_control_confirmation.py").resolve()),
        "scoring_logic_sha256": sha256_file(ROOT / "theory_oracle/evaluate_qwen3_grpo_training_control_confirmation.py"),
        "evaluator": str(EVALUATOR.resolve()),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "prior_prompt_hashes": len(prior),
        "new_sets_overlap": len(new_sets[0] & new_sets[1]),
        "dynamo_recompile_limit": 64,
        "trajectories": records,
        "passed": passed,
    }
    out = ROOT / "theory_oracle/QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_MANIFEST_V0_2.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
