#!/usr/bin/env python
"""Freeze eligibility for the Qwen3 GRPO training-control confirmation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "theory_oracle/QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md"
EVALUATOR = ROOT / "theory_oracle/evaluate_qwen3_grpo_training_control_confirmation.py"
SPECS = [
    ("A", ROOT / "configs/oracle_qwen3_grpo_confirm_a.yaml"),
    ("B", ROOT / "configs/oracle_qwen3_grpo_confirm_b.yaml"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_phase0():
    path = ROOT / "scripts/phase0_grpo_train.py"
    spec = importlib.util.spec_from_file_location("phase0_for_oracle_confirmation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from forkcert.config import load_config

    phase0 = load_phase0()
    seen_files = [
        ROOT / "data/phase0_grpo_samples.jsonl",
        ROOT / "data/phase4_online_full_samples.jsonl",
        ROOT / "data/phase6_step5_replay_samples.jsonl",
        ROOT / "data/phase8_step11_samples.jsonl",
        ROOT / "data/phase9_step14_replay_samples.jsonl",
        ROOT / "results/r1/from240_samples.jsonl",
        ROOT / "results/r1/from270_samples.jsonl",
        ROOT / "data/external_datasets/qwen3_impact_confirmation_32/data.jsonl",
    ]
    seen_hashes = {
        prompt_hash(str(row["prompt"]))
        for path in seen_files
        for row in read_jsonl(path)
        if "prompt" in row
    }
    records = []
    new_sets = []
    for name, config_path in SPECS:
        cfg = load_config(config_path)
        dataset, source = phase0.prepare_dataset(cfg)
        prompts = [str(row["prompt"]) for row in dataset]
        hashes = {prompt_hash(value) for value in prompts}
        model_path = ROOT / cfg["model"]["model_name_or_path"]
        files = []
        for filename in ("config.json", "model.safetensors", "tokenizer.json"):
            path = model_path / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append({"name": filename, "sha256": sha256_file(path), "size": path.stat().st_size})
        records.append(
            {
                "trajectory": name,
                "config": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "dataset_source": source,
                "prompt_count": len(prompts),
                "unique_prompt_hashes": len(hashes),
                "prompt_hash_set_sha256": hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
                "overlap_with_prior_banks": len(hashes & seen_hashes),
                "model_path": str(model_path.resolve()),
                "model_files": files,
                "optimizer_initialization": "new empty state; no historical replay",
                "expected_optimizer_steps": int(cfg["training"]["max_steps"]),
                "expected_scored_rollout_states": int(cfg["training"]["max_steps"]) // int(cfg["training"]["num_iterations"]),
            }
        )
        new_sets.append(hashes)
    passed = (
        CONTRACT.is_file()
        and all(record["overlap_with_prior_banks"] == 0 for record in records)
        and all(record["prompt_count"] == record["unique_prompt_hashes"] == 64 for record in records)
        and len(new_sets[0] & new_sets[1]) == 0
        and len({record["model_files"][1]["sha256"] for record in records}) == 2
        and sum(record["expected_scored_rollout_states"] for record in records) == 20
    )
    payload = {
        "schema_version": "forkcert.qwen3-grpo-training-control-confirmation-manifest.v0.1",
        "contract": str(CONTRACT.resolve()),
        "contract_sha256": sha256_file(CONTRACT),
        "executor": str((ROOT / "scripts/phase0_grpo_train.py").resolve()),
        "executor_sha256": sha256_file(ROOT / "scripts/phase0_grpo_train.py"),
        "evaluator": str(EVALUATOR.resolve()),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "seen_files": [str(path) for path in seen_files],
        "prior_unique_prompt_hashes": len(seen_hashes),
        "new_sets_overlap": len(new_sets[0] & new_sets[1]),
        "trajectories": records,
        "passed": passed,
    }
    out = ROOT / "theory_oracle/QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_MANIFEST_V0_1.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
