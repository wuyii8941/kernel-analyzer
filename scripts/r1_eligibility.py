#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase0_module():
    path = Path(__file__).with_name("phase0_grpo_train.py")
    spec = importlib.util.spec_from_file_location("phase0_grpo_train_for_r1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prompt_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze R1 held-out checkpoint and prompt eligibility evidence.")
    parser.add_argument("--out", default="results/r1/eligibility_manifest.json")
    args = parser.parse_args()
    phase0 = load_phase0_module()
    seen_files = [
        "data/phase0_grpo_samples.jsonl",
        "data/phase4_online_full_samples.jsonl",
        "data/phase6_step5_replay_samples.jsonl",
        "data/phase8_step11_samples.jsonl",
        "data/phase9_step14_replay_samples.jsonl",
    ]
    seen_hashes = {
        prompt_hash(str(row["prompt"]))
        for path in seen_files
        if Path(path).exists()
        for row in read_jsonl(path)
    }
    specifications = [
        ("from240", "configs/r1_heldout_from240.yaml", "data/phase0_policy_final/checkpoint-240", 242),
        ("from270", "configs/r1_heldout_from270.yaml", "data/phase0_policy_final/checkpoint-270", 272),
    ]
    states = []
    heldout_sets = []
    for name, config_path, checkpoint_path, target_step in specifications:
        cfg = load_config(config_path)
        dataset, source = phase0.prepare_dataset(cfg)
        hashes = [prompt_hash(str(row["prompt"])) for row in dataset]
        checkpoint = Path(checkpoint_path)
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
        state = {
            "name": name,
            "source_checkpoint": str(checkpoint.resolve()),
            "source_optimizer_step": int(trainer_state["global_step"]),
            "source_model_sha256": sha256_file(checkpoint / "model.safetensors"),
            "source_trainer_state_sha256": sha256_file(checkpoint / "trainer_state.json"),
            "target_pre_minibatch_step": target_step,
            "target_policy_iteration": target_step % int(cfg["training"]["num_iterations"]),
            "config": str(Path(config_path).resolve()),
            "config_sha256": sha256_file(Path(config_path)),
            "dataset_source": source,
            "prompt_count": len(hashes),
            "prompt_hash_set_sha256": hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
            "overlap_with_existing_prompt_hashes": len(set(hashes) & seen_hashes),
            "unique_prompt_hashes": len(set(hashes)),
        }
        states.append(state)
        heldout_sets.append(set(hashes))
    payload = {
        "schema_version": "forkcert.r1.eligibility.v1",
        "existing_prompt_hashes": len(seen_hashes),
        "existing_sample_files": seen_files,
        "heldout_sets_overlap_each_other": len(heldout_sets[0] & heldout_sets[1]),
        "source_checkpoint_direct_experiment_references": {
            "checkpoint-240": [],
            "checkpoint-270": [],
            "audit_command": "rg -n 'checkpoint-240|checkpoint-270' configs scripts results reports logs",
        },
        "states": states,
        "passed": all(state["overlap_with_existing_prompt_hashes"] == 0 for state in states)
        and len(heldout_sets[0] & heldout_sets[1]) == 0
        and all(state["target_policy_iteration"] == 2 for state in states),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
