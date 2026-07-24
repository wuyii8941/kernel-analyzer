#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the audited natural clipping-fork baseline.")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--checkpoint", default="data/phase6_policy_step5_pre")
    parser.add_argument("--out", default="results/baseline_manifest.json")
    args = parser.parse_args()

    certificate_path = Path(args.certificates)
    forks = []
    with certificate_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not row.get("actual_fork"):
                continue
            state = row["metadata"]["phase1_metadata"]["online_state"]
            fork_id = f"clip-step{state['optimizer_step']}-{row['case_id']}-t{row['token_index']}"
            step = int(state["optimizer_step"])
            replay_paths = {
                5: ("configs/hf_compile_sdpa_math_step5.yaml", "data/phase6_step5_replay_samples.jsonl"),
                11: ("configs/hf_compile_sdpa_math_step11.yaml", "data/phase8_step11_samples.jsonl"),
                14: ("configs/hf_compile_sdpa_math_step14.yaml", "data/phase9_step14_replay_samples.jsonl"),
            }
            replayable = step in replay_paths
            replay_config, replay_samples = replay_paths.get(step, (None, None))
            forks.append(
                {
                    "fork_id": fork_id,
                    "certificate_line": line_number,
                    "case_id": row["case_id"],
                    "token_index": row["token_index"],
                    "token_id": row["token_id"],
                    "token_text": row.get("token_text"),
                    "optimizer_step": state["optimizer_step"],
                    "rollout_batch": state["rollout_batch"],
                    "old_logp": row["old_logp"],
                    "logp_ref": row["logp_ref"],
                    "logp_alt": row["logp_alt"],
                    "signed_delta_alt_minus_ref": row["logp_alt"] - row["logp_ref"],
                    "clip_boundary": row["clip_boundary"],
                    "signed_margin_ref_minus_boundary": row["logp_ref"] - row["old_logp"] - row["clip_boundary"],
                    "advantage_sign": row["advantage_sign"],
                    "clip_ref": row["clip_ref"],
                    "clip_alt": row["clip_alt"],
                    "replayable_from_frozen_state": replayable,
                    "replay_limitation": None if replayable else "No pre-minibatch checkpoint was frozen for this optimizer step.",
                    "replay_command": (
                        "CUDA_VISIBLE_DEVICES=0 /data1/tzh/conda-envs/forkcert/bin/python "
                        "scripts/phase6_grad_contrib.py --certificates results/phase4_certificates.jsonl "
                        f"--samples {replay_samples} --config {replay_config} "
                        f"--case-id {row['case_id']} --token-index {row['token_index']} "
                        f"--out-jsonl results/replay/{fork_id}.jsonl --report reports/replay/{fork_id}.md"
                    ) if replayable else None,
                }
            )

    checkpoint = Path(args.checkpoint)
    files = {}
    for name in ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "forkcert_snapshot.json"]:
        path = checkpoint / name
        if path.exists():
            files[name] = {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}
    try:
        import torch
        import transformers
        versions = {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}
    except Exception as exc:
        versions = {"error": repr(exc)}
    try:
        git = subprocess.run(
            ["git", "-C", str(Path.cwd()), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        git_commit = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        git_commit = None

    replay_checkpoints = {}
    for step, directory in [
        (5, Path(args.checkpoint)),
        (11, Path("data/phase8_policy_step11_pre")),
        (14, Path("data/phase9_policy_step14_pre")),
    ]:
        if not directory.exists():
            continue
        replay_checkpoints[str(step)] = {
            name: {"path": str((directory / name).resolve()), "size": (directory / name).stat().st_size, "sha256": sha256(directory / name)}
            for name in ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "forkcert_snapshot.json"]
            if (directory / name).exists()
        }
    payload = {
        "schema_version": "forkcert.baseline.v1",
        "source_certificates": {"path": str(certificate_path.resolve()), "sha256": sha256(certificate_path)},
        "canonical_fork_count": len(forks),
        "forks": forks,
        "frozen_step5_checkpoint": files,
        "replay_checkpoints": replay_checkpoints,
        "step5_samples": {"path": str(Path(args.samples).resolve()), "sha256": sha256(Path(args.samples))},
        "environment": {"platform": platform.platform(), "versions": versions, "gpu": "Tesla T4", "seed": 0},
        "git_commit": git_commit,
        "git_note": "Workspace is not currently recognized as a Git worktree; commit may be null.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "forks": len(forks), "replayable": sum(f["replayable_from_frozen_state"] for f in forks)}, indent=2))


if __name__ == "__main__":
    main()
