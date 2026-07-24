#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "token_id",
    "old_logp",
    "new_logp",
    "advantage",
    "advantage_sign",
    "policy_iteration",
    "rollout_batch",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected(path: Path, step: int, rollout: int) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("optimizer_step", -1)) != step or int(row.get("rollout_batch", -1)) != rollout:
                continue
            if row.get("state") != "pre_minibatch":
                continue
            rows[(str(row["case_id"]), int(row["token_index"]))] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deterministic reconstruction of a frozen online state.")
    parser.add_argument("--original", default="data/phase4_online_full_dump.jsonl")
    parser.add_argument("--replay", default="data/phase9_step14_replay_dump.jsonl")
    parser.add_argument("--snapshot", default="data/phase9_policy_step14_pre")
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--rollout", type=int, default=4)
    parser.add_argument("--out", default="results/step14_reconstruction_validation.json")
    parser.add_argument("--report", default="reports/step14_reconstruction_validation.md")
    args = parser.parse_args()
    original = selected(Path(args.original), args.step, args.rollout)
    replay = selected(Path(args.replay), args.step, args.rollout)
    common = sorted(original.keys() & replay.keys())
    field_mismatches = {
        field: sum(original[key][field] != replay[key][field] for key in common)
        for field in FIELDS
    }
    snapshot = Path(args.snapshot)
    snapshot_files = {
        name: {"size": path.stat().st_size, "sha256": sha256(path)}
        for name in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "forkcert_snapshot.json"]
        if (path := snapshot / name).exists()
    }
    passed = (
        len(original) == len(replay) == len(common) == 512
        and not any(field_mismatches.values())
        and "model.safetensors" in snapshot_files
        and "forkcert_snapshot.json" in snapshot_files
    )
    payload = {
        "schema_version": "forkcert.reconstruction_validation.v1",
        "optimizer_step": args.step,
        "rollout_batch": args.rollout,
        "original_rows": len(original),
        "replay_rows": len(replay),
        "common_keys": len(common),
        "missing_from_replay": len(original.keys() - replay.keys()),
        "extra_in_replay": len(replay.keys() - original.keys()),
        "field_mismatches": field_mismatches,
        "snapshot_files": snapshot_files,
        "passed": passed,
        "claim_scope": "Exact online-state reconstruction gate; fork logprob replay is validated separately by phase6_grad_contrib.py.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Step-14 Reconstruction Validation",
        "",
        "## Result",
        "",
        f"- Gate: `{'PASS' if passed else 'FAIL'}`",
        f"- Original/replay/common rows: `{len(original)}/{len(replay)}/{len(common)}`",
        f"- Missing/extra keys: `{payload['missing_from_replay']}/{payload['extra_in_replay']}`",
        "- Field mismatches: `" + json.dumps(field_mismatches, sort_keys=True) + "`",
        "",
        "The reconstruction is accepted only on exact equality; no numeric tolerance is used for rollout state fields.",
        "",
        "## Artifacts",
        "",
        f"- `{args.out}`",
        "- `results/replay/step14_forks_validated.jsonl`",
        "- `data/phase9_policy_step14_pre/forkcert_snapshot.json`",
        "",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
