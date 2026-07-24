#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forkcert.io import read_jsonl, write_jsonl


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict) -> tuple[str, int]:
    return str(row["case_id"]), int(row["token_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Align HF-vLLM rows to a frozen canonical training state.")
    parser.add_argument("--logprobs", required=True)
    parser.add_argument("--state-certificates", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument("--policy-iteration", type=int, required=True)
    parser.add_argument("--rollout-batch", type=int, required=True)
    parser.add_argument("--state", default="pre_minibatch")
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--out-logprobs", required=True)
    parser.add_argument("--out-state", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    source = read_jsonl(args.logprobs)
    state_by_key = {key(row): row for row in read_jsonl(args.state_certificates)}
    aligned_logprobs = []
    aligned_state = []
    missing = 0
    token_mismatch = 0
    wrong_frozen_state = 0
    for row in source:
        state = state_by_key.get(key(row))
        if state is None:
            missing += 1
            continue
        if int(row["token_id"]) != int(state["token_id"]):
            token_mismatch += 1
            continue
        online_state = (((state.get("metadata") or {}).get("phase1_metadata") or {}).get("online_state") or {})
        expected_state = (
            int(online_state.get("optimizer_step", -1)) == args.optimizer_step
            and int(online_state.get("policy_iteration", -1)) == args.policy_iteration
            and int(online_state.get("rollout_batch", -1)) == args.rollout_batch
            and str(online_state.get("state")) == args.state
        )
        if not expected_state:
            wrong_frozen_state += 1
            continue
        aligned_logprobs.append(row)
        aligned_state.append(
            {
                "case_id": str(state["case_id"]),
                "token_index": int(state["token_index"]),
                "token_id": int(state["token_id"]),
                "token_text": state.get("token_text"),
                "old_logp": float(state["old_logp"]),
                "advantage_sign": int(state["advantage_sign"]),
                "advantage": (state.get("metadata") or {}).get("rollout_advantage"),
                "optimizer_step": int(online_state["optimizer_step"]),
                "policy_iteration": int(online_state["policy_iteration"]),
                "rollout_batch": int(online_state["rollout_batch"]),
                "state": str(online_state["state"]),
            }
        )
    write_jsonl(args.out_logprobs, aligned_logprobs)
    write_jsonl(args.out_state, aligned_state)
    checkpoint = Path(args.checkpoint)
    tokenizer = Path(args.tokenizer)
    samples = Path(args.samples)
    payload = {
        "schema_version": "forkcert.p1.vllm.state-alignment.v1",
        "source_rows": len(source),
        "aligned_rows": len(aligned_logprobs),
        "aligned_cases": len({row["case_id"] for row in aligned_logprobs}),
        "missing_state_rows": missing,
        "token_mismatches": token_mismatch,
        "wrong_frozen_state_rows": wrong_frozen_state,
        "expected_rows": args.expected_rows,
        "state": {
            "optimizer_step": args.optimizer_step,
            "policy_iteration": args.policy_iteration,
            "rollout_batch": args.rollout_batch,
            "label": args.state,
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "tokenizer": str(tokenizer.resolve()),
        "tokenizer_sha256": file_sha256(tokenizer),
        "samples": str(samples.resolve()),
        "samples_sha256": file_sha256(samples),
        "passed": len(aligned_logprobs) == args.expected_rows and token_mismatch == 0,
    }
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
