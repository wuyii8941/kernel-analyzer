#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def keyed(payload: dict) -> dict[tuple[str, int, int], float]:
    return {
        (str(row["case_id"]), int(row["token_index"]), int(row["token_id"])): float(row["logp"])
        for row in payload["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that an in-sampler vLLM canary reaches chosen logprobs.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--canary", required=True)
    parser.add_argument("--expected-shift", type=float, default=1e-3)
    parser.add_argument("--minimum-shared-tokens", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_payload = json.loads(Path(args.baseline).read_text())
    canary_payload = json.loads(Path(args.canary).read_text())
    baseline = keyed(baseline_payload)
    canary = keyed(canary_payload)
    shared = sorted(set(baseline) & set(canary))
    deltas = [canary[key] - baseline[key] for key in shared]
    injected = float(canary_payload["metadata"].get("canary_even_token_logit_shift", 0.0))
    controlled_fields = [
        "model",
        "dtype",
        "seed",
        "engine_enforce_eager",
        "engine_chunked_prefill",
        "engine_max_num_batched_tokens",
        "engine_max_num_seqs",
        "engine_gpu_memory_utilization",
        "attention_backend_env",
        "requests",
        "request_order",
    ]
    metadata_match = all(
        baseline_payload["metadata"].get(field) == canary_payload["metadata"].get(field)
        for field in controlled_fields
    )
    payload = {
        "schema_version": "forkcert.p1.vllm-canary.v1",
        "name": args.name,
        "injected_logit_shift": injected,
        "expected_logit_shift": args.expected_shift,
        "shared_tokens": len(shared),
        "nonzero_chosen_logprob_deltas": sum(delta != 0.0 for delta in deltas),
        "max_abs_chosen_logprob_delta": max((abs(delta) for delta in deltas), default=0.0),
        "minimum_shared_tokens": args.minimum_shared_tokens,
        "controlled_metadata_fields": controlled_fields,
        "controlled_metadata_match": metadata_match,
        "passed": (
            len(shared) >= args.minimum_shared_tokens
            and metadata_match
            and injected == args.expected_shift
            and any(delta != 0.0 for delta in deltas)
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
