#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import merge_pair_outputs


def load_metadata(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge four isolated R1 path runs after blind preregistration.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ref-a", required=True)
    parser.add_argument("--ref-b", required=True)
    parser.add_argument("--alt-a", required=True)
    parser.add_argument("--alt-b", required=True)
    parser.add_argument("--ref-a-meta", required=True)
    parser.add_argument("--ref-b-meta", required=True)
    parser.add_argument("--alt-a-meta", required=True)
    parser.add_argument("--alt-b-meta", required=True)
    parser.add_argument("--prereg-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metas = [
        load_metadata(args.ref_a_meta),
        load_metadata(args.ref_b_meta),
        load_metadata(args.alt_a_meta),
        load_metadata(args.alt_b_meta),
    ]
    pids = [int(meta["pid"]) for meta in metas]
    if len(set(pids)) != 4:
        raise ValueError("R1 self/cross runs must use four independent OS processes")
    fingerprints = [meta["model_artifact_fingerprint"]["aggregate_sha256"] for meta in metas]
    if len(set(fingerprints)) != 1:
        raise ValueError("R1 path runs do not share an identical model artifact fingerprint")
    if not all(meta["deterministic_algorithms"] for meta in metas):
        raise ValueError("deterministic algorithms were not enabled in every R1 path run")

    rows = merge_pair_outputs(
        ref_runs=[read_jsonl(args.ref_a), read_jsonl(args.ref_b)],
        alt_runs=[read_jsonl(args.alt_a), read_jsonl(args.alt_b)],
        path_ref=str(cfg["path_ref"]["name"]),
        path_alt=str(cfg["path_alt"]["name"]),
        metadata={
            "experiment": "r1_heldout",
            "state": args.name,
            "preregistration_commit": args.prereg_commit,
            "independent_processes": True,
            "process_ids": pids,
            "cuda_device_uuids": [str(meta["cuda_device_uuid"]) for meta in metas],
            "model_artifact_fingerprint": fingerprints[0],
            "fixed_response_tokens": True,
            "attention_backend": "MATH",
            "training_compute_dtype": "fp16",
            "t4_external_validity_scope": True,
        },
    )
    if max((float(row["delta_self_ref"]) for row in rows), default=0.0) != 0.0:
        raise ValueError("R1 reference independent-process self control is nonzero")
    if max((float(row["delta_self_alt"]) for row in rows), default=0.0) != 0.0:
        raise ValueError("R1 alternative independent-process self control is nonzero")
    write_jsonl(args.out, rows)
    print(
        json.dumps(
            {
                "name": args.name,
                "rows": len(rows),
                "independent_processes": True,
                "model_fingerprint_match": True,
                "token_alignment": True,
                "self_ref_max": 0.0,
                "self_alt_max": 0.0,
                "preregistration_commit": args.prereg_commit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
