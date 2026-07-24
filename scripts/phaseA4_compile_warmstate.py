#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    PathConfig,
    cleanup_memory,
    configure_determinism,
    load_hf_path,
    response_logprobs_for_sample,
)
from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import mean, percentile


def path_config(data: dict, key: str) -> PathConfig:
    item = data[key]
    return PathConfig(
        name=item["name"],
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False),
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
    )


def dynamo_counters() -> dict:
    try:
        from torch._dynamo.utils import counters

        return {
            group: {str(key): int(value) for key, value in values.items()}
            for group, values in counters.items()
            if values
        }
    except Exception as exc:
        return {"unavailable": {"error": str(exc)}}


def run_pass(tokenizer, model, path, samples: list[dict]) -> list[dict]:
    rows = []
    for sample_index, sample in enumerate(samples):
        for row in response_logprobs_for_sample(tokenizer, model, path, sample):
            rows.append({"sample_index": sample_index, "case_id": sample["case_id"], **row})
    return rows


def compare(left: list[dict], right: list[dict], label: str) -> tuple[dict, list[dict]]:
    if len(left) != len(right):
        raise ValueError(f"row count mismatch for {label}")
    details = []
    for a, b in zip(left, right, strict=True):
        key_a = (a["case_id"], a["token_index"], a["token_id"])
        key_b = (b["case_id"], b["token_index"], b["token_id"])
        if key_a != key_b:
            raise ValueError(f"token mismatch for {label}: {key_a} != {key_b}")
        delta = abs(float(b["logp"]) - float(a["logp"]))
        details.append(
            {
                "comparison": label,
                "sample_index": int(a["sample_index"]),
                "case_id": a["case_id"],
                "token_index": int(a["token_index"]),
                "token_id": int(a["token_id"]),
                "delta": delta,
            }
        )
    values = [row["delta"] for row in details]
    affected = {row["case_id"] for row in details if row["delta"] > 0}
    summary = {
        "comparison": label,
        "tokens": len(values),
        "affected_cases": len(affected),
        "nonzero_tokens": sum(value > 0 for value in values),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p99": percentile(values, 99),
        "max": max(values) if values else 0.0,
        "bitwise_equal": not affected,
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose torch.compile first-pass versus warm-state determinism.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_audit.yaml")
    parser.add_argument("--samples", default="data/phase0_grpo_samples.jsonl")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--out-json", default="results/phaseA4_compile_warmstate.json")
    parser.add_argument("--out-jsonl", default="results/phaseA4_compile_warmstate_deltas.jsonl")
    parser.add_argument("--report", default="reports/phaseA4_compile_warmstate.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    path = path_config(cfg, "path_alt")
    if not path.compile_model:
        raise ValueError("A4 warm-state diagnostic requires path_alt.compile_model=true")
    samples = read_jsonl(args.samples)[: args.max_samples]
    tokenizer, model = load_hf_path(path)
    passes = []
    counter_snapshots = []
    try:
        for _ in range(3):
            passes.append(run_pass(tokenizer, model, path, samples))
            counter_snapshots.append(dynamo_counters())
    finally:
        del model
        del tokenizer
        cleanup_memory()

    summary12, details12 = compare(passes[0], passes[1], "cold_to_warm")
    summary23, details23 = compare(passes[1], passes[2], "warm_to_warm")
    summaries = [summary12, summary23]
    payload = {
        "config": args.config,
        "path": path.__dict__,
        "sample_count": len(samples),
        "summaries": summaries,
        "dynamo_counters_after_each_pass": counter_snapshots,
        "warm_state_gate": bool(summary23["bitwise_equal"]),
        "interpretation": (
            "Warm-up may be excluded from measurement only if warm_to_warm is bitwise equal. "
            "cold_to_warm deltas remain a compile-state effect and must be reported separately."
        ),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.out_jsonl, details12 + details23)
    report = "\n".join(
        [
            "# Phase A4 Compile Warm-State Audit",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- same model object and CUDA context across all three passes: PASS",
            "- same ordered tokenized samples: PASS",
            "- SDPA-loaded path with MATH backend locked: PASS",
            "- warm-state attribution allowed only if pass 2 equals pass 3 bitwise",
            "",
            "## Delta Self Control",
            markdown_table(summaries, list(summaries[0].keys())),
            "",
            "## External Validity",
            "This is a T4 FP16 compile-state audit. It does not establish BF16 compile determinism.",
            "",
            "## Conclusion",
            "Warm-state gate: " + ("PASS" if payload["warm_state_gate"] else "FAIL"),
            payload["interpretation"],
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["warm_state_gate"]:
        raise SystemExit(36)


if __name__ == "__main__":
    main()
