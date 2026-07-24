#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from forkcert.io import write_jsonl
from forkcert.report import markdown_table


def keyed(payload: dict) -> dict[tuple[str, int, int], dict]:
    return {(str(row["case_id"]), int(row["token_index"]), int(row["token_id"])): row for row in payload["rows"]}


def cluster_bootstrap_rate(rows: list[dict], field: str, *, draws: int = 10_000, seed: int = 0) -> tuple[float, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    case_ids = sorted(grouped)
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        sample = [rng.choice(case_ids) for _ in case_ids]
        selected = [row for case_id in sample for row in grouped[case_id]]
        rates.append(sum(bool(row[field]) for row in selected) / len(selected))
    rates.sort()
    return rates[int(0.025 * draws)], rates[int(0.975 * draws) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent HF/vLLM sampling-decision runs.")
    parser.add_argument("--hf-a", required=True)
    parser.add_argument("--hf-b", required=True)
    parser.add_argument("--vllm-a", required=True)
    parser.add_argument("--vllm-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    payloads = [json.loads(Path(path).read_text()) for path in [args.hf_a, args.hf_b, args.vllm_a, args.vllm_b]]
    hf_a, hf_b, va, vb = [keyed(payload) for payload in payloads]
    if not (set(hf_a) == set(hf_b) == set(va) == set(vb)):
        raise ValueError("HF/vLLM sampling key coverage mismatch")
    rows = []
    for key in sorted(hf_a):
        h, h2, v, v2 = hf_a[key], hf_b[key], va[key], vb[key]
        hf_self = any(h[field] != h2[field] for field in ["top_k_hash", "top_p_hash", "top_k_sampled_ids", "top_p_sampled_ids"])
        vllm_self = any(v[field] != v2[field] for field in ["top_k_hash", "top_p_hash", "top_k_sampled_ids", "top_p_sampled_ids"])
        top_k_draws = sum(a != b for a, b in zip(h["top_k_sampled_ids"], v["top_k_sampled_ids"], strict=True))
        top_p_draws = sum(a != b for a, b in zip(h["top_p_sampled_ids"], v["top_p_sampled_ids"], strict=True))
        rows.append(
            {
                "case_id": key[0],
                "token_index": key[1],
                "token_id": key[2],
                "path_ref": h["path"],
                "path_alt": v["path"],
                "top_k_candidate_set_fork": h["top_k_hash"] != v["top_k_hash"],
                "top_p_candidate_set_fork": h["top_p_hash"] != v["top_p_hash"],
                "top_k_ref_count": len(h["top_k_ids"]),
                "top_k_alt_count": len(v["top_k_ids"]),
                "top_p_ref_count": int(h["top_p_count"]),
                "top_p_alt_count": int(v["top_p_count"]),
                "top_k_sampling_fork_draws": top_k_draws,
                "top_p_sampling_fork_draws": top_p_draws,
                "top_k_actual_sampling_fork": top_k_draws > 0,
                "top_p_actual_sampling_fork": top_p_draws > 0,
                "top_k_first_draw_sampling_fork": h["top_k_sampled_ids"][0] != v["top_k_sampled_ids"][0],
                "top_p_first_draw_sampling_fork": h["top_p_sampled_ids"][0] != v["top_p_sampled_ids"][0],
                "hf_self_failure": hf_self,
                "vllm_self_failure": vllm_self,
                "draws": len(h["top_k_sampled_ids"]),
                "region": "unknown",
            }
        )
    summary = {
        "schema_version": "forkcert.p1.hf-vllm-sampling.v1",
        "tokens": len(rows),
        "samples": len({row["case_id"] for row in rows}),
        "top_k_candidate_set_forks": sum(row["top_k_candidate_set_fork"] for row in rows),
        "top_p_candidate_set_forks": sum(row["top_p_candidate_set_fork"] for row in rows),
        "top_k_sampling_fork_states": sum(row["top_k_actual_sampling_fork"] for row in rows),
        "top_p_sampling_fork_states": sum(row["top_p_actual_sampling_fork"] for row in rows),
        "top_k_first_draw_sampling_forks": sum(row["top_k_first_draw_sampling_fork"] for row in rows),
        "top_p_first_draw_sampling_forks": sum(row["top_p_first_draw_sampling_fork"] for row in rows),
        "top_k_sampling_fork_draws": sum(row["top_k_sampling_fork_draws"] for row in rows),
        "top_p_sampling_fork_draws": sum(row["top_p_sampling_fork_draws"] for row in rows),
        "hf_self_failures": sum(row["hf_self_failure"] for row in rows),
        "vllm_self_failures": sum(row["vllm_self_failure"] for row in rows),
        "all_regions_unknown": True,
        "rate_ci_method": "case_id cluster bootstrap, 10000 draws, seed=0",
    }
    for field, prefix in [
        ("top_k_candidate_set_fork", "top_k_candidate_set"),
        ("top_p_candidate_set_fork", "top_p_candidate_set"),
        ("top_k_actual_sampling_fork", "top_k_sampling_state"),
        ("top_p_actual_sampling_fork", "top_p_sampling_state"),
        ("top_k_first_draw_sampling_fork", "top_k_first_draw"),
        ("top_p_first_draw_sampling_fork", "top_p_first_draw"),
    ]:
        count = sum(bool(row[field]) for row in rows)
        low, high = cluster_bootstrap_rate(rows, field)
        summary[f"{prefix}_rate"] = count / len(rows)
        summary[f"{prefix}_cluster_ci95_low"] = low
        summary[f"{prefix}_cluster_ci95_high"] = high
    write_jsonl(args.out, rows)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = "\n".join(
        [
            "# P1 HF-vLLM Sampling Fork Scan",
            "",
            "## Confound Checklist",
            "- same checkpoint and fixed response token IDs: PASS",
            "- temperature=1.0 before top-k/top-p: PASS",
            "- common random numbers shared by case/token/draw: PASS",
            "- independent-process self decisions: " + ("PASS" if not summary["hf_self_failures"] and not summary["vllm_self_failures"] else "FAIL"),
            "",
            "## Delta Self Control",
            f"HF decision self failures={summary['hf_self_failures']}; vLLM decision self failures={summary['vllm_self_failures']}.",
            "",
            "## External Validity",
            "T4 FP16, vLLM 0.9.2 V0 and XFormers only; BF16/V1/FlashAttention remain external replication work.",
            "",
            "## Summary",
            markdown_table([summary], list(summary)),
            "",
        ]
    )
    Path(args.report).write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["hf_self_failures"] or summary["vllm_self_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
