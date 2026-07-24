#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from forkcert.io import write_jsonl


def keyed(payload: dict) -> dict[tuple[str, int, int], dict]:
    return {
        (str(row["case_id"]), int(row["token_index"]), int(row["token_id"])): row
        for row in payload["rows"]
    }


def cluster_bootstrap_rate(rows: list[dict], field: str, *, draws: int = 10_000, seed: int = 0) -> list[float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    case_ids = sorted(grouped)
    if not case_ids:
        return [0.0, 0.0]
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        selected = [row for _ in case_ids for row in grouped[rng.choice(case_ids)]]
        rates.append(sum(bool(row[field]) for row in selected) / len(selected))
    rates.sort()
    return [rates[int(0.025 * draws)], rates[int(0.975 * draws) - 1]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge four isolated R1 sampling runs.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ref-a", required=True)
    parser.add_argument("--ref-b", required=True)
    parser.add_argument("--alt-a", required=True)
    parser.add_argument("--alt-b", required=True)
    parser.add_argument("--prereg-commit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    payloads = [json.loads(Path(path).read_text()) for path in [args.ref_a, args.ref_b, args.alt_a, args.alt_b]]
    metas = [payload["metadata"] for payload in payloads]
    if len({int(meta["pid"]) for meta in metas}) != 4:
        raise ValueError("R1 sampling self/cross runs must use four independent processes")
    fingerprints = [meta["model_artifact_fingerprint"]["aggregate_sha256"] for meta in metas]
    if len(set(fingerprints)) != 1:
        raise ValueError("R1 sampling runs do not share one checkpoint fingerprint")
    ref_a, ref_b, alt_a, alt_b = [keyed(payload) for payload in payloads]
    if not (set(ref_a) == set(ref_b) == set(alt_a) == set(alt_b)):
        raise ValueError("R1 sampling key coverage mismatch")

    rows = []
    self_fields = ["top_k_hash", "top_p_hash", "top_k_sampled_ids", "top_p_sampled_ids", "common_uniforms"]
    for key in sorted(ref_a):
        r, r2, a, a2 = ref_a[key], ref_b[key], alt_a[key], alt_b[key]
        ref_self = any(r[field] != r2[field] for field in self_fields)
        alt_self = any(a[field] != a2[field] for field in self_fields)
        if r["common_uniforms"] != a["common_uniforms"]:
            raise ValueError(f"common-random-number mismatch for {key}")
        top_k_draws = sum(x != y for x, y in zip(r["top_k_sampled_ids"], a["top_k_sampled_ids"], strict=True))
        top_p_draws = sum(x != y for x, y in zip(r["top_p_sampled_ids"], a["top_p_sampled_ids"], strict=True))
        rows.append(
            {
                "case_id": key[0],
                "token_index": key[1],
                "token_id": key[2],
                "path_ref": r["path"],
                "path_alt": a["path"],
                "top_k_candidate_set_fork": r["top_k_hash"] != a["top_k_hash"],
                "top_p_candidate_set_fork": r["top_p_hash"] != a["top_p_hash"],
                "top_k_sampling_fork_draws": top_k_draws,
                "top_p_sampling_fork_draws": top_p_draws,
                "top_k_actual_sampling_fork": top_k_draws > 0,
                "top_p_actual_sampling_fork": top_p_draws > 0,
                "top_k_first_draw_sampling_fork": r["top_k_sampled_ids"][0] != a["top_k_sampled_ids"][0],
                "top_p_first_draw_sampling_fork": r["top_p_sampled_ids"][0] != a["top_p_sampled_ids"][0],
                "top_k_first_draw_ref": r["top_k_sampled_ids"][0],
                "top_k_first_draw_alt": a["top_k_sampled_ids"][0],
                "top_p_first_draw_ref": r["top_p_sampled_ids"][0],
                "top_p_first_draw_alt": a["top_p_sampled_ids"][0],
                "draws": len(r["common_uniforms"]),
                "ref_self_failure": ref_self,
                "alt_self_failure": alt_self,
                "preregistration_commit": args.prereg_commit,
                "region": "unknown",
            }
        )
    if any(row["ref_self_failure"] or row["alt_self_failure"] for row in rows):
        raise ValueError("R1 independent-process sampling self gate failed")
    write_jsonl(args.out, rows)
    summary = {
        "schema_version": "forkcert.r1.sampling-summary.v1",
        "name": args.name,
        "tokens": len(rows),
        "samples": len({row["case_id"] for row in rows}),
        "draws_per_token": rows[0]["draws"] if rows else 0,
        "self_failures": 0,
        "model_fingerprint_match": True,
        "independent_processes": True,
        "common_random_numbers": True,
        "preregistration_commit": args.prereg_commit,
        "top_k_candidate_set_forks": sum(row["top_k_candidate_set_fork"] for row in rows),
        "top_p_candidate_set_forks": sum(row["top_p_candidate_set_fork"] for row in rows),
        "top_k_sampling_fork_states": sum(row["top_k_actual_sampling_fork"] for row in rows),
        "top_p_sampling_fork_states": sum(row["top_p_actual_sampling_fork"] for row in rows),
        "top_k_first_draw_sampling_forks": sum(row["top_k_first_draw_sampling_fork"] for row in rows),
        "top_p_first_draw_sampling_forks": sum(row["top_p_first_draw_sampling_fork"] for row in rows),
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
        summary[f"{prefix}_rate"] = count / len(rows) if rows else 0.0
        summary[f"{prefix}_cluster_ci95"] = cluster_bootstrap_rate(rows, field)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
