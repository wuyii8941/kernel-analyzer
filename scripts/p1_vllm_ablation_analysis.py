#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.detector import detect_clipping_fork
from forkcert.io import read_jsonl, write_jsonl


def row_key(row: dict) -> tuple[str, int, int]:
    return str(row["case_id"]), int(row["token_index"]), int(row["token_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one canary-gated vLLM component intervention.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline-certificates", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--canary-audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline_certificates)
    baseline = {row_key(row): row for row in baseline_rows}
    ablation_payload = json.loads(Path(args.ablation).read_text())
    ablation = {row_key(row): row for row in ablation_payload["rows"]}
    canary = json.loads(Path(args.canary_audit).read_text())
    if not canary.get("passed"):
        raise ValueError(f"canary did not pass for {args.name}")
    if not set(baseline).issubset(ablation):
        missing = len(set(baseline) - set(ablation))
        extra = len(set(ablation) - set(baseline))
        raise ValueError(f"ablation key coverage mismatch: missing={missing}, extra={extra}")

    output_rows = []
    for key in sorted(baseline):
        before = baseline[key]
        after = detect_clipping_fork(
            case_id=key[0],
            token_index=key[1],
            token_id=key[2],
            token_text=before.get("token_text"),
            logp_ref=float(before["logp_ref"]),
            logp_alt=float(ablation[key]["logp"]),
            old_logp=float(before["old_logp"]),
            advantage_sign_value=int(before["advantage_sign"]),
            eps=float(before["eps"]),
            path_ref=str(before["path_ref"]),
            path_alt=f"vllm-{args.name}",
        ).to_json_dict()
        output_rows.append(
            {
                **after,
                "baseline_actual_fork": bool(before["actual_fork"]),
                "baseline_clip_alt": bool(before["clip_alt"]),
                "baseline_logp_alt": float(before["logp_alt"]),
                "ablation_signed_logp_change": float(after["logp_alt"]) - float(before["logp_alt"]),
                "baseline_fork_repaired": bool(before["actual_fork"]) and not bool(after["actual_fork"]),
                "new_fork_introduced": not bool(before["actual_fork"]) and bool(after["actual_fork"]),
            }
        )

    baseline_forks = sum(bool(row["baseline_actual_fork"]) for row in output_rows)
    repaired = sum(bool(row["baseline_fork_repaired"]) for row in output_rows)
    retained = sum(bool(row["baseline_actual_fork"]) and bool(row["actual_fork"]) for row in output_rows)
    introduced = sum(bool(row["new_fork_introduced"]) for row in output_rows)
    changes = [abs(float(row["ablation_signed_logp_change"])) for row in output_rows]
    summary = {
        "schema_version": "forkcert.p1.vllm-ablation.v1",
        "name": args.name,
        "tokens": len(output_rows),
        "baseline_forks": baseline_forks,
        "ablation_forks": sum(bool(row["actual_fork"]) for row in output_rows),
        "baseline_forks_repaired": repaired,
        "baseline_forks_retained": retained,
        "new_forks_introduced": introduced,
        "repair_fraction": repaired / baseline_forks if baseline_forks else None,
        "changed_logprob_tokens": sum(value != 0.0 for value in changes),
        "max_abs_logprob_change": max(changes, default=0.0),
        "canary_audit": canary,
        "ablation_metadata": ablation_payload["metadata"],
    }
    write_jsonl(args.out, output_rows)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
