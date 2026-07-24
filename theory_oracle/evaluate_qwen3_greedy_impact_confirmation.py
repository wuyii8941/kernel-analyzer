#!/usr/bin/env python
"""Apply the frozen Qwen3 strict greedy-compatibility confirmation contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--bank-manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = read_jsonl(result_dir / "states.jsonl")
    run_summary = json.loads(
        (result_dir / "summary.json").read_text(encoding="utf-8")
    )
    bank_manifest = json.loads(
        Path(args.bank_manifest).read_text(encoding="utf-8")
    )
    if int(bank_manifest["rows"]) != 32:
        raise ValueError("frozen confirmation bank does not have 32 rows")
    if int(bank_manifest["unique_prompts"]) != 32:
        raise ValueError("frozen confirmation bank does not have 32 unique prompts")
    if int(bank_manifest["unique_rollout_batches"]) != 32:
        raise ValueError("frozen confirmation bank does not have 32 rollout clusters")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["state_id"]), []).append(row)
    if set(grouped) != set(bank_manifest["case_ids"]):
        raise ValueError("scored state IDs differ from the frozen confirmation bank")

    identity_valid = all(
        bool(row["candidate_identity_valid"]) for row in rows
    )
    exact_accept = all(
        row["exact_transition_verdict"] == "ACCEPT" for row in rows
    )
    repeats_complete = all(len(state_rows) == 2 for state_rows in grouped.values())
    repeat_stable = True
    details_present = True
    primary = []
    for state_id, state_rows in sorted(grouped.items()):
        state_rows.sort(key=lambda row: int(row["repeat"]))
        if len(state_rows) != 2:
            repeat_stable = False
            continue
        signatures = {
            json.dumps(
                {
                    "count": row["greedy_token_disagreement_count"],
                    "sequence": row["greedy_sequence_disagreement"],
                    "records": row.get("greedy_disagreement_records"),
                },
                sort_keys=True,
            )
            for row in state_rows
        }
        repeat_stable = repeat_stable and len(signatures) == 1
        details_present = details_present and all(
            "greedy_disagreement_records" in row for row in state_rows
        )
        primary.append(state_rows[0])

    if not identity_valid:
        verdict = "INVALID"
        reason = "candidate identity failed"
    elif not exact_accept:
        verdict = "INDETERMINATE"
        reason = "whole-step exact core did not accept every scored state"
    elif not repeats_complete or not repeat_stable or not details_present:
        verdict = "INDETERMINATE"
        reason = "repeat stability or required decision detail is incomplete"
    else:
        disagreement_states = sum(
            bool(row["greedy_sequence_disagreement"]) for row in primary
        )
        verdict = "ACCEPT" if disagreement_states == 0 else "REJECT"
        reason = (
            "zero greedy sequence disagreements on the frozen bank"
            if verdict == "ACCEPT"
            else "one or more stable greedy sequence disagreements on the frozen bank"
        )

    event_records = []
    transition_counts: Counter[str] = Counter()
    for row in primary:
        for record in row.get("greedy_disagreement_records", []):
            left = record["reference"]
            right = record["candidate"]
            transition = f"{left['top1_token']}->{right['top1_token']}"
            transition_counts[transition] += 1
            event_records.append(
                {
                    "state_id": row["state_id"],
                    "token_position": int(left["token_position"]),
                    "target_token": int(left["target_token"]),
                    "eager_top1_token": int(left["top1_token"]),
                    "compiled_top1_token": int(right["top1_token"]),
                    "eager_margin": float(left["top1_top2_margin"]),
                    "compiled_margin": float(right["top1_top2_margin"]),
                }
            )

    sequence_disagreements = sum(
        bool(row["greedy_sequence_disagreement"]) for row in primary
    )
    token_disagreements = sum(
        int(row["greedy_token_disagreement_count"]) for row in primary
    )
    output = {
        "schema_version": "forkcert.qwen3-greedy-impact-confirmation.v0.1",
        "contract": str(
            (Path(__file__).parent / "QWEN3_GREEDY_IMPACT_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md").resolve()
        ),
        "source_result_dir": str(result_dir.resolve()),
        "bank_output_sha256": bank_manifest["output_sha256"],
        "states": len(primary),
        "repeats_per_state": 2 if repeats_complete else None,
        "candidate_identity_all_valid": identity_valid,
        "exact_core_all_accept": exact_accept,
        "repeat_stable": repeat_stable,
        "strict_greedy_compatibility_verdict": verdict,
        "verdict_reason": reason,
        "sequence_disagreement_states": sequence_disagreements,
        "finite_bank_sequence_disagreement_proportion": (
            sequence_disagreements / len(primary) if primary else None
        ),
        "token_disagreements": token_disagreements,
        "token_transition_counts": dict(sorted(transition_counts.items())),
        "event_records": event_records,
        "numerical_transition_verdict": run_summary["numerical_transition_verdict"],
        "correctness_claim": "NONE: compatibility is not mathematical correctness",
        "population_inference": "NOT CLAIMED: deterministic finite bank",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

