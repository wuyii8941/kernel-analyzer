#!/usr/bin/env python
"""Evaluate finite-bank incremental value of GRPO boundary conditioning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def average_precision(rows: list[dict[str, Any]], score: str) -> float:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[score]), str(row["trajectory"]),
            str(row["state_id"]), int(row["flat_index"]),
        ),
    )
    positives = sum(bool(row["event"]) for row in ordered)
    if positives == 0:
        return float("nan")
    seen = 0
    total = 0.0
    for rank, row in enumerate(ordered, 1):
        if row["event"]:
            seen += 1
            total += seen / rank
    return total / positives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", required=True, nargs="+")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parent = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    frozen_events = {
        (str(row["trajectory"]), str(row["state_id"]), int(row["flat_index"]))
        for row in parent["events"]
    }
    rows: list[dict[str, Any]] = []
    token_rows: list[tuple[str, dict[str, Any]]] = []
    for spec in args.tokens:
        if "=" not in spec:
            raise ValueError("each --tokens item must be TRAJECTORY=PATH")
        trajectory, raw_path = spec.split("=", 1)
        if trajectory not in {"A", "B"}:
            raise ValueError(f"unsupported trajectory label: {trajectory}")
        token_rows.extend((trajectory, row) for row in read_jsonl(Path(raw_path)))
    for trajectory, row in token_rows:
        sign = int(row["advantage_sign"])
        if sign == 0:
            continue
        ref = float(row["logp_ref_first"])
        alt = float(row["logp_alt_first"])
        old = float(row["old_logp"])
        delta = alt - ref
        boundary = math.log1p(0.2) if sign > 0 else math.log1p(-0.2)
        margin = sign * (boundary - (ref - old))
        aligned = sign * delta
        if margin > 0.0:
            pressure = max(0.0, aligned) / margin
        elif margin < 0.0:
            pressure = max(0.0, -aligned) / (-margin)
        else:
            pressure = math.inf if delta != 0.0 else 0.0
        key = (trajectory, str(row["state_id"]), int(row["flat_index"]))
        rows.append(
            {
                "trajectory": key[0],
                "state_id": key[1],
                "flat_index": key[2],
                "case_id": str(row["case_id"]),
                "token_index": int(row["token_index"]),
                "raw_score": abs(delta),
                "signed_delta": delta,
                "advantage_sign": sign,
                "reference_signed_margin": margin,
                "reference_boundary_distance": abs(margin),
                "boundary_score": pressure,
                "event": key in frozen_events,
            }
        )
    observed_events = {
        (row["trajectory"], row["state_id"], row["flat_index"])
        for row in rows if row["event"]
    }
    if observed_events != frozen_events:
        raise RuntimeError("event identity differs from frozen parent evaluation")
    if len(rows) != int(parent["applicable_tokens"]):
        raise RuntimeError(
            f"applicable denominator mismatch: {len(rows)} != {parent['applicable_tokens']}"
        )

    raw_order = sorted(
        rows,
        key=lambda row: (
            -row["raw_score"], row["trajectory"], row["state_id"], row["flat_index"]
        ),
    )
    boundary_order = sorted(
        rows,
        key=lambda row: (
            -row["boundary_score"], row["trajectory"], row["state_id"], row["flat_index"]
        ),
    )
    raw_rank = {
        (row["trajectory"], row["state_id"], row["flat_index"]): rank
        for rank, row in enumerate(raw_order, 1)
    }
    boundary_rank = {
        (row["trajectory"], row["state_id"], row["flat_index"]): rank
        for rank, row in enumerate(boundary_order, 1)
    }
    events = []
    for row in rows:
        if row["event"]:
            key = (row["trajectory"], row["state_id"], row["flat_index"])
            events.append({**row, "raw_rank": raw_rank[key], "boundary_rank": boundary_rank[key]})
    non_events = [row for row in rows if not row["event"]]
    max_raw_non_event = max(non_events, key=lambda row: row["raw_score"])
    min_raw_event = min(events, key=lambda row: row["raw_score"])
    payload = {
        "schema_version": "forkcert.qwen3-grpo-boundary-value.v0.1",
        "claim_scope": "RETROSPECTIVE_FINITE_BANK_CONSTRUCT_DIAGNOSTIC",
        "applicable_tokens": len(rows),
        "state_clusters": len({(row["trajectory"], row["state_id"]) for row in rows}),
        "parent_state_clusters": int(parent["total_rollout_states"]),
        "stable_event_count": len(events),
        "events": sorted(
            events,
            key=lambda row: (row["trajectory"], row["state_id"], row["flat_index"]),
        ),
        "raw_average_precision": average_precision(rows, "raw_score"),
        "boundary_average_precision": average_precision(rows, "boundary_score"),
        "rankings_identical": [
            (row["trajectory"], row["state_id"], row["flat_index"])
            for row in raw_order
        ] == [
            (row["trajectory"], row["state_id"], row["flat_index"])
            for row in boundary_order
        ],
        "max_raw_non_event": max_raw_non_event,
        "min_raw_event": min_raw_event,
        "nonredundant_on_frozen_bank": (
            not all(raw_rank[key] == boundary_rank[key] for key in frozen_events)
            or max_raw_non_event["raw_score"] > min_raw_event["raw_score"]
        ),
        "correctness": "NO CLAIM",
        "population_prevalence": "NO CLAIM",
        "predictive_generalization": "NO CLAIM",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
