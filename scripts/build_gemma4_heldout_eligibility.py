#!/usr/bin/env python3
"""Freeze the nonzero, new-implementation Gemma-4 follow-up denominator."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


def read(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--prior-censuses", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--negative-sample-size", type=int, default=12)
    args = parser.parse_args()

    known_patterns = set()
    for path in args.prior_censuses:
        for row in read(path).get("implementations", []):
            known_patterns.add(row["implementation_pattern_id"])
    census = read(args.census)
    identities = {
        row["implementation_pattern_id"]: row
        for row in census["implementations"]
    }
    rows = []
    for row in read(args.screen)["rows"]:
        nonzero = row["verdict"] == "FINITE_SCREENED" and (row.get("rms_mean") or 0) > 0
        new_impl = row["implementation_pattern_id"] not in known_patterns
        disposition = (
            "ELIGIBLE_NONZERO_NEW_IMPL"
            if nonzero and new_impl
            else "NOT_APPLICABLE_NO_RESIDUAL"
            if row["verdict"] == "EXACT_ZERO"
            else "SEEN_IMPLEMENTATION"
            if nonzero
            else "UNRESOLVED_OR_NONFINITE"
        )
        rows.append({
            **row,
            "implementation_novelty": "NEW_IMPL_PATTERN" if new_impl else "SEEN_IMPL",
            "eligibility": disposition,
            "invocation_count": identities.get(row["implementation_pattern_id"], {}).get(
                "invocation_count"
            ),
        })

    eligible = [row for row in rows if row["eligibility"] == "ELIGIBLE_NONZERO_NEW_IMPL"]
    # Semantic hotspots are selected only by the preregistered bottleneck type,
    # never by p-value, amplification, or residual magnitude.
    hotspot_rules = {
        "LOSS_SOFTCAP_CE_FB": lambda row: (
            row["phase"] == "BACKWARD" and "log_softmax" in row["operation"]
            and "tanh_backward" in row["operation"] and "nll_loss_backward" in row["operation"]
        ),
        "NORMALIZATION_PLE_RMS_FB": lambda row: (
            row["phase"] == "FORWARD" and "embedding_mean_mul_pow" in row["operation"]
        ),
        "ATTENTION_SOFTMAX_FB": lambda row: (
            row["phase"] == "FORWARD" and "new_ones_prepare_softmax_online" in row["operation"]
        ),
    }
    hotspots = []
    for role, rule in hotspot_rules.items():
        choices = sorted(
            (row for row in eligible if rule(row)),
            key=lambda row: (row["implementation_pattern_id"], row["endpoint"]),
        )
        hotspots.append({
            "role": role,
            "status": "BOUND" if choices else "UNRESOLVED_ABSENT",
            "representative": choices[0] if choices else None,
            "selection": "LEXICOGRAPHIC_WITHIN_PREREGISTERED_SEMANTIC_BOTTLENECK",
        })

    hotspot_keys = {
        (row["representative"]["implementation_pattern_id"], row["representative"]["endpoint"])
        for row in hotspots if row["representative"] is not None
    }
    raw_negative_pool = sorted(
        (
            row for row in eligible
            if not row["screen_positive_bh_q_0_10"]
            and (row["implementation_pattern_id"], row["endpoint"]) not in hotspot_keys
        ),
        key=lambda row: (row["implementation_pattern_id"], row["endpoint"]),
    )
    # Recall auditing is about missed semantic mechanisms, not repeated layers,
    # endpoints, or shape-specialized copies of the same operator family.
    by_semantic_family = {}
    for row in raw_negative_pool:
        by_semantic_family.setdefault(row["semantic_family_id"], row)
    negative_pool = sorted(
        by_semantic_family.values(),
        key=lambda row: (row["semantic_family_id"], row["implementation_pattern_id"], row["endpoint"]),
    )
    generator = random.Random(args.seed)
    selected = generator.sample(
        negative_pool, min(args.negative_sample_size, len(negative_pool))
    )
    selected.sort(key=lambda row: (row["implementation_pattern_id"], row["endpoint"]))
    payload = {
        "schema": "kernel-analyzer-gemma4-heldout-eligibility-v1",
        "status": "FROZEN_BEFORE_PARAMETER_REACH_OR_TRAJECTORY",
        "seed": args.seed,
        "denominator": {
            "pattern_endpoints": len(rows),
            "nonzero_new_impl_pattern_endpoints": len(eligible),
            "bh_screen_positive": sum(row["screen_positive_bh_q_0_10"] for row in eligible),
            "recall_audit_random_screen_negatives": len(selected),
            "recall_audit_distinct_semantic_families": len({
                row["semantic_family_id"] for row in selected
            }),
        },
        "semantic_hotspots": hotspots,
        "random_screen_negative_recall_audit": selected,
        "all_rows": rows,
        "counting_rule": (
            "All invocations remain in coverage. One representative per implementation pattern "
            "is deeply measured; zero residual and no parameter reach are NOT_APPLICABLE, not negatives."
        ),
        "claim_boundary": (
            "Screen statistics only determine nonzero eligibility. Hotspots are selected by a "
            "frozen semantic-bottleneck rule and recall controls by seeded random sampling."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["denominator"], sort_keys=True))


if __name__ == "__main__":
    main()
