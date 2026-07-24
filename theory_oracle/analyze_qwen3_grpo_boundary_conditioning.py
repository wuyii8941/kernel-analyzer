#!/usr/bin/env python
"""Descriptive boundary-vs-raw-delta analysis for the valid GRPO v0.2 bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median

from forkcert.detector import clip_active, clip_boundary


def auc(labels: list[int], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores, strict=True) if label]
    negative = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positive or not negative:
        return None
    wins = 0.0
    for left in positive:
        for right in negative:
            wins += 1.0 if left > right else 0.5 if left == right else 0.0
    return wins / (len(positive) * len(negative))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for trajectory, path in (("A", Path(args.a)), ("B", Path(args.b))):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sign = int(row["advantage_sign"])
            if sign == 0:
                continue
            ref, alt, old = map(float, (row["logp_ref"], row["logp_alt"], row["old_logp"]))
            delta = alt - ref
            margin = abs((ref - old) - clip_boundary(sign, 0.2))
            event = clip_active(ref, old, sign, 0.2) != clip_active(alt, old, sign, 0.2)
            rows.append(
                {
                    "trajectory": trajectory,
                    "event": event,
                    "signed_delta": delta,
                    "absolute_delta": abs(delta),
                    "reference_margin": margin,
                    "boundary_ratio": abs(delta) / max(margin, 1e-300),
                }
            )
    labels = [int(row["event"]) for row in rows]
    raw_scores = [row["absolute_delta"] for row in rows]
    proximity_scores = [-row["reference_margin"] for row in rows]
    ratio_scores = [row["boundary_ratio"] for row in rows]
    events = [row for row in rows if row["event"]]
    non_events = [row for row in rows if not row["event"]]
    minimum_event_delta = min(row["absolute_delta"] for row in events)
    maximum_event_margin = max(row["reference_margin"] for row in events)
    payload = {
        "schema_version": "forkcert.qwen3-grpo-boundary-conditioning.v0.1",
        "scope": "descriptive finite-bank analysis; no new compatibility verdict",
        "applicable_tokens": len(rows),
        "events": len(events),
        "mean_signed_delta": fmean(row["signed_delta"] for row in rows),
        "event_absolute_delta_min": minimum_event_delta,
        "event_absolute_delta_median": median(row["absolute_delta"] for row in events),
        "event_absolute_delta_max": max(row["absolute_delta"] for row in events),
        "event_reference_margin_max": maximum_event_margin,
        "non_event_absolute_delta_max": max(row["absolute_delta"] for row in non_events),
        "non_events_with_delta_at_least_min_event_delta": sum(
            row["absolute_delta"] >= minimum_event_delta for row in non_events
        ),
        "non_events_at_least_as_close_as_farthest_event": sum(
            row["reference_margin"] <= maximum_event_margin for row in non_events
        ),
        "auc_raw_absolute_delta": auc(labels, raw_scores),
        "auc_boundary_proximity_only": auc(labels, proximity_scores),
        "auc_delta_to_margin_ratio": auc(labels, ratio_scores),
        "interpretation": (
            "Neither magnitude nor proximity alone defines an event. Their relational geometry is more informative: "
            "the delta must be large enough and aligned toward the sign-specific boundary."
        ),
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
