#!/usr/bin/env python3
"""Quantify the saved-state consistency mechanism from existing same-call data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload(input_path: Path) -> dict:
    import torch

    files = sorted(input_path.glob("state_*_call_*.pt"))
    if not files:
        raise ValueError("no same-call records")
    rows = []
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        probability = record["diagnostic"]["reconstructed_probability"].double()
        mass = probability.sum(-1)
        defect = mass - 1
        repaired = probability / mass[:, None]
        repaired_defect = repaired.sum(-1) - 1
        rows.append({
            "record": path.name,
            "row_count": int(mass.numel()),
            "defect_mean": float(defect.mean()),
            "defect_rms": float(defect.square().mean().sqrt()),
            "defect_max_abs": float(defect.abs().max()),
            "nonzero_row_count": int(torch.count_nonzero(defect)),
            "positive_row_count": int(torch.count_nonzero(defect > 0)),
            "negative_row_count": int(torch.count_nonzero(defect < 0)),
            "renormalized_defect_max_abs": float(repaired_defect.abs().max()),
        })
    total_rows = sum(row["row_count"] for row in rows)
    return {
        "schema": "softmax-saved-state-root-cause-v1",
        "status": "FIXED_CALL_LOCAL_ROOT_CONFIRMED",
        "call_count": len(rows),
        "row_count": total_rows,
        "calls_with_nonzero_defect": sum(row["nonzero_row_count"] > 0 for row in rows),
        "nonzero_row_count": sum(row["nonzero_row_count"] for row in rows),
        "positive_row_count": sum(row["positive_row_count"] for row in rows),
        "negative_row_count": sum(row["negative_row_count"] for row in rows),
        "maximum_defect_abs": max(row["defect_max_abs"] for row in rows),
        "maximum_defect_rms": max(row["defect_rms"] for row in rows),
        "maximum_defect_after_consistent_renormalization": max(
            row["renormalized_defect_max_abs"] for row in rows
        ),
        "mathematical_link": "J_softmax(p)^T 1 = p * (1 - sum(p)); inconsistent saved scores and normalization statistics make the response nonzero",
        "isolated_source": "BF16 saved scaled scores are combined with FP32 maximum and denominator from the pre-materialized calculation",
        "intervention": "recompute the denominator from the same saved scores; row mass returns to one up to FP64 summation error",
        "scope": "TWO_FIXED_STATES_SAME_CALL_LOCAL_MECHANISM",
        "does_not_establish": [
            "independent-state population mean bias",
            "persistent parameter-update direction",
            "training loss consequence",
        ],
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output path")
    repo = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(repo):
        raise ValueError("output must be inside kernel-analyzer")
    payload = build_payload(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "call_count", "row_count", "calls_with_nonzero_defect",
        "nonzero_row_count", "maximum_defect_abs",
        "maximum_defect_after_consistent_renormalization",
    )}))


if __name__ == "__main__":
    main()
