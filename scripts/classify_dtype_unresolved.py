#!/usr/bin/env python3
"""Classify unresolved dtype topology rows without assigning correctness labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def boundary_class(symbol: str, reason: str) -> str:
    s = symbol.lower()
    if "softmax" in s or "log_softmax" in s:
        return "INTERNAL_SOFTMAX_SCHEDULE"
    if "rotary" in s or "bmm_cat_cos" in s:
        return "INTERNAL_ROTARY_SCHEDULE"
    if "rmsnorm" in s or "mean_mul_pow" in s or "add_div_expand_mul_pow_sum" in s or "add_mul_sum" in s:
        return "INTERNAL_RMSNORM_SCHEDULE"
    if "embedding" in s:
        return "SPECIALIZED_EMBEDDING_BOUNDARY"
    if "arange" in s or "cumsum" in s or "constant_pad" in s or "clone_expand" in s:
        return "POSITION_OR_LAYOUT_BOUNDARY"
    if reason == "family is recognized but pointer topology is absent from the BF16 catalog":
        return "RECOGNIZED_FAMILY_UNMATCHED_TOPOLOGY"
    return "UNREGISTERED_GENERATED_FAMILY"


def main() -> None:
    entries = []
    for dtype, tf32 in (("fp32", False), ("tf32", True)):
        for seq in (64, 128, 256):
            name = f"dtype_mapping_{dtype}_seq{seq}.json"
            data = json.loads((FINAL / name).read_text())
            rows = [row for row in data["rows"] if row["status"] != "MAPPED"]
            counts = Counter()
            invocation_counts = Counter()
            for row in rows:
                category = boundary_class(str(row["symbol"]), str(row["reason"]))
                counts[category] += 1
                invocation_counts[category] += int(row["invocations"])
            entries.append({
                "dtype": "fp32",
                "tf32": tf32,
                "seq_len": seq,
                "mapping_file": name,
                "unresolved_symbols": len(rows),
                "unresolved_invocations": sum(int(row["invocations"]) for row in rows),
                "by_boundary_class": [
                    {"class": category, "symbols": counts[category], "invocations": invocation_counts[category]}
                    for category in sorted(counts)
                ],
                "candidate_values_used_to_select_or_classify": False,
                "correctness_verdict_assigned": False,
            })
    output = {
        "schema": "kernel-analyzer-dtype-unresolved-boundary-v1",
        "subject": "Unresolved strict-FP32/TF32 generated topology classification",
        "entries": entries,
        "boundary": "Classes are name/topology diagnostics only. They do not prove that a row is internal, safe, or numerically equivalent; all unresolved rows remain in the implementation denominator.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = FINAL / "dtype_unresolved_boundary.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "entries": len(entries)}))


if __name__ == "__main__":
    main()
