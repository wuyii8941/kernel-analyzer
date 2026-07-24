#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.stats import mean, percentile


def keyed(payload: dict) -> dict[tuple[str, int, int], float]:
    return {
        (str(row["case_id"]), int(row["token_index"]), int(row["token_id"])): float(row["logp"])
        for row in payload["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two archived vLLM chosen-logprob runs.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pa = json.loads(Path(args.a).read_text())
    pb = json.loads(Path(args.b).read_text())
    a = keyed(pa)
    b = keyed(pb)
    if set(a) != set(b):
        raise ValueError("score key coverage mismatch")
    signed = [b[key] - a[key] for key in sorted(a)]
    absolute = [abs(value) for value in signed]
    result = {
        "schema_version": "forkcert.p1.vllm-score-comparison.v1",
        "name": args.name,
        "a": args.a,
        "b": args.b,
        "tokens": len(absolute),
        "bitwise_mismatches": sum(value != 0.0 for value in signed),
        "bitwise_equal": all(value == 0.0 for value in signed),
        "mean_signed_b_minus_a": mean(signed),
        "mean_abs": mean(absolute),
        "p99_abs": percentile(absolute, 99),
        "max_abs": max(absolute, default=0.0),
        "metadata_a": pa["metadata"],
        "metadata_b": pb["metadata"],
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
