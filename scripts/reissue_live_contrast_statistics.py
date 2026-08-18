#!/usr/bin/env python3
"""Reissue live direction certificates from retained complete Gram matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.streaming import direction_certificate_from_gram  # noqa: E402


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    payload = json.loads(args.path.read_text())
    prior_hash = payload["result_sha256"]
    for row in payload["results"]:
        old = row["direction"]
        row["direction"] = direction_certificate_from_gram(
            np.asarray(old["gram"], dtype=np.float64),
            coordinates=int(old["coordinates"]), state_ids=old["state_ids"],
            state_vector_sha256=old["state_vector_sha256"],
            bootstrap_draws=args.bootstrap_draws, seed=14031,
        )
        row["t1_eligible"] = (
            row["contrast_axis"] != "TOTAL" and row["finite"]
            and row["max_abs"] > 0.0
            and row["direction"]["cluster_bootstrap_95"]["lower_95"] > 0.0
        )
    payload["statistics_revision"] = {
        "method": "DISTINCT_ORIGINAL_CLUSTER_PAIR_BOOTSTRAP",
        "supersedes_result_sha256": prior_hash,
    }
    payload["result_sha256"] = canonical({
        key: value for key, value in payload.items() if key != "result_sha256"
    })
    temporary = args.path.with_name("." + args.path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.path)
    print(json.dumps({
        "path": str(args.path),
        "t1_positive_arms": sum(row["t1_eligible"] for row in payload["results"]),
        "result_sha256": payload["result_sha256"],
    }))


if __name__ == "__main__":
    main()
