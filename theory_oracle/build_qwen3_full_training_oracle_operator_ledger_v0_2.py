#!/usr/bin/env python
"""Extend the integrated Qwen3 ledger with final-norm transport evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--final-norm", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base_path = Path(args.base).resolve()
    norm_path = Path(args.final_norm).resolve()
    base = json.loads(base_path.read_text())
    norm = json.loads(norm_path.read_text())
    if base["status"] != "VALID_INTEGRATED_LEDGER":
        raise ValueError("base integrated ledger is invalid")
    if norm["status"] != "VALID_FINAL_NORM_ATTRIBUTION_TRANSPORT_EVALUATION":
        raise ValueError("final-norm transport evaluation is invalid")
    if not all(norm["gates"].values()):
        raise ValueError("final-norm transport gates failed")

    payload = copy.deepcopy(base)
    payload["schema_version"] = "forkcert.qwen3-full-training-oracle-operator-ledger.v0.2"
    payload["causal_coverage"]["backward"]["transport_tested_nonnull_candidates"] = 2
    payload["oracle_application"]["operator_attribution_mechanism_diversity"] = {
        "activation_backward_region": {
            "treatment": "middle SiLU backward call",
            "verdict": payload["oracle_application"]["operator_attribution_transport"][
                "verdict"
            ],
        },
        "norm_reduction_backward_region": {
            "treatment": "singleton final-RMSNorm backward region",
            "verdict": norm["verdict"],
            "endpoint_verdicts": norm["endpoint_verdicts"],
            "runtime_repeats_exact": norm["gates"]["all_runtime_repeats_exact"],
            "negative_control_exact_null": norm["gates"][
                "cast_control_exact_null_all_states"
            ],
        },
        "joint_interpretation": (
            "two mechanistically distinct fused-region repairs are state-conditional; "
            "A is null, B/C are non-null, and B/C effect vectors are nearly orthogonal"
        ),
    }
    payload["input_artifacts"]["base_integrated_ledger"] = {
        "path": str(base_path),
        "sha256": sha256(base_path),
    }
    payload["input_artifacts"]["final_norm_transport"] = {
        "path": str(norm_path),
        "sha256": sha256(norm_path),
    }
    payload["blocking_gaps"] = [
        gap.replace(
            "three selected transport states",
            "three selected transport states across two non-null-at-B treatments",
        )
        for gap in payload["blocking_gaps"]
    ]

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "verdicts": payload["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
