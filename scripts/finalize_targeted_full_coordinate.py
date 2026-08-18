#!/usr/bin/env python3
"""Finalize a saved targeted T1 population without rerunning the GPU step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.analyze_generated_fp32_screen import bootstrap, u_statistic
from scripts.run_targeted_full_coordinate import (
    canonical_hash, nondegenerate_bootstrap_counts, write,
)
from scripts.run_generated_fp32_screen import file_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload["status"] != "RUNNING" or payload["states_complete"] < 2:
        raise RuntimeError("input is not a complete-enough saved T1 population")
    queue = json.loads(args.queue.read_text())
    capture = json.loads((args.release_dir / "capture.json").read_text())
    rows = payload["rows"]
    errors = []
    for row in rows:
        repeats = row["repeats"]
        if len(repeats) != 2 or repeats[0]["identity"] != repeats[1]["identity"]:
            raise RuntimeError(f"repeat identity failure: {row['state_id']}")
        left = np.asarray(repeats[0]["target"]["signed_error"], dtype=np.float64)
        right = np.asarray(repeats[1]["target"]["signed_error"], dtype=np.float64)
        if not np.array_equal(left, right):
            raise RuntimeError(f"repeat target error failure: {row['state_id']}")
        errors.append((left + right) / 2.0)
    matrix = np.stack(errors)
    counts = nondegenerate_bootstrap_counts(len(rows), args.bootstrap_draws, 14031)
    confidence = bootstrap(matrix, counts)
    statistic = u_statistic(matrix)
    output = {
        "schema": "kernel-analyzer-targeted-full-coordinate-t1-v1",
        "status": (
            "COMPLETE_FULL_COORDINATE_T1_PILOT"
            if len(rows) < 32 else "COMPLETE_FULL_COORDINATE_T1"
        ),
        "candidate_id": payload["candidate_id"],
        "queue_sha256": queue["result_sha256"],
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(args.input_bank),
        "states": len(rows),
        "repeats": 2,
        "coordinates": int(matrix.shape[1]),
        "cross_state_inner_product_u": statistic,
        "cluster_bootstrap_95": confidence,
        "directional_positive": confidence["lower_95"] > 0.0,
        "runtime_repeat_exact": True,
        "rows": rows,
        "gates": {
            "exact_frozen_wrapper_identity": True,
            "all_coordinates_observed": int(matrix.shape[1]) == 1536,
            "runtime_repeat_exact": True,
            "directional_t1": confidence["lower_95"] > 0.0,
            "independent_32_state_population": len(rows) == 32,
        },
        "claim_boundary": (
            "T1 only. A positive pilot is not a case and does not establish generated backward "
            "binding, causal repair, a complete carrier, or accumulation."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    write(args.input, output)
    print(json.dumps({
        "states": len(rows), "coordinates": int(matrix.shape[1]),
        "u": statistic, "lower_95": confidence["lower_95"],
        "directional_positive": output["directional_positive"],
    }))


if __name__ == "__main__":
    main()
