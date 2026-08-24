#!/usr/bin/env python3
"""Freeze a clean v4.1 roster without repairing old experiment identity.

The earlier v4 package contains useful results, but several historical rows do
not contain all fields needed to replay them.  This command creates a new,
pre-run manifest that contains only rows with a complete identity.  Rows from
the old package are reported as ineligible; their numbers are never copied
into the new roster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "results/property/direct_persistence_v4"
DEFAULT_OUT = ROOT / "results/property/direct_persistence_v4_1"

REQUIRED = (
    "model",
    "exact_endpoint",
    "sequence_length",
    "state_order",
    "parameter_coordinate_digest",
    "optimizer",
    "moment_state",
    "horizon",
    "repair",
    "runner_source_digest",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def complete_identity(row: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [key for key in REQUIRED if row.get(key) in (None, "", [], {})]
    return not missing, missing


def build_gemma_row(pool_row: dict[str, Any]) -> dict[str, Any]:
    """Normalize the independently frozen Gemma row into v4.1 fields."""

    identity = {
        "case_id": pool_row.get("case_id"),
        "role": "NEW_IMPL",
        "model": pool_row.get("model"),
        "exact_endpoint": pool_row.get("endpoint"),
        "sequence_length": pool_row.get("sequence_length"),
        "state_order": {
            "formation": pool_row.get("formation_state_order"),
            "trajectory": pool_row.get("trajectory_state_order"),
        },
        "state_bank_digest": {
            "formation": pool_row.get("formation_state_bank_digest"),
            "trajectory": pool_row.get("trajectory_state_bank_digest"),
        },
        "parameter_coordinate_digest": pool_row.get("parameter_coordinate_digest"),
        "optimizer": pool_row.get("optimizer"),
        "moment_state": pool_row.get("optimizer", {}).get("moments"),
        "horizon": 32,
        "repair": pool_row.get("repair"),
        "runner_source_digest": pool_row.get("runtime_capture_digest"),
    }
    ok, missing = complete_identity(identity)
    return {
        **identity,
        "status": "READY" if ok else "INELIGIBLE_MISSING_IDENTITY",
        "missing": missing,
        "source": "results/property/direct_persistence_v4/heldout_gemma_pool.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    protocol = load(V4 / "protocol.json")
    protocol = {
        **protocol,
        "schema": "kernel-analyzer-direct-persistence-v4_1-protocol-v1",
        "status": "FROZEN_BEFORE_NEW_RUN",
        "supersedes": "results/property/direct_persistence_v4",
        "v4_1_rule": "Only complete experiment identities enter the executable roster; old missing fields are never inferred.",
    }

    pool = load(V4 / "heldout_gemma_pool.json")
    ready_rows = [build_gemma_row(row) for row in pool.get("rows", [])]

    # The three raw Gemma replays are useful evidence, but their compact v4
    # summaries do not contain all required identity fields.  Keep them in a
    # feasibility report rather than silently promoting them to a new roster.
    raw_candidates = [
        "gemma4_raw_softmax_backward",
        "gemma4_raw_gelu_loss_backward",
        "gemma4_raw_gelu_backward1860",
    ]
    ineligible = [
        {
            "case_id": case_id,
            "status": "INELIGIBLE_MISSING_IDENTITY",
            "missing": ["parameter_coordinate_digest", "moment_state", "repair", "runner_source_digest"],
            "source": "results/property/direct_persistence_v4/heldout/new_impl_targets_v2.json",
        }
        for case_id in raw_candidates
    ]

    roster = {
        "schema": "kernel-analyzer-direct-persistence-v4_1-roster-v1",
        "status": "FROZEN_BEFORE_NEW_RUN",
        "selection_rule": "Only rows with every required identity field are executable; incomplete old rows remain outside the roster.",
        "required_identity": list(REQUIRED),
        "rows": [row for row in ready_rows if row["status"] == "READY"],
        "excluded_rows": ineligible,
        "source_pool": "results/property/direct_persistence_v4/heldout_gemma_pool.json",
    }
    feasibility = {
        "schema": "kernel-analyzer-direct-persistence-v4_1-feasibility-v1",
        "status": "READY_ONE_ROW_EXPLICIT_EXCLUSIONS",
        "ready": [row["case_id"] for row in roster["rows"]],
        "ineligible": ineligible,
        "claim_boundary": "This is a clean pre-run roster, not a new scientific result. Existing v4 results are not copied into v4.1 measurements.",
    }
    execution = {
        "schema": "kernel-analyzer-direct-persistence-v4_1-execution-status-v1",
        "status": "NOT_STARTED_NEW_FREEZE",
        "roster": "roster.json",
        "ready_rows": len(roster["rows"]),
        "required_runs": ["16-step short screen", "32-step confirmation"],
        "all_rows_receive_confirmation": True,
        "missing_identity_action": "INELIGIBLE_WITH_REASON",
        "universal_oracle": "NOT_SUPPORTED",
    }
    manifest = {
        "schema": "kernel-analyzer-direct-persistence-v4_1-manifest-v1",
        "status": "FROZEN_BEFORE_NEW_RUN",
        "protocol": "protocol.json",
        "roster": "roster.json",
        "feasibility": "feasibility_report.json",
        "execution_status": "execution_status.json",
        "source_v4": digest(V4 / "protocol.json"),
    }
    for name, value in {
        "protocol.json": protocol,
        "roster.json": roster,
        "feasibility_report.json": feasibility,
        "execution_status.json": execution,
        "manifest.json": manifest,
    }.items():
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"status": feasibility["status"], "ready": len(roster["rows"]), "ineligible": len(ineligible)}, sort_keys=True))


if __name__ == "__main__":
    main()
