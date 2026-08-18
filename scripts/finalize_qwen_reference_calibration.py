#!/usr/bin/env python3
"""Align and aggregate the compressed 48-state reference calibration."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from forkcert.all_op_reference_calibration import (  # noqa: E402
    _digest,
    build_stratified_reference_calibration,
)
from forkcert.reference_pair_alignment import build_reference_pair_alignment  # noqa: E402


DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"
PROTOCOL = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/all_op_reference_protocol_v2.json"
INPUT = ROOT / "results/calibration"
OUTPUT = ROOT / "results/coverage/qwen_reference_calibration.json.gz"
MANIFEST = ROOT / "results/calibration/manifest.json"


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def artifact(state_id: str, kind: str) -> Path:
    return INPUT / f"{state_id}.{kind}.json.gz"


def main() -> None:
    design = json.loads(DESIGN.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    states = [row for row in design["records"] if row["split"] == "calibration"]
    bundles = []
    bindings = []
    for state in states:
        state_id = state["sequence_id"]
        paths = {
            "bf16_repeat0": artifact(state_id, "bf16.r0"),
            "bf16_repeat1": artifact(state_id, "bf16.r1"),
            "fp32_repeat0": artifact(state_id, "fp32.r0"),
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"calibration incomplete for {state_id}: {missing}")
        values = {name: load(path) for name, path in paths.items()}
        alignment = build_reference_pair_alignment(
            bf16_artifact=values["bf16_repeat0"],
            fp32_artifact=values["fp32_repeat0"],
            bf16_repeat_artifact=values["bf16_repeat1"],
        )
        required_alignment_gates = (
            "matched_state",
            "bf16_complete_step_stable",
            "fp32_complete_step_stable",
            "all_bf16_operations_accounted",
            "all_fp32_operations_accounted",
            "bf16_repeat_population_exact",
            "bf16_repeat_values_exact",
        )
        if not all(alignment["gates"][key] for key in required_alignment_gates):
            raise RuntimeError(f"reference alignment failed for {state_id}")
        bundles.append({"alignment": alignment, **values})
        bindings.append({
            "state_id": state_id,
            "stratum": state["length_bucket"],
            "artifacts": {
                name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "alignment_sha256": alignment["alignment_sha256"],
            "alignment_status": alignment["status"],
            "all_reference_pairs_resolved": alignment["gates"]["all_reference_pairs_resolved"],
        })

    result = build_stratified_reference_calibration(
        protocol=protocol,
        state_bundles=bundles,
        expected_strata=("seq64", "seq128", "seq256"),
        minimum_states_per_stratum=16,
        expected_state_ids=[row["sequence_id"] for row in states],
        quantile=0.95,
        bootstrap_draws=1000,
        bootstrap_seed=3407,
    )
    result["bindings"] = {
        "state_design_sha256": design["design_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "states": bindings,
    }
    result.pop("calibration_sha256", None)
    result["calibration_sha256"] = _digest(result)
    with gzip.open(OUTPUT, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema": "kernel-analyzer-qwen-calibration-artifact-manifest-v1",
        "status": "COMPLETE_48_STATE_REFERENCE_CALIBRATION",
        "states": bindings,
        "aggregate": {
            "path": str(OUTPUT.relative_to(ROOT)),
            "sha256": sha256_file(OUTPUT),
            "calibration_sha256": result["calibration_sha256"],
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "status": result["status"],
        "state_counts_by_stratum": result["state_counts_by_stratum"],
        "summary": result["summary"],
        "calibration_sha256": result["calibration_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
