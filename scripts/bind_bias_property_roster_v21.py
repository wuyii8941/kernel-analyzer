#!/usr/bin/env python3
"""Deterministic v2.1 roster/provenance binder.

The v2 scaffold's runner SHA was a transient local commit and its wrapper
fields duplicated result artifact hashes.  v2.1 records immutable generator
and protocol commits, distinct source/wrapper fields, and explicit unavailable
repair/sham bindings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_v2_1"
BINDING_GENERATOR_COMMIT = "2642ca2bdb007cbf47edee1a6e9b53021906549a"
PROTOCOL_FREEZE_COMMIT = "02d9743a1c91b260e15fb0133576ab28f55eba21"
PROPERTY_SPECS_PATH = ROOT / "results/property/bias_formation_v2/property_specs.json"

sys.path.insert(0, str(ROOT))
from scripts.bind_bias_property_roster import CASE_BINDINGS, load, sha256, walk_state_ids, model_manifest  # noqa: E402


def script_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_hashes(value: Any, key_fragment: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if isinstance(child, str) and "sha256" in key_text.lower():
                found.append({"field": key_text, "sha256": child})
            found.extend(_collect_hashes(child, key_text))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_hashes(child, key_fragment))
    return found


def bind_case(spec: Mapping[str, Any]) -> dict[str, Any]:
    artifact_paths = [ROOT / str(path) for path in spec["artifacts"]]
    state_paths = [ROOT / str(path) for path in spec["state_sources"]]
    artifact_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths}
    state_ids: list[str] = []
    wrapper_hashes: list[dict[str, str]] = []
    for path in artifact_paths:
        if path.exists():
            payload = load(path)
            state_ids.extend(walk_state_ids(payload))
            wrapper_hashes.extend(_collect_hashes(payload))
    for path in state_paths:
        if path.exists():
            state_ids.extend(walk_state_ids(load(path)))
    unique_states = list(dict.fromkeys(state_ids))
    reasons: list[str] = []
    if spec["case_id"] == "flash_attention_literature_anchor":
        reasons.append("LITERATURE_ANCHOR_NOT_MEASURED")
    if len(set(unique_states)) < 32:
        reasons.append(f"INSUFFICIENT_UNIQUE_STATE_IDS:{len(set(unique_states))}")
    if spec["case_id"] == "qwen3vl_silu_seq160":
        reasons.append("INELIGIBLE_IN_V2_1_INSUFFICIENT_STATES:only_six_unique_natural_states")
    if spec["case_id"] in {"qwen_l23_key_materialization_seq1024", "qwen_rsqrt_seq128"}:
        reasons.append("INELIGIBLE_MISSING_CONSEQUENCE_TRACE")
    candidate_wrapper = [item for item in wrapper_hashes if "generated_source" in item["field"].lower() or "wrapper" in item["field"].lower()]
    candidate_wrapper_bound = bool(candidate_wrapper)
    # Existing result artifacts do not contain an independently addressable
    # repair/sham source path.  Never turn a name into a provenance claim.
    repair_bound = False
    sham_bound = False
    if not candidate_wrapper_bound:
        reasons.append("BLOCKED_MISSING_CANDIDATE_WRAPPER_SOURCE_HASH")
    if not repair_bound:
        reasons.append("BLOCKED_MISSING_REPAIR_SOURCE")
    if not sham_bound:
        reasons.append("BLOCKED_MISSING_SHAM_SOURCE")
    model = model_manifest(str(spec["model_root"])) if spec["model_root"] else {"complete": False, "manifest_sha256": None}
    if spec["model_root"] and not model["complete"]:
        reasons.append("MODEL_MANIFEST_INCOMPLETE")
    return {
        **dict(spec),
        "artifact_hashes": artifact_hashes,
        "candidate_release_and_wrapper_hashes": {
            "release_id": spec.get("proof_unit_id"),
            "wrapper_sources": candidate_wrapper,
            "artifact_hashes_are_not_wrapper_hashes": True,
        },
        "repair_binding": {
            "implementation_path": None,
            "source_sha256": None,
            "target_endpoint": spec.get("endpoint_or_region_id"),
            "bound": repair_bound,
        },
        "sham_binding": {
            "implementation_path": None,
            "source_sha256": None,
            "target_endpoint": spec.get("endpoint_or_region_id"),
            "bound": sham_bound,
        },
        "repair_and_sham_bound": repair_bound and sham_bound,
        "state_ids": unique_states,
        "state_count": len(set(unique_states)),
        "model_manifest": model,
        "feasibility": "CAPTURE_READY" if not reasons else "BLOCKED",
        "ineligibility_reasons": reasons,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [bind_case(spec) for spec in CASE_BINDINGS]
    provenance = {
        "binding_generator_commit": BINDING_GENERATOR_COMMIT,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "binding_script_sha256": script_sha256(),
        "property_specs_sha256": sha256(PROPERTY_SPECS_PATH),
        "runtime_commit": None,
        "dirty_worktree": None,
        "runner_source_sha256": None,
        "environment_manifest_sha256": None,
    }
    roster = {
        "schema": "kernel-analyzer-bias-property-roster-v2_1",
        "status": "PRE_MEASUREMENT_STATISTICAL_CORRECTION",
        "supersedes": "bias_formation_v2",
        "v2_gpu_measurements": 0,
        "provenance": provenance,
        "cases": cases,
    }
    feasibility = {
        "schema": "kernel-analyzer-bias-property-feasibility-v2_1",
        "status": "PRE_MEASUREMENT_STATISTICAL_CORRECTION",
        "provenance": provenance,
        "gpu_campaign_started": False,
        "cases": [{"case_id": row["case_id"], "feasibility": row["feasibility"], "reasons": row["ineligibility_reasons"]} for row in cases],
    }
    for name, value in (("roster_bound.json", roster), ("feasibility_report.json", feasibility)):
        path = OUT / name
        temp = path.with_name("." + path.name + ".tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(path)
    print(json.dumps({"output_dir": str(OUT), "gpu_campaign_started": False}, sort_keys=True))


if __name__ == "__main__":
    main()
