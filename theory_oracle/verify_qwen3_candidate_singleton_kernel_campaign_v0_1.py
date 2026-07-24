#!/usr/bin/env python
"""Audit shared-compile singleton campaign without crediting invalid families."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path, result_path, out = map(lambda value: Path(value).resolve(), (args.manifest, args.result, args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    root = manifest_path.parents[1]
    manifest, result = json.loads(manifest_path.read_text()), json.loads(result_path.read_text())
    artifact_checks = {}
    for name, row in manifest["artifacts"].items():
        observed = sha(root / row["path"])
        artifact_checks[name] = {"expected": row["sha256"], "observed": observed, "pass": observed == row["sha256"]}

    expected_families = manifest["families"]
    family_audits = {}
    for family in expected_families:
        row = result.get("families", {}).get(family, {})
        valid = row.get("status", "").startswith("VALID_ORIGINAL_CANDIDATE_")
        semantic_reconstruction_valid = family != "triton_poi_fused__to_copy_t_17"
        semantic_issue = None if semantic_reconstruction_valid else (
            "v0.1 wrote a reshaped source into a non-contiguous transposed destination; "
            "it did not reproduce the generated kernel's physical cast layout"
        )
        gates = row.get("gates", {})
        call_integrity = len(row.get("call_records", [])) == 2 and all(
            len([item for item in record.values() if item.get("calls", 0) > 0]) == 1
            and [item for item in record.values() if item.get("calls", 0) > 0][0] == {"calls": 1, "repairs": 1}
            for record in row.get("call_records", [])
        )
        arithmetic = False
        if valid:
            try:
                effect = float(row["candidate_to_repair"]["l2"])
                direction = row["direction"]
                base, repaired = float(direction["eager_candidate_l2"]), float(direction["eager_repair_l2"])
                change, fraction = float(direction["l2_distance_change"]), float(direction["fractional_l2_reduction"])
                cosine = direction["cosine_repair_with_candidate_to_eager"]
                arithmetic = all(math.isfinite(value) for value in (effect, base, repaired, change, fraction))
                arithmetic &= math.isclose(change, repaired - base, rel_tol=1e-7, abs_tol=1e-9)
                arithmetic &= math.isclose(fraction, -change / base, rel_tol=1e-7, abs_tol=1e-9)
                arithmetic &= (cosine is None) if effect == 0.0 else (cosine is not None and math.isfinite(float(cosine)) and abs(float(cosine)) <= 1.000001)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                arithmetic = False
        family_audits[family] = {
            "status": "VALID_FAMILY_AUDIT" if valid and semantic_reconstruction_valid and gates and all(gates.values()) and call_integrity and arithmetic else "INVALID_FAMILY_AUDIT",
            "treatment_status": row.get("status"),
            "gates": gates,
            "call_integrity": call_integrity,
            "arithmetic_integrity": arithmetic,
            "semantic_reconstruction_valid": semantic_reconstruction_valid,
            "semantic_issue": semantic_issue,
            "failure": row.get("failure"),
            "effect": row.get("candidate_to_repair"),
            "direction": row.get("direction"),
        }

    campaign_gates = {
        "manifest_frozen": manifest.get("status") == "FROZEN_PRE_EXECUTION",
        "artifact_hashes_exact": all(row["pass"] for row in artifact_checks.values()),
        "campaign_status_valid": result.get("status") == "VALID_FAIL_CLOSED_CAMPAIGN",
        "global_gates_true": bool(result.get("global_gates")) and all(result["global_gates"].values()),
        "family_set_exact": sorted(result.get("families", {})) == sorted(expected_families),
    }
    valid_count = sum(row["status"] == "VALID_FAMILY_AUDIT" for row in family_audits.values())
    payload = {
        "schema_version": "forkcert.qwen3-candidate-singleton-kernel-campaign-audit.v0.1",
        "status": "VALID_FAIL_CLOSED_CAMPAIGN_AUDIT" if all(campaign_gates.values()) else "INVALID_CAMPAIGN_AUDIT",
        "campaign_gates": campaign_gates,
        "artifact_checks": artifact_checks,
        "valid_family_audits": valid_count,
        "invalid_family_audits": len(family_audits) - valid_count,
        "families": family_audits,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "campaign_gates": campaign_gates, "valid": valid_count, "invalid": len(family_audits) - valid_count, "families": {key: {"audit": row["status"], "treatment": row["treatment_status"], "failure": row["failure"]} for key, row in family_audits.items()}}, indent=2, sort_keys=True))
    if payload["status"] == "INVALID_CAMPAIGN_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
