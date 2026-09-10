#!/usr/bin/env python3
"""Attach rechecked selected-plan measurements to the release-qualified inventory.

This is bookkeeping only.  A verified saved measurement becomes a verified
inventory position; it does not become a bias finding, a population claim, or
a training-outcome result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALID = {"VERIFIED", "RECORDED_MEASUREMENT_CHECKED"}


def _read(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(inventory, audit, mappings, *, verify_artifacts=False):
    if audit.get("schema") != "numerical-execution-manifest-audit-v1":
        raise ValueError("Unexpected execution-audit schema")
    result = copy.deepcopy(inventory)
    records = result.get("records", [])
    by_identity = {}
    releases_by_name = {}
    for row in records:
        release = str(Path(row["release"]).resolve())
        identity = (release, row["task_id"])
        if identity in by_identity:
            raise ValueError(f"Duplicate release-qualified position: {identity}")
        by_identity[identity] = row
        releases_by_name.setdefault(Path(release).name, set()).add(release)

    audited = {row["name"]: row for row in audit["families"]}
    updated = []
    already_valid = []
    seen = set()
    for spec in mappings:
        name = spec["audit_family"]
        if name not in audited:
            raise ValueError(f"Audit family is absent: {name}")
        family = audited[name]
        if not family.get("eligible_measurement_complete"):
            raise ValueError(f"Selected plan is not complete: {name}")
        verified = [row for row in family["records"] if row["status"] in VALID]
        if len(verified) != family["counts"].get("VERIFIED", 0):
            raise ValueError(f"Verified count differs from audit summary: {name}")
        matching_releases = releases_by_name.get(spec["release_name"], set())
        if len(matching_releases) != 1:
            raise ValueError(
                f"Release name does not identify exactly one release: "
                f"{spec['release_name']} ({len(matching_releases)} matches)"
            )
        release = next(iter(matching_releases))
        for evidence in verified:
            identity = (release, evidence["task_id"])
            if identity in seen:
                raise ValueError(f"Repeated merge target: {identity}")
            seen.add(identity)
            if identity not in by_identity:
                raise ValueError(f"Inventory position is absent: {identity}")
            if verify_artifacts:
                artifact = Path(evidence["raw_artifact"])
                if not artifact.is_absolute():
                    artifact = ROOT / artifact
                if not artifact.is_file() or _sha(artifact) != evidence["raw_sha256"]:
                    raise ValueError(f"Saved measurement artifact changed: {artifact}")
            target = by_identity[identity]
            previous = target.get("runtime_measurement_status")
            if previous in VALID:
                already_valid.append(identity)
            else:
                target["runtime_measurement_status"] = "VERIFIED"
                updated.append(identity)
            references = list(target.get("reference_candidates", []))
            if not any(ref.get("family") == spec["reference_family"] for ref in references):
                references.append({
                    "family": spec["reference_family"],
                    "source": "RECHECKED_SELECTED_PLAN_MEASUREMENT",
                })
            target["reference_candidates"] = references
            target["selected_plan_measurement_evidence"] = {
                "audit_family": name,
                "case_id": evidence["case_id"],
                "raw_artifact": evidence["raw_artifact"],
                "raw_sha256": evidence["raw_sha256"],
                "claim_scope": evidence["analysis"]["claim_scope"],
            }

    result["selected_plan_measurement_merge"] = {
        "updated_positions": len(updated),
        "already_valid_positions": len(already_valid),
        "merged_positions": len(seen),
        "scope": (
            "Saved measurements reattached to release-qualified positions; "
            "not a bias, population, mechanism, or training-outcome claim"
        ),
    }
    statuses = {}
    for row in records:
        status = row.get("runtime_measurement_status", "NOT_ASSESSED")
        statuses[status] = statuses.get(status, 0) + 1
    result["runtime_counts"] = statuses
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    manifest = _read(args.manifest)
    if manifest.get("schema") != "family-measurement-inventory-merge-v1":
        raise ValueError("Unexpected merge-manifest schema")
    result = merge(
        _read(args.inventory), _read(args.audit), manifest["mappings"],
        verify_artifacts=True,
    )
    result["measurement_merge_input_sha256"] = {
        str(args.inventory): _sha(args.inventory),
        str(args.audit): _sha(args.audit),
        str(args.manifest): _sha(args.manifest),
        str(Path(__file__)): _sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(result["selected_plan_measurement_merge"], ensure_ascii=False))


if __name__ == "__main__":
    main()
