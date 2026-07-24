#!/usr/bin/env python
"""Independently audit the frozen Qwen3 backward argument-metadata census."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_safe(record: dict[str, Any]) -> bool:
    allowed_tensor = {
        "kind",
        "shape",
        "stride",
        "dtype",
        "device",
        "storage_offset",
        "requires_grad",
    }
    allowed_scalar = {"kind", "type", "value"}
    allowed_opaque = {"kind", "type"}
    kind = record.get("kind")
    if kind == "tensor":
        return set(record) == allowed_tensor
    if kind in {"scalar", "scalar_like"}:
        return set(record) == allowed_scalar
    if kind == "opaque":
        return set(record) == allowed_opaque
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--transition-result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    metadata_path = Path(args.metadata).resolve()
    transition_path = Path(args.transition_result).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    root = manifest_path.parents[1]
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    transition = json.loads(transition_path.read_text())
    static_path = root / manifest["artifacts"]["static_backward_summary"]["path"]
    static = json.loads(static_path.read_text())
    expected_family = {row["name"]: int(row["call_count"]) for row in static["kernel_families"]}
    expected_external = {
        name: int(count) for name, count in static["extern_kernel_call_site_counts"].items()
    }
    observed_family = {name: len(rows) for name, rows in metadata.get("family_calls", {}).items()}
    observed_external = {name: len(rows) for name, rows in metadata.get("external_calls", {}).items()}

    safe = True
    indices_exact = True
    for collection in (metadata.get("family_calls", {}), metadata.get("external_calls", {})):
        for rows in collection.values():
            if [row.get("index") for row in rows] != list(range(len(rows))):
                indices_exact = False
            for row in rows:
                for value in row.get("positional", []):
                    safe = safe and metadata_safe(value)
                for value in row.get("keyword", {}).values():
                    safe = safe and metadata_safe(value)

    artifact_checks: dict[str, Any] = {}
    for name, record in manifest["artifacts"].items():
        observed = digest(root / record["path"])
        artifact_checks[name] = {
            "expected": record["sha256"],
            "observed": observed,
            "pass": observed == record["sha256"],
        }
    gates = {
        "manifest_frozen": manifest.get("status") == "FROZEN_PRE_EXECUTION",
        "artifact_hashes_exact": all(row["pass"] for row in artifact_checks.values()),
        "metadata_status_valid": metadata.get("status") == "VALID_BACKWARD_RUNTIME_METADATA",
        "all_executor_gates_true": bool(metadata.get("gates")) and all(metadata["gates"].values()),
        "transition_valid": transition.get("valid") is True and transition.get("verdict") == "VALID",
        "candidate_identity_valid": transition.get("compiler", {}).get("candidate_identity_valid") is True,
        "scorer_anchor_exact": transition.get("anchors", {}).get("scorer_anchor_exact") is True,
        "family_set_and_counts_exact": observed_family == expected_family,
        "external_set_and_counts_exact": observed_external == expected_external,
        "call_indices_exact": indices_exact,
        "metadata_schema_excludes_tensor_values": safe,
        "embedded_status_exact": transition.get("backward_runtime_metadata_status")
        == metadata.get("status"),
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-runtime-metadata-audit.v0.1",
        "status": "VALID_BACKWARD_RUNTIME_METADATA_AUDIT" if all(gates.values()) else "INVALID_AUDIT",
        "gates": gates,
        "artifact_checks": artifact_checks,
        "counts": {
            "triton_families": len(observed_family),
            "triton_calls": sum(observed_family.values()),
            "external_calls": observed_external,
        },
        "claim_limits": metadata.get("claim_limits", []),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates, "counts": payload["counts"]}, indent=2, sort_keys=True))
    if payload["status"] != "VALID_BACKWARD_RUNTIME_METADATA_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
