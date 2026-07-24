#!/usr/bin/env python
"""Partition live Qwen3 backward calls by exact non-value argument signature."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_value(value: dict[str, Any]) -> dict[str, Any]:
    if value["kind"] == "tensor":
        return {
            "kind": "tensor",
            "shape": value["shape"],
            "stride": value["stride"],
            "dtype": value["dtype"],
            "device": value["device"],
            "requires_grad": value["requires_grad"],
        }
    return value


def signature(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {
        "positional": [canonical_value(value) for value in record["positional"]],
        "keyword": {
            key: canonical_value(value) for key, value in sorted(record.get("keyword", {}).items())
        },
        "keyword_types": record.get("keyword_types", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def cluster_calls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest, payload = signature(row)
        cluster = clusters.setdefault(
            digest,
            {"signature_sha256": digest, "signature": payload, "call_indices": []},
        )
        cluster["call_indices"].append(int(row["index"]))
    result = sorted(clusters.values(), key=lambda row: row["call_indices"][0])
    for cluster in result:
        indices = cluster["call_indices"]
        cluster["calls"] = len(indices)
        cluster["candidate_representatives"] = sorted(
            set([indices[0], indices[len(indices) // 2], indices[-1]])
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--metadata-audit", required=True)
    parser.add_argument("--static-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text())
    audit = json.loads(Path(args.metadata_audit).read_text())
    static = json.loads(Path(args.static_summary).read_text())
    if metadata["status"] != "VALID_BACKWARD_RUNTIME_METADATA":
        raise ValueError("runtime metadata is invalid")
    if audit["status"] != "VALID_BACKWARD_RUNTIME_METADATA_AUDIT":
        raise ValueError("runtime metadata audit is invalid")

    static_rows = {row["name"]: row for row in static["kernel_families"]}
    families: list[dict[str, Any]] = []
    for family, calls in metadata["family_calls"].items():
        clusters = cluster_calls(calls)
        static_calls = int(static_rows[family]["call_count"])
        conservative_multi_role = static_calls not in (1, 27, 28)
        families.append(
            {
                "treatment_id": family,
                "kind": "generated_triton_family",
                "runtime_calls": len(calls),
                "signature_clusters": clusters,
                "signature_cluster_count": len(clusters),
                "semantic_role_status": (
                    "UNRESOLVED_MULTI_ROLE_DESPITE_SIGNATURES"
                    if conservative_multi_role
                    else "CANDIDATE_SINGLE_ROLE_UNVALIDATED"
                ),
                "constituent_aten": static_rows[family]["original_aten"],
            }
        )
    for operation, calls in metadata["external_calls"].items():
        clusters = cluster_calls(calls)
        families.append(
            {
                "treatment_id": f"extern:{operation}",
                "kind": "external_kernel_family",
                "runtime_calls": len(calls),
                "signature_clusters": clusters,
                "signature_cluster_count": len(clusters),
                "semantic_role_status": "UNRESOLVED_MULTI_ROLE_DESPITE_SIGNATURES",
                "constituent_aten": [f"aten.{operation}"],
            }
        )
    families.sort(key=lambda row: (row["kind"], row["treatment_id"]))
    payload = {
        "schema_version": "forkcert.qwen3-backward-signature-clusters.v0.1",
        "status": "VALID_SIGNATURE_PARTITION_NOT_SEMANTIC_EQUIVALENCE",
        "metrics": {
            "families": len(families),
            "calls": sum(row["runtime_calls"] for row in families),
            "signature_clusters": sum(row["signature_cluster_count"] for row in families),
            "families_with_multiple_signatures": sum(
                row["signature_cluster_count"] > 1 for row in families
            ),
            "semantic_multi_role_families_unresolved": sum(
                row["semantic_role_status"] == "UNRESOLVED_MULTI_ROLE_DESPITE_SIGNATURES"
                for row in families
            ),
        },
        "families": families,
        "claim_limits": [
            "exact argument-signature strata, not semantic-role equivalence classes",
            "storage offsets and tensor values excluded from signatures",
            "same signature can still contain different semantic roles",
            "one selected state and shape only",
            "no causal or population coverage credit",
        ],
    }
    if len(families) != 41 or payload["metrics"]["calls"] != 1857:
        raise ValueError("signature partition denominator mismatch")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
