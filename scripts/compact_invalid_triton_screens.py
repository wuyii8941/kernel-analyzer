#!/usr/bin/env python3
"""Manifest and finalize removal of raw screens invalidated by the ABI audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
MANIFEST = COVERAGE / "invalid_triton_raw_manifest.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def prepare() -> None:
    abi_path = COVERAGE / "triton_reference_abi_audit.json"
    abi = json.loads(abi_path.read_text())
    if abi["status"] != "INVALID_REFERENCE_ABI":
        raise RuntimeError("Triton ABI invalidation is not established")
    files = []
    for screen in sorted((COVERAGE / "runtime_releases").glob("*_r1/triton_screen.json.gz")):
        oracle = screen.parent / "triton_oracle.json.gz"
        if not oracle.is_file():
            raise RuntimeError(f"missing retained compact Oracle: {oracle}")
        files.append({
            "path": str(screen.relative_to(ROOT)),
            "bytes": screen.stat().st_size,
            "sha256": sha256(screen),
            "state": "PRESENT_INVALID_ABI_READY_FOR_DELETION",
            "retained_oracle": str(oracle.relative_to(ROOT)),
            "retained_oracle_sha256": sha256(oracle),
        })
    if len(files) != 12:
        raise RuntimeError(f"expected 12 raw screens, found {len(files)}")
    payload = {
        "schema": "kernel-analyzer-invalid-triton-raw-manifest-v1",
        "status": "READY_FOR_EXACT_DELETION",
        "abi_audit": str(abi_path.relative_to(ROOT)),
        "abi_audit_result_sha256": abi["result_sha256"],
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
        "disposition": (
            "Invalid pointer-ABI rows cannot support verdicts. Compact Oracles, execution "
            "counts, file hashes and the ABI counterexample remain."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": 12, "total_bytes": payload["total_bytes"]}, sort_keys=True))


def finalize() -> None:
    payload = json.loads(MANIFEST.read_text())
    for row in payload["files"]:
        if (ROOT / row["path"]).exists():
            raise RuntimeError(f"raw invalid screen still exists: {row['path']}")
        oracle = ROOT / row["retained_oracle"]
        if not oracle.is_file() or sha256(oracle) != row["retained_oracle_sha256"]:
            raise RuntimeError(f"retained Oracle missing or changed: {oracle}")
        row["state"] = "DELETED_INVALID_ABI_REGENERABLE"
    payload["status"] = "COMPACTED_INVALID_RAW_REMOVED"
    payload.pop("result_sha256", None)
    payload["result_sha256"] = canonical(payload)
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "removed_bytes": payload["total_bytes"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "finalize"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else finalize()


if __name__ == "__main__":
    main()
