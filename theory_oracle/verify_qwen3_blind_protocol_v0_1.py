#!/usr/bin/env python
"""Verify that an opaque Qwen3 case does not leak historical bug metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FORBIDDEN_MANIFEST_KEYS = {"issue", "upstream_issue", "upstream_fix", "fixed_commit", "buggy_commit", "patch", "root_cause"}
FORBIDDEN_PATH_PARTS = {".git", "reference_run", "fixed", "buggy", "patch"}


def walk_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(str(key).lower() for key in value)
        for child in value.values():
            found.update(walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(walk_keys(child))
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(case_dir: Path, blind_report: Path | None = None) -> dict[str, Any]:
    manifest_path = case_dir / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    key_leaks = sorted(FORBIDDEN_MANIFEST_KEYS.intersection(walk_keys(manifest)))
    path_leaks = []
    files = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file()):
        relative = str(path.relative_to(case_dir))
        files.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
        if any(part.lower() in FORBIDDEN_PATH_PARTS for part in path.parts):
            path_leaks.append(relative)
    exclusion_text = " ".join(str(item).lower() for item in manifest.get("locator_exclusions", []))
    required_exclusions = ["issue", "fixed", "patch", "root-cause"]
    exclusions_complete = all(item in exclusion_text for item in required_exclusions)
    report = {
        "schema_version": "forkcert.qwen3-blind-protocol-audit.v0.1",
        "case_dir": str(case_dir.resolve()),
        "case_id": manifest.get("case_id"),
        "visibility": manifest.get("visibility"),
        "manifest_key_leaks": key_leaks,
        "path_leaks": path_leaks,
        "required_exclusions_present": exclusions_complete,
        "artifact_count": len(files),
        "artifacts": files,
        "blind_report_present": blind_report is not None and blind_report.is_file(),
        "status": "VALID_PATCH_FREE_OPAQUE_CASE"
        if manifest.get("visibility") == "patch_free_opaque_case" and not key_leaks and not path_leaks and exclusions_complete
        else "INVALID_OR_LEAKING_CASE",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--blind-report", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.case_dir.resolve(), args.blind_report.resolve() if args.blind_report else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "VALID_PATCH_FREE_OPAQUE_CASE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
