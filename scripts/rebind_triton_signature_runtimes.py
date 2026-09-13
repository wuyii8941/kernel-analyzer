#!/usr/bin/env python3
"""Rebuild stale runtime releases used by frozen Triton signature groups.

The task inventory is copied only after ``rebind_runtime_release.py`` proves
that every frozen region/symbol identity still exists.  This is deliberately
weaker than validating copied reference graph hashes; run the separate runtime
preflight before freezing measurements.  Existing repaired releases may be
declared explicitly; no case or task is substituted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def plan(manifest: dict[str, Any], *, release_root: Path,
         existing_overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    existing_overrides = existing_overrides or {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in manifest.get("groups", []):
        old = str(Path(group["release"]).resolve())
        if old in seen:
            continue
        seen.add(old)
        new = existing_overrides.get(
            old, str((release_root / (Path(old).name + "_signature_r2")).resolve())
        )
        rows.append({
            "old_release": old,
            "new_release": new,
            "declared_existing_override": old in existing_overrides,
            "architecture": group["runtime"]["architecture"],
            "model": group["runtime"]["model"],
            "input_bank": group["runtime"]["input_bank"],
            "allow_graph_breaks": bool(group["runtime"].get("allow_graph_breaks")),
        })
    return rows


def _complete_release(path: Path) -> bool:
    return all((path / name).exists() for name in (
        "capture.json", "inventory.json.gz", "campaign.json.gz", "same_dtype_tasks.json.gz"
    ))


def run_one(row: dict[str, Any], device: str) -> dict[str, Any]:
    new = Path(row["new_release"])
    command = [
        sys.executable, str(ROOT / "scripts/rebind_runtime_release.py"),
        "--architecture", row["architecture"], "--model", row["model"],
        "--input-bank", row["input_bank"], "--old-release", row["old_release"],
        "--new-release", row["new_release"], "--device", device,
    ]
    if row["allow_graph_breaks"]:
        command.append("--allow-graph-breaks")
    if _complete_release(new):
        command.append("--reuse-existing")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        **row, "device": device, "command": command,
        "returncode": completed.returncode,
        "status": "VALIDATED" if completed.returncode == 0 and _complete_release(new) else "FAILED",
        "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--existing-override", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.release_root.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("--release-root must be under /data1/tzh")
    overrides: dict[str, str] = {}
    for item in args.existing_override:
        if "=" not in item:
            parser.error("--existing-override must be OLD=NEW")
        old, new = item.split("=", 1)
        old, new = str(Path(old).resolve()), str(Path(new).resolve())
        if old in overrides:
            parser.error("Duplicate --existing-override")
        overrides[old] = new
    manifest_path = args.manifest.resolve()
    rows = plan(json.loads(manifest_path.read_text()), release_root=args.release_root,
                existing_overrides=overrides)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    results: list[dict[str, Any]] = []
    if not args.dry_run:
        partitions = [rows[offset::len(args.device)] for offset in range(len(args.device))]
        def run_partition(device: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [run_one(row, device) for row in selected]
        with ThreadPoolExecutor(max_workers=len(args.device)) as pool:
            futures = [pool.submit(run_partition, device, selected)
                       for device, selected in zip(args.device, partitions) if selected]
            for future in as_completed(futures):
                results.extend(future.result())
        results.sort(key=lambda row: row["old_release"])
    result = {
        "schema": "triton-signature-runtime-rebind-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dry_run": args.dry_run,
        "planned": rows,
        "results": results,
        "all_validated": bool(results) and all(row["status"] == "VALIDATED" for row in results),
        "validation_scope": (
            "Region/symbol identity and required release files only; reference graph hashes "
            "require the separate execution preflight."
        ),
    }
    record_dir = manifest_path.parent / "runtime_rebinding"
    record_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target = record_dir / f"rebind-{stamp}.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"record": str(target), "planned": len(rows),
                      "validated": sum(row["status"] == "VALIDATED" for row in results)}))


if __name__ == "__main__":
    main()
