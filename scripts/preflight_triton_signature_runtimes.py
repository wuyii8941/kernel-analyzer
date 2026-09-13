#!/usr/bin/env python3
"""Validate and, when proved safe, refresh reference graph hashes.

Every frozen task in a runtime group is exercised together using the existing
one-state engineering mode. A cut hash is refreshed only after the production
reference-cut validator reports an exact port match and the sole mismatch is
the expected/actual graph code hash. Testing only one task per graph is not
sufficient: different saved cuts can carry independently stale hashes. No
numerical outcome is used and no scientific measurement is emitted here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MISMATCH = re.compile(
    r"reference cut AOT graph hash mismatch after exact port match: "
    r"(same-dtype:(forward|backward):graph(\d+):[^ ]+) "
    r"expected=([0-9a-f]{64}) actual=([0-9a-f]{64})"
)


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def preflight_cases(group: dict[str, Any]) -> list[dict[str, Any]]:
    release = Path(group["release"])
    plan = _load(release / "same_dtype_tasks.json.gz")
    tasks = {str(row["task_id"]): row for row in plan["rows"]}
    cuts = {
        str(row["task_id"]).removeprefix("same-dtype:"): row
        for row in plan["reference_cut_tasks"]
    }
    selected: list[dict[str, Any]] = []
    for case in group["cases"]:
        task = tasks[str(case["task_id"])]
        endpoint = str(task.get("exact_aot_endpoint_id"))
        cut = cuts.get(endpoint)
        if cut is None:
            raise ValueError("Selected AOT_REPLAY task has no reference cut: " + str(case["task_id"]))
        match = re.match(r"same-dtype:(forward|backward):graph(\d+):", str(cut["task_id"]))
        if match is None:
            raise ValueError("Reference cut has no runtime graph identity: " + str(cut["task_id"]))
        selected.append(case)
    return selected


def run_group(group: dict[str, Any], *, root: Path, device: str,
              max_refreshes: int) -> dict[str, Any]:
    group_root = root / group["group_id"]
    group_root.mkdir(parents=True, exist_ok=False)
    cases = preflight_cases(group)
    attempts: list[dict[str, Any]] = []
    refreshes: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for attempt_index in range(max_refreshes + 1):
        attempt = group_root / f"attempt-{attempt_index:02d}"
        attempt.mkdir()
        case_plan = attempt / "case_plan.json"
        case_plan.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
        command = [
            sys.executable, str(ROOT / "scripts/capture_bound_endpoint_bias_formation_v21.py"),
            "--architecture", group["runtime"]["architecture"],
            "--model", group["runtime"]["model"],
            "--input-bank", group["runtime"]["input_bank"],
            "--release-dir", group["release"], "--case-plan", str(case_plan),
            "--output-dir", str(attempt / "legacy"), "--spool-dir", str(attempt / "spool"),
            "--states", "1", "--device", device, "--engineering-reach-only",
        ]
        if group["runtime"].get("allow_graph_breaks"):
            command.append("--allow-graph-breaks")
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=f"{ROOT}/src:{ROOT}",
            HF_HOME="/data1/tzh/cache/huggingface",
            XDG_CACHE_HOME="/data1/tzh/cache/xdg",
            TRITON_CACHE_DIR="/data1/tzh/cache/triton",
            TORCHINDUCTOR_CACHE_DIR="/data1/tzh/cache/torchinductor",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        log = attempt / "preflight.log"
        with log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT)
        text = log.read_text(errors="replace")
        row = {"attempt": attempt_index, "returncode": completed.returncode,
               "command": command, "log": str(log)}
        attempts.append(row)
        if completed.returncode == 0:
            return {
                "group_id": group["group_id"], "release": group["release"],
                "device": device, "status": "VALID", "preflight_cases": cases,
                "attempts": attempts, "refreshes": refreshes,
                "scientific_measurement_emitted": False,
            }
        matches = list(MISMATCH.finditer(text))
        if not matches:
            return {
                "group_id": group["group_id"], "release": group["release"],
                "device": device, "status": "FAILED_NON_HASH_PREFLIGHT",
                "preflight_cases": cases, "attempts": attempts,
                "refreshes": refreshes, "scientific_measurement_emitted": False,
            }
        cut_id, phase_token, graph_token, old_hash, new_hash = matches[-1].groups()
        identity = (cut_id, old_hash, new_hash)
        if identity in seen:
            return {
                "group_id": group["group_id"], "release": group["release"],
                "device": device, "status": "FAILED_REPEATED_HASH_MISMATCH",
                "preflight_cases": cases, "attempts": attempts,
                "refreshes": refreshes, "scientific_measurement_emitted": False,
            }
        seen.add(identity)
        refresh_command = [
            sys.executable, str(ROOT / "scripts/refresh_reference_graph_hashes.py"),
            "--release", group["release"], "--phase", phase_token.upper(),
            "--cut-id", cut_id, "--old-hash", old_hash,
            "--new-hash", new_hash,
        ]
        refreshed = subprocess.run(refresh_command, cwd=ROOT, text=True, capture_output=True)
        refreshes.append({
            "phase": phase_token.upper(), "graph_index_from_runtime_cut_id": int(graph_token),
            "cut_id": cut_id,
            "old_hash": old_hash, "new_hash": new_hash,
            "returncode": refreshed.returncode,
            "stdout": refreshed.stdout[-2000:], "stderr": refreshed.stderr[-2000:],
        })
        if refreshed.returncode != 0:
            return {
                "group_id": group["group_id"], "release": group["release"],
                "device": device, "status": "FAILED_HASH_REFRESH",
                "preflight_cases": cases, "attempts": attempts,
                "refreshes": refreshes, "scientific_measurement_emitted": False,
            }
    return {
        "group_id": group["group_id"], "release": group["release"],
        "device": device, "status": "FAILED_REFRESH_LIMIT",
        "preflight_cases": cases, "attempts": attempts,
        "refreshes": refreshes, "scientific_measurement_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--max-refreshes", type=int, default=8)
    parser.add_argument("--group-id", action="append", default=[])
    args = parser.parse_args()
    if args.output_root.exists() or not args.output_root.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output root under /data1/tzh")
    if args.max_refreshes < 0:
        parser.error("--max-refreshes must be nonnegative")
    manifest_path = args.manifest.resolve()
    manifest = _load(manifest_path)
    groups = manifest["groups"]
    if args.group_id:
        requested = set(args.group_id)
        groups = [group for group in groups if group["group_id"] in requested]
        if len(groups) != len(requested):
            parser.error("One or more --group-id values are absent")
    args.output_root.mkdir(parents=True)
    partitions = [groups[offset::len(args.device)] for offset in range(len(args.device))]
    results: list[dict[str, Any]] = []
    def run_partition(device: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [run_group(group, root=args.output_root, device=device,
                          max_refreshes=args.max_refreshes) for group in selected]
    with ThreadPoolExecutor(max_workers=len(args.device)) as pool:
        futures = [pool.submit(run_partition, device, selected)
                   for device, selected in zip(args.device, partitions) if selected]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: row["group_id"])
    report = {
        "schema": "triton-signature-runtime-preflight-v2",
        "manifest": str(manifest_path),
        "manifest_sha256_before_preflight": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "status": "PASS" if results and all(row["status"] == "VALID" for row in results) else "INCOMPLETE",
        "results": results,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Execution/reference validation only; no bias or training outcome is measured.",
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"status": report["status"], "groups": len(results),
                      "valid": sum(row["status"] == "VALID" for row in results)}))


if __name__ == "__main__":
    main()
