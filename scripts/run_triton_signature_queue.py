#!/usr/bin/env python3
"""Resume a frozen Triton-signature campaign manifest across GPU devices.

Every campaign keeps its own frozen protocol and authoritative status.  A
failure is recorded and the next independent signature may run; no failed
campaign is replaced, renamed, or silently retried.  Reinvocation skips valid
and terminal campaigns and continues only work that has never started.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def campaign_status(campaign: dict[str, Any]) -> str:
    output = Path(campaign["campaign_output"])
    timeout_record = output / "execution_timeout.json"
    if timeout_record.exists():
        return "TERMINAL_TIMEOUT"
    adapter = campaign.get("adapter")
    if adapter not in {None, "GENERIC_COVERAGE"}:
        completion = output / "completion_verification.json"
        if completion.exists():
            records = _read(completion).get("records", [])
            row = next(
                (item for item in records if str(item.get("task_id")) == str(campaign["task_id"])),
                None,
            )
            if row and row.get("status") in {"VERIFIED", "RECORDED_MEASUREMENT_CHECKED"}:
                return "VALID"
            return "TERMINAL_INVALID"
        if (output / "raw").exists() or (output / "spool").exists():
            return "INCOMPLETE_ATTEMPT"
        return "NOT_STARTED"
    run_id = hashlib.sha256(str(campaign["task_id"]).encode()).hexdigest()[:20]
    run = output / "runs" / run_id
    status = run / "status.json"
    if status.exists():
        value = str(_read(status).get("status", "UNDECLARED"))
        return "VALID" if value == "VALID" else "TERMINAL_" + value
    if run.exists():
        return "INCOMPLETE_ATTEMPT"
    return "NOT_STARTED"


def pending_indices(
    manifest: dict[str, Any], *, limit: int | None = None,
    adapters: set[str] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    pending: list[int] = []
    inventory: list[dict[str, Any]] = []
    for index, campaign in enumerate(manifest.get("campaigns", [])):
        status = campaign_status(campaign)
        inventory.append({
            "index": index,
            "case_id": campaign["case"]["case_id"],
            "operator_family": campaign["operator_family"],
            "task_id": campaign["task_id"],
            "status": status,
            "adapter": campaign.get("adapter", "GENERIC_COVERAGE"),
        })
        adapter = str(campaign.get("adapter", "GENERIC_COVERAGE"))
        selected_adapter = adapters is None or adapter in adapters
        if status == "NOT_STARTED" and selected_adapter and (limit is None or len(pending) < limit):
            pending.append(index)
    return pending, inventory


def run_one(
    manifest_path: Path,
    index: int,
    device: str,
    *,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_family_first_campaigns.py"),
        "run",
        "--manifest", str(manifest_path),
        "--start-index", str(index),
        "--limit-campaigns", "1",
        "--device", device,
    ]
    campaign = _read(manifest_path)["campaigns"][index]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        returncode: int | None = process.returncode
        stdout_tail = stdout[-4000:]
        stderr_tail = stderr[-4000:]
        timed_out = False
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        output = Path(campaign["campaign_output"])
        timeout_record = output / "execution_timeout.json"
        record = {
            "schema": "triton-signature-execution-timeout-v1",
            "task_id": campaign["task_id"],
            "case_id": campaign["case"]["case_id"],
            "device": device,
            "timeout_seconds": timeout_seconds,
            "command": command,
            "partial_output_is_not_a_valid_measurement": True,
            "automatic_retry_or_replacement": False,
        }
        timeout_record.parent.mkdir(parents=True, exist_ok=True)
        with timeout_record.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        returncode = None
        stdout_tail = stdout[-4000:]
        stderr_tail = stderr[-4000:]
        timed_out = True
    return {
        "index": index,
        "case_id": campaign["case"]["case_id"],
        "operator_family": campaign["operator_family"],
        "task_id": campaign["task_id"],
        "device": device,
        "returncode": returncode,
        "timed_out": timed_out,
        "status": campaign_status(campaign),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter", action="append", default=[],
                        help="Restrict execution to explicitly named adapters; does not inspect outcomes")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    manifest_path = args.manifest.resolve()
    manifest = _read(manifest_path)
    adapters = set(args.adapter) if args.adapter else None
    pending, before = pending_indices(manifest, limit=args.limit, adapters=adapters)
    snapshot_dir = args.manifest.parent / "signature_queue_orchestration"
    snapshot_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    if not args.dry_run and pending:
        # One sequential stream per device avoids simultaneous model loads on
        # the same GPU while still allowing independent GPUs to make progress.
        partitions = [pending[offset::len(args.device)] for offset in range(len(args.device))]

        def run_partition(device: str, indices: list[int]) -> list[dict[str, Any]]:
            return [
                run_one(
                    manifest_path,
                    index,
                    device,
                    timeout_seconds=args.timeout_seconds,
                )
                for index in indices
            ]

        with ThreadPoolExecutor(max_workers=len(args.device)) as pool:
            futures = [
                pool.submit(run_partition, device, indices)
                for device, indices in zip(args.device, partitions)
                if indices
            ]
            for future in as_completed(futures):
                rows.extend(future.result())
        rows.sort(key=lambda row: row["index"])
    after_pending, after = pending_indices(manifest, adapters=adapters)
    result = {
        "schema": "triton-signature-queue-execution-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "devices": args.device,
        "dry_run": args.dry_run,
        "timeout_seconds_per_campaign": args.timeout_seconds,
        "selected_indices": pending,
        "adapter_filter": sorted(adapters) if adapters is not None else None,
        "before": before,
        "executed": rows,
        "after": after,
        "remaining_not_started": len(after_pending),
        "failure_does_not_select_a_replacement": True,
        "numerical_outcomes_not_used_for_queue_order": True,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = snapshot_dir / ("queue-" + stamp + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "snapshot": str(path),
        "selected": len(pending),
        "executed": len(rows),
        "valid_after": sum(item["status"] == "VALID" for item in after),
        "terminal_or_incomplete_after": sum(
            item["status"] not in {"VALID", "NOT_STARTED"} for item in after
        ),
        "remaining_not_started": len(after_pending),
    }))


if __name__ == "__main__":
    main()
