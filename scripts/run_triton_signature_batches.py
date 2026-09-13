#!/usr/bin/env python3
"""Freeze and resume runtime-batched Triton signature measurements."""
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
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_PROCESS_GROUPS: set[int] = set()
_ACTIVE_PROCESS_GROUPS_LOCK = threading.Lock()


def _remember_process_group(pid: int) -> None:
    with _ACTIVE_PROCESS_GROUPS_LOCK:
        _ACTIVE_PROCESS_GROUPS.add(pid)


def _forget_process_group(pid: int) -> None:
    with _ACTIVE_PROCESS_GROUPS_LOCK:
        _ACTIVE_PROCESS_GROUPS.discard(pid)


def _terminate_all_process_groups() -> None:
    with _ACTIVE_PROCESS_GROUPS_LOCK:
        pids = tuple(_ACTIVE_PROCESS_GROUPS)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _task_status(group: dict[str, Any], task_id: str) -> str:
    run_id = hashlib.sha256(task_id.encode()).hexdigest()[:20]
    run = Path(group["campaign_output"]) / "runs" / run_id
    status = run / "status.json"
    if status.exists():
        return str(_read(status).get("status", "UNDECLARED"))
    return "INCOMPLETE_ATTEMPT" if run.exists() else "NOT_STARTED"


def group_status(group: dict[str, Any]) -> dict[str, Any]:
    output = Path(group["campaign_output"])
    timeout = output / "execution_timeout.json"
    task_statuses = {task: _task_status(group, task) for task in group["task_ids"]}
    if timeout.exists():
        execution = "TERMINAL_TIMEOUT"
    elif not (output / "protocol.json").exists():
        execution = "NOT_FROZEN"
    elif all(value == "VALID" for value in task_statuses.values()):
        execution = "VALID"
    elif any(value == "NOT_STARTED" for value in task_statuses.values()):
        execution = "PARTIAL_OR_NOT_STARTED"
    elif any(value == "INCOMPLETE_ATTEMPT" for value in task_statuses.values()):
        execution = "INCOMPLETE_ATTEMPT"
    else:
        execution = "TERMINAL_WITHOUT_VALID_MEASUREMENT"
    return {"execution_status": execution, "task_statuses": task_statuses}


def _command(group: dict[str, Any], action: str, device: str, batch_size: int) -> list[str]:
    base = [sys.executable, str(ROOT / "scripts/run_training_numerical_analysis.py"), "coverage"]
    if action == "freeze":
        command = [
            *base, "freeze", "--output", group["campaign_output"],
            "--release", group["release"], "--plans", group["case_plan"],
            "--architecture", group["runtime"]["architecture"],
            "--model", group["runtime"]["model"],
            "--input-bank", group["runtime"]["input_bank"],
            "--batch-size", str(batch_size),
        ]
        if group["runtime"].get("allow_graph_breaks"):
            command.append("--allow-graph-breaks")
        return command
    return [
        *base, "run", "--output", group["campaign_output"],
        "--device", device, "--limit", str(len(group["task_ids"])),
    ]


def _run_process(command: list[str], timeout_seconds: float | None) -> tuple[int | None, str, str, bool]:
    process = subprocess.Popen(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    _remember_process_group(process.pid)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout, stderr, False
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
        return None, stdout, stderr, True
    finally:
        _forget_process_group(process.pid)


def run_group(group: dict[str, Any], *, action: str, device: str, batch_size: int,
              timeout_seconds: float | None) -> dict[str, Any]:
    before = group_status(group)
    output = Path(group["campaign_output"])
    if action == "freeze" and before["execution_status"] != "NOT_FROZEN":
        return {"group_id": group["group_id"], "status": "ALREADY_FROZEN", "before": before}
    if action == "run" and before["execution_status"] in {
        "VALID", "TERMINAL_TIMEOUT", "TERMINAL_WITHOUT_VALID_MEASUREMENT", "INCOMPLETE_ATTEMPT"
    }:
        return {"group_id": group["group_id"], "status": "NOT_RERUN_TERMINAL_OR_INCOMPLETE", "before": before}
    command = _command(group, action, device, batch_size)
    returncode, stdout, stderr, timed_out = _run_process(command, timeout_seconds if action == "run" else None)
    if timed_out:
        record = {
            "schema": "triton-signature-runtime-batch-timeout-v1",
            "group_id": group["group_id"], "device": device,
            "timeout_seconds": timeout_seconds, "command": command,
            "completed_task_statuses_remain_valid": True,
            "incomplete_task_outputs_are_not_valid_measurements": True,
            "automatic_retry_or_replacement": False,
        }
        with (output / "execution_timeout.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return {
        "group_id": group["group_id"], "device": device, "command": command,
        "returncode": returncode, "timed_out": timed_out,
        "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:],
        "before": before, "after": group_status(group),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "status"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", action="append", default=[])
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--group-id", action="append", default=[],
                        help="Run only explicitly named frozen groups; selection is not based on outcomes")
    parser.add_argument("--timeout-seconds", type=float, default=14400.0)
    args = parser.parse_args()
    if args.action == "run" and not args.device:
        parser.error("run requires at least one --device")
    if args.limit_groups is not None and args.limit_groups < 1:
        parser.error("--limit-groups must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    manifest_path = args.manifest.resolve()
    payload = _read(manifest_path)
    groups = payload["groups"]
    if args.group_id:
        requested = set(args.group_id)
        known = {group["group_id"] for group in groups}
        missing = requested - known
        if missing:
            parser.error("Unknown --group-id: " + ", ".join(sorted(missing)))
        groups = [group for group in groups if group["group_id"] in requested]
    if args.limit_groups is not None:
        groups = groups[:args.limit_groups]
    rows: list[dict[str, Any]] = []
    if args.action == "status":
        rows = [{"group_id": group["group_id"], **group_status(group)} for group in groups]
    elif args.action == "freeze":
        rows = [run_group(group, action="freeze", device="NOT_USED",
                          batch_size=payload["batch_size"], timeout_seconds=None) for group in groups]
    else:
        runnable = [group for group in groups if group_status(group)["execution_status"] in {
            "PARTIAL_OR_NOT_STARTED",
        }]
        partitions = [runnable[offset::len(args.device)] for offset in range(len(args.device))]
        def run_partition(device: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [run_group(group, action="run", device=device,
                              batch_size=payload["batch_size"],
                              timeout_seconds=args.timeout_seconds) for group in selected]
        pool = ThreadPoolExecutor(max_workers=len(args.device))
        futures = [pool.submit(run_partition, device, selected)
                   for device, selected in zip(args.device, partitions) if selected]
        interrupted = False
        try:
            for future in as_completed(futures):
                rows.extend(future.result())
        except KeyboardInterrupt:
            _terminate_all_process_groups()
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            interrupted = True
        else:
            pool.shutdown()
        rows.sort(key=lambda row: row["group_id"])
    if args.action != "run":
        interrupted = False
    result = {
        "schema": "triton-signature-runtime-batch-execution-v1",
        "action": args.action, "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "rows": rows, "interrupted": interrupted,
    }
    record_dir = manifest_path.parent / "orchestration"
    record_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = record_dir / f"{args.action}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"record": str(path), "group_count": len(rows)}))
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
