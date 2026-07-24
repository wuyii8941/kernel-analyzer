#!/usr/bin/env python
"""Resume every unfinished state in one frozen Qwen3 calibration trajectory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--skip-state-id", action="append", default=[])
    parser.add_argument("--max-states", type=int)
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    targets = [row for row in plan["targets"] if row["state_id"] not in set(args.skip_state_id)]
    if args.max_states is not None:
        targets = targets[: args.max_states]
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    ledger_path = results_root / "remaining_chain_ledger.json"
    ledger: dict[str, Any] = {
        "schema_version": "forkcert.qwen3-calibration-remaining-ledger.v0.1",
        "status": "RUNNING",
        "plan": str(plan_path),
        "states": [],
    }
    atomic_json(ledger_path, ledger)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    for target in targets:
        state_id = target["state_id"]
        command = [
            sys.executable,
            "theory_oracle/run_qwen3_calibration_state_chain_v0_1.py",
            "--plan", str(plan_path),
            "--state-id", state_id,
            "--results-root", str(results_root),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=False,
        )
        state_row = {
            "state_id": state_id,
            "optimizer_step": int(target["optimizer_step"]),
            "returncode": completed.returncode,
            "status": "COMPLETE_VALID" if completed.returncode == 0 else "FAILED_CLOSED",
        }
        ledger["states"].append(state_row)
        atomic_json(ledger_path, ledger)
        if completed.returncode != 0:
            ledger["status"] = "FAILED_CLOSED"
            ledger["failed_state_id"] = state_id
            atomic_json(ledger_path, ledger)
            raise SystemExit(2)
    ledger["status"] = "COMPLETE_VALID"
    atomic_json(ledger_path, ledger)
    print(json.dumps({"status": ledger["status"], "states": len(targets)}, indent=2))


if __name__ == "__main__":
    main()
