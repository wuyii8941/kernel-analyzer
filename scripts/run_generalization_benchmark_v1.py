#!/usr/bin/env python3
"""Run frozen generalization groups without changing their case selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {"phi4": "phi"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spool-root", type=Path, required=True)
    parser.add_argument(
        "--status-root",
        type=Path,
        help="Failure records; defaults to OUTPUT_ROOT/status.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected = set(args.models)
    groups = json.loads(args.index.read_text())["groups"]
    rows = [row for row in groups if row["model"] in selected]
    if not rows:
        raise SystemExit("no frozen execution groups matched --models")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    status_root = args.status_root or (args.output_root / "status")
    status_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        group = f"{row['model']}_seq{row['sequence_length']}"
        expected = [
            args.output_root / "raw" / f"{case['case_id']}.json"
            for case in json.loads(Path(row["case_plan"]).read_text())["cases"]
        ]
        if args.skip_existing and expected and all(path.exists() for path in expected):
            print(json.dumps({"event": "GROUP_SKIPPED_COMPLETE", "group": group}), flush=True)
            continue
        command = [
            str(args.python),
            "scripts/capture_bound_endpoint_bias_formation_v21.py",
            "--architecture", ARCHITECTURES.get(row["model"], row["model"]),
            "--model", row["model_path"],
            "--input-bank", row["input_bank"],
            "--release-dir", row["runtime_release"],
            "--case-plan", row["case_plan"],
            "--output-dir", str(args.output_root / "legacy" / group),
            "--spool-dir", str(args.spool_root / f"{group}_spool"),
            "--states", "32",
            "--training-bias-profile-v2-output-dir", str(args.output_root / "raw"),
        ]
        if row["model"] in {"phi4", "mamba"}:
            command.append("--allow-graph-breaks")
        print(json.dumps({"event": "GROUP_STARTED", "group": group}), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        except subprocess.CalledProcessError as error:
            for case, target in zip(
                json.loads(Path(row["case_plan"]).read_text())["cases"], expected
            ):
                if target.exists():
                    continue
                status_path = status_root / f"{case['case_id']}.json"
                status_path.write_text(json.dumps({
                    "schema": "kernel-analyzer-generalization-execution-status-v1",
                    "status": "ABSTAIN_RUNTIME_EXECUTION_FAILED",
                    "case_id": case["case_id"],
                    "execution_group": group,
                    "return_code": error.returncode,
                    "reason": (
                        "The original frozen execution group failed. The case remains "
                        "in the benchmark and was not replaced."
                    ),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }, indent=2, sort_keys=True) + "\n")
            print(json.dumps({
                "event": "GROUP_ABSTAINED_RUNTIME_FAILURE",
                "group": group,
                "return_code": error.returncode,
            }), flush=True)
            continue
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise RuntimeError(f"group {group} finished without expected results: {missing}")
        print(json.dumps({"event": "GROUP_COMPLETE", "group": group}), flush=True)


if __name__ == "__main__":
    main()
