#!/usr/bin/env python
"""Apply the frozen v0.1 scoring logic and relabel the v0.2 contract record."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-online", required=True)
    parser.add_argument("--b-online", required=True)
    parser.add_argument("--a-metadata", required=True)
    parser.add_argument("--b-metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = Path(__file__).with_name("evaluate_qwen3_grpo_training_control_confirmation.py")
    with tempfile.TemporaryDirectory(prefix="forkcert-qwen-grpo-v02-") as directory:
        intermediate = Path(directory) / "evaluation.json"
        command = [
            sys.executable, str(base),
            "--a-online", args.a_online,
            "--b-online", args.b_online,
            "--a-metadata", args.a_metadata,
            "--b-metadata", args.b_metadata,
            "--out", str(intermediate),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        payload = json.loads(intermediate.read_text())
    payload["schema_version"] = "forkcert.qwen3-grpo-training-control-confirmation.v0.2"
    payload["contract"] = str(
        Path(__file__).with_name(
            "QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_CONTRACT_V0_2_2026-07-17.md"
        ).resolve()
    )
    payload["scoring_logic"] = str(base.resolve())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
