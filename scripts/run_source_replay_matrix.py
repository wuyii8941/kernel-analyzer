#!/usr/bin/env python3
"""Run the six source-replay cells sequentially on one GPU.

The schedule is intentionally checkpoint-expanded, while
``replay_source_bank.py`` handles the eight checkpoints in one cell.  This
orchestrator keeps cells sequential to bound compiler/GPU memory and never
rewrites old artifacts unless ``--overwrite`` is explicit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=FINAL / "source_replay_schedule.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="3")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = json.loads(args.schedule.read_text())
    cells: dict[tuple[str, bool, int], dict] = {}
    for row in schedule["rows"]:
        key = (str(row["dtype"]), bool(row["tf32"]), int(row["seq_len"]))
        cells.setdefault(key, row)
    commands = []
    for (dtype, tf32, seq_len), row in sorted(cells.items(), key=lambda item: item[0]):
        command = [
            args.python,
            str(ROOT / "scripts" / "replay_source_bank.py"),
            "--seq-len", str(seq_len),
            "--mapping", str(FINAL / str(row["mapping_file"])),
            "--output-dir", str(args.output_dir),
            "--cache-dir", str(args.cache_dir),
            "--python", args.python,
            "--device", str(args.device),
        ]
        if tf32:
            command.append("--tf32")
        if args.overwrite:
            command.append("--overwrite")
        commands.append({"dtype": dtype, "tf32": tf32, "seq_len": seq_len, "command": command})

    if args.dry_run:
        print(json.dumps({"cell_count": len(commands), "commands": commands}, indent=2, sort_keys=True))
        return

    for item in commands:
        print(json.dumps({"status": "RUNNING", "dtype": item["dtype"], "tf32": item["tf32"], "seq_len": item["seq_len"]}), flush=True)
        subprocess.run(item["command"], cwd=ROOT, check=True)
        print(json.dumps({"status": "OK", "dtype": item["dtype"], "tf32": item["tf32"], "seq_len": item["seq_len"]}), flush=True)


if __name__ == "__main__":
    main()
