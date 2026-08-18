#!/usr/bin/env python3
"""Run the frozen live-contrast cells sequentially with bounded resources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text()).get("status") == "COMPLETE_LIVE_FULL_COORDINATE_CONTRASTS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "results/coverage/live_contrast_plan.json")
    parser.add_argument("--device-index", type=int, default=3)
    parser.add_argument("--log-root", type=Path, default=Path("/data1/tzh/cache/kernel_analyzer_logs"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.log_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.device_index),
        "HF_HOME": "/data1/tzh/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
        "XDG_CACHE_HOME": "/data1/tzh/cache/xdg",
    })
    for index, cell in enumerate(plan["execution_order"]):
        output = Path(cell["output"])
        if complete(output):
            print(json.dumps({"event": "SKIP_COMPLETE", "output": str(output)}), flush=True)
            continue
        cache = Path(cell["torchinductor_cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
        log = args.log_root / (output.stem + ".log")
        command = list(cell["argv"]) + ["--device", "cuda:0"]
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "START", "argv": command}) + "\n")
            handle.flush()
            print(json.dumps({"event": "CELL_START", "index": index,
                              "output": str(output), "log": str(log)}), flush=True)
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode or not complete(output):
            raise RuntimeError("cell failed closed; inspect " + str(log))
        print(json.dumps({"event": "CELL_COMPLETE", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
