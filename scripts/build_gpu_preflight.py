#!/usr/bin/env python3
"""Record the host CUDA preflight without attempting a GPU workaround."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def command_status(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("sandbox", "external"), default="sandbox")
    parser.add_argument("--output", type=Path, default=FINAL / "gpu_preflight.json")
    args = parser.parse_args()
    nodes = sorted(glob.glob("/dev/nvidia*"))
    proc_path = Path("/proc/driver/nvidia/version")
    proc_version = proc_path.read_text().strip() if proc_path.exists() else None
    module_code, module_text = command_status(["/bin/bash", "-lc", "lsmod 2>/dev/null | awk '$1 ~ /^nvidia/ {print $1}'"])
    modules = sorted(set(module_text.split())) if module_code == 0 else []
    smi_code, smi_text = command_status(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"])
    torch_info = {}
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover - environment diagnostic
        torch_info = {"import_error": repr(exc)}
    externally_ready = smi_code == 0 and bool(torch_info.get("cuda_available"))
    ready = externally_ready if args.view == "external" else bool(nodes and externally_ready)
    output = {
        "schema": "kernel-analyzer-gpu-preflight-v1",
        "view": args.view,
        "candidate_values_used_to_select_or_classify": False,
        "device_nodes": nodes,
        "device_nodes_present": bool(nodes),
        "nvidia_proc_version_present": proc_version is not None,
        "nvidia_proc_version": proc_version,
        "loaded_nvidia_modules": modules,
        "nvidia_smi_exit_code": smi_code,
        "nvidia_smi_output": smi_text,
        "torch": torch_info,
        "status": "READY" if ready else "GPU_DEVICE_NODE_UNAVAILABLE",
        "replay_authorized": ready,
        "boundary": "This preflight records device visibility only. It never creates device nodes, changes drivers, or substitutes CPU execution for a CUDA replay.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = args.output.resolve()
    if ROOT not in (path, *path.parents):
        raise ValueError(f"output must stay under {ROOT}: {path}")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "status": output["status"], "replay_authorized": output["replay_authorized"]}, sort_keys=True))


if __name__ == "__main__":
    main()
