from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EnvAudit:
    python: str
    platform: str
    cwd: str
    torch: dict[str, Any]
    packages: dict[str, str | None]
    nvidia_smi: dict[str, Any]
    deterministic_env: dict[str, str | None]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {"returncode": None, "stdout": exc.stdout, "stderr": exc.stderr, "timeout": timeout}
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def torch_info() -> dict[str, Any]:
    try:
        import torch

        info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        if torch.cuda.is_available():
            info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return info
    except Exception as exc:
        return {"import_error": repr(exc)}


def audit_environment() -> EnvAudit:
    packages = {
        name: _package_version(name)
        for name in ["torch", "transformers", "vllm", "trl", "verl", "flash-attn", "datasets", "accelerate"]
    }
    deterministic_env = {
        key: os.environ.get(key)
        for key in [
            "CUBLAS_WORKSPACE_CONFIG",
            "PYTHONHASHSEED",
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
        ]
    }
    return EnvAudit(
        python=sys.executable,
        platform=platform.platform(),
        cwd=os.getcwd(),
        torch=torch_info(),
        packages=packages,
        nvidia_smi=_run(["nvidia-smi"], timeout=20),
        deterministic_env=deterministic_env,
    )
