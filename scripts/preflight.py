#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def module_import_status(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def model_ids_from_config(cfg: dict) -> list[str]:
    ids: list[str] = []
    for key in ["model", "policy", "path_ref", "path_alt"]:
        value = cfg.get(key)
        if isinstance(value, dict) and value.get("model_name_or_path"):
            ids.append(str(value["model_name_or_path"]))
    if cfg.get("model_name_or_path"):
        ids.append(str(cfg["model_name_or_path"]))
    return sorted(set(ids))


def precision_requirements(configs: list[dict]) -> dict[str, bool]:
    dtypes = set()
    attention_backends = set()
    for cfg in configs:
        for key in ["model", "policy", "path_ref", "path_alt"]:
            item = cfg.get(key)
            if not isinstance(item, dict):
                continue
            if item.get("dtype"):
                dtypes.add(str(item["dtype"]).lower())
            if item.get("attention_backend"):
                attention_backends.add(str(item["attention_backend"]).lower())
    return {
        "bf16_requested": bool(dtypes & {"bf16", "bfloat16"}),
        "flash_attention_requested": bool(attention_backends & {"flash", "flash_attention"}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast preflight checks for ForkCert runner.")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--samples", default="data/prompt_pairs.jsonl")
    parser.add_argument("--require-ml", action="store_true", help="Fail if torch/transformers/numpy are unavailable.")
    parser.add_argument("--require-rl", action="store_true", help="Fail if TRL and datasets are unavailable.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--out", default="results/preflight.json")
    args = parser.parse_args()

    checks: dict[str, object] = {
        "python": sys.executable,
        "cwd": os.getcwd(),
        "modules": {name: module_available(name) for name in ["torch", "transformers", "trl", "datasets", "yaml", "numpy"]},
        "module_imports": {
            name: module_import_status(name)
            for name in ["torch", "transformers", "fsspec", "datasets", "trl", "yaml", "numpy"]
        },
        "env": {
            key: os.environ.get(key)
            for key in [
                "HF_HOME",
                "HF_HUB_CACHE",
                "TRANSFORMERS_CACHE",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "XDG_CACHE_HOME",
                "PIP_CACHE_DIR",
                "MPLCONFIGDIR",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
            ]
        },
        "configs": {},
        "samples": {},
        "cuda": {},
    }

    errors: list[str] = []
    loaded_configs = []
    for cfg in args.config:
        try:
            loaded = load_config(cfg)
            loaded_configs.append(loaded)
            checks["configs"][cfg] = {"ok": True, "keys": sorted(loaded.keys()), "model_ids": model_ids_from_config(loaded)}
        except Exception as exc:
            checks["configs"][cfg] = {"ok": False, "error": repr(exc)}
            errors.append(f"config failed: {cfg}: {exc!r}")

    try:
        rows = read_jsonl(args.samples)
        checks["samples"] = {"ok": True, "path": args.samples, "rows": len(rows)}
        if not rows:
            errors.append(f"samples file is empty: {args.samples}")
        for field in ["case_id", "prompt", "response"]:
            if rows and field not in rows[0]:
                errors.append(f"samples missing required field {field}: {args.samples}")
    except Exception as exc:
        checks["samples"] = {"ok": False, "path": args.samples, "error": repr(exc)}
        errors.append(f"samples failed: {args.samples}: {exc!r}")

    if args.require_ml:
        modules = checks["module_imports"]
        for name in ["torch", "transformers", "numpy"]:
            if not modules[name]["ok"]:
                errors.append(f"required Python module cannot be imported: {name}: {modules[name].get('error')}")
    if args.require_rl:
        modules = checks["module_imports"]
        for name in ["fsspec", "datasets", "trl"]:
            if not modules[name]["ok"]:
                errors.append(f"required RL Python module cannot be imported: {name}: {modules[name].get('error')}")

    if args.require_cuda:
        try:
            import torch

            available = torch.cuda.is_available()
            count = torch.cuda.device_count()
            devices = []
            for index in range(count):
                capability = torch.cuda.get_device_capability(index)
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(capability),
                        "free_bytes": free_bytes,
                        "total_bytes": total_bytes,
                        "bf16_hardware": capability[0] >= 8,
                        "flash_sdp_hardware": capability[0] >= 8,
                    }
                )
            requirements = precision_requirements(loaded_configs)
            checks["cuda"] = {
                "available": available,
                "device_count": count,
                "devices": devices,
                "selected_device": devices[0] if devices else None,
                "precision_requirements": requirements,
            }
            if available and count:
                checks["cuda"]["device_name_0"] = torch.cuda.get_device_name(0)
                if requirements["bf16_requested"] and not devices[0]["bf16_hardware"]:
                    errors.append(
                        "configs request BF16 but selected CUDA device lacks BF16 hardware support; "
                        "select an Ampere-or-newer GPU or resolve the experiment configs to FP16"
                    )
                if requirements["flash_attention_requested"] and not devices[0]["flash_sdp_hardware"]:
                    errors.append(
                        "configs request FlashAttention SDP but selected CUDA device is pre-Ampere; "
                        "select an Ampere-or-newer GPU or use the SDPA efficient backend"
                    )
            else:
                errors.append("CUDA required but torch reports no CUDA device")
        except Exception as exc:
            checks["cuda"] = {"error": repr(exc)}
            errors.append(f"CUDA check failed: {exc!r}")

    checks["ok"] = not errors
    checks["errors"] = errors
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
