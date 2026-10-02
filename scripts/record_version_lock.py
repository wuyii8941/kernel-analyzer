"""Record the version lock required by stage summary section 5.

Writes the Triton wheel version, a hash of the compiled Triton library (the
wheel has no git commit), ptxas version, target GPU, default CUDA compile
options and the IR extraction stage.  Run in the mainline environment:

    /data1/tzh/envs/ka_main/bin/python scripts/record_version_lock.py
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "reference_eval" / "version_lock.json"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> dict:
    import torch
    import triton
    from triton.backends.nvidia import compiler as nvidia

    triton_dir = os.path.dirname(triton.__file__)
    libraries = sorted(glob.glob(os.path.join(triton_dir, "_C", "libtriton*.so")))
    options = {
        field.name: (field.default if field.default is not dataclasses.MISSING else None)
        for field in dataclasses.fields(nvidia.CUDAOptions)
    }
    json_safe = {k: v if isinstance(v, (bool, int, float, str, type(None))) else repr(v) for k, v in options.items()}
    device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    lock = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "triton_commit": None,
        "triton_commit_note": "pip wheel without git metadata; identified by version and library hash",
        "libtriton_sha256": {os.path.basename(p): _sha256(p) for p in libraries},
        "ptxas_version": str(nvidia.get_ptxas_version()) if hasattr(nvidia, "get_ptxas_version") else None,
        "gpu": None if device is None else {
            "name": device.name,
            "capability": f"sm_{device.major}{device.minor}",
            "count": torch.cuda.device_count(),
        },
        "cuda_options_defaults": json_safe,
        "ir_extraction_stage": "asm['ttir'] of the compiled kernel (after make_ttir)",
        "reference_mode_default": "numerical_difference",
    }
    try:
        import gmpy2
        import flint

        lock["gmpy2"] = gmpy2.version()
        lock["mpfr"] = gmpy2.mpfr_version()
        lock["python_flint"] = flint.__version__
    except ImportError as exc:  # pragma: no cover - recorded, not fatal
        lock["reference_libraries_error"] = str(exc)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    lock = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
