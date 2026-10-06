#!/usr/bin/env python3
"""Build .cache/pylibs/unsloth_shim: Unsloth's Triton kernel modules (unsloth/kernels at the pinned commit), importable
without the rest of Unsloth, unsloth_zoo, bitsandbytes or transformers.

The kernel files and ``kernels/utils.py`` (launch settings: ``calculate_settings``, ``triton_tanh``, ...) are copied
verbatim; only infrastructure that they import is stubbed (device type, bitsandbytes availability, unsloth_zoo
version / logging / patch hooks, the transformers logger and the ``LlamaRMSNorm`` base class that
``rms_layernorm.py`` subclasses at import).  No kernel body or launch configuration is replaced.

    python scripts/build_unsloth_shim.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".cache/src/unsloth-main/unsloth"
DST = ROOT / ".cache/pylibs/unsloth_shim"

FILES = ["kernels/utils.py", "kernels/fp8.py", "kernels/nvfp4.py", "kernels/int4_packed.py",
         "kernels/rms_layernorm.py", "kernels/layernorm.py", "kernels/swiglu.py", "kernels/geglu.py",
         "kernels/cross_entropy_loss.py", "kernels/rope_embedding.py"]

STUBS = {
    "unsloth/__init__.py": "",
    "unsloth/kernels/__init__.py": "",
    "unsloth/import_fixes.py": (
        "import contextlib\n\n\n@contextlib.contextmanager\ndef suppress_cuda_printf(*a, **k):\n    yield\n"),
    "unsloth/device_type.py": (
        "import torch\n\nDEVICE_TYPE = 'cuda'\nDEVICE_TYPE_TORCH = 'cuda'\nDEVICE_COUNT = torch.cuda.device_count()\n"
        "ALLOW_PREQUANTIZED_MODELS = True\n\n\ndef is_hip():\n    return False\n\n\ndef get_device_type():\n"
        "    return 'cuda'\n"),
    "unsloth/bnb_availability.py": "def native_kernels_ready(bnb, device_type):\n    return False\n",
    "unsloth_zoo/__init__.py": "",
    "unsloth_zoo/utils.py": "from packaging.version import Version  # noqa: F401\n",
    "unsloth_zoo/log.py": "import logging\n\nlogger = logging.getLogger('unsloth_zoo')\n",
    "unsloth_zoo/temporary_patches/__init__.py": "",
    "unsloth_zoo/temporary_patches/common.py": (
        "def torch_compile(*args, **kwargs):\n    if args and callable(args[0]) and len(args) == 1 and not kwargs:\n"
        "        return args[0]\n    return lambda f: f\n"),
    "unsloth_zoo/patching_utils.py": "def patch_layernorm(*a, **k):\n    return None\n",
    "unsloth_zoo/loss_utils.py": (
        "def patch_loss_functions(*a, **k):\n    return None\n\n\ndef post_patch_loss_function(*a, **k):\n    return None\n"),
    "transformers/__init__.py": "",
    "transformers/models/__init__.py": "",
    "transformers/models/llama/__init__.py": "",
    "transformers/models/llama/modeling_llama.py": (
        "import logging\n\nimport torch\n\nlogger = logging.getLogger('transformers')\n\n\n"
        "class LlamaRMSNorm(torch.nn.Module):  # stub base class (Unsloth subclasses it at import)\n"
        "    def __init__(self, hidden_size, eps=1e-6):\n        super().__init__()\n"
        "        self.weight = torch.nn.Parameter(torch.ones(hidden_size))\n        self.variance_epsilon = eps\n"),
}


def main():
    if DST.exists():
        shutil.rmtree(DST)
    for rel in FILES:
        (DST / "unsloth" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / rel, DST / "unsloth" / rel)
    for rel, text in STUBS.items():
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
        (DST / rel).write_text(text)
    print("built", DST)


if __name__ == "__main__":
    main()
