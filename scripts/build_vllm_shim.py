#!/usr/bin/env python3
"""Build .cache/pylibs/vllm_shim: the Triton kernel modules of vLLM main, importable without the rest of vLLM.

The listed source files are copied verbatim from .cache/src/vllm-main; every other ``vllm.*`` module that they import
is replaced by a stub (``vllm/__init__.py`` installs a finder that creates them on demand), except a few small
modules written out below (triton_utils, envs, logger, platforms, kv_cache_interface).

    python scripts/build_vllm_shim.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".cache/src/vllm-main/vllm"
DST = ROOT / ".cache/pylibs/vllm_shim/vllm"

FILES = [
    "v1/attention/ops/triton_unified_attention.py",
    "v1/attention/ops/triton_attention_helpers.py",
    "v1/attention/ops/triton_decode_attention.py",
    "v1/attention/ops/triton_merge_attn_states.py",
    "v1/attention/ops/triton_prefill_attention.py",
    "v1/attention/ops/prefix_prefill.py",
    "v1/attention/ops/chunked_prefill_paged_decode.py",
    "model_executor/layers/mamba/ops/causal_conv1d.py",
    "model_executor/layers/rotary_embedding/mrope.py",
    "model_executor/layers/fused_qk_norm_rope.py",
    "v1/worker/gpu/sample/min_p.py",
    "v1/worker/gpu/sample/penalties.py",
    "v1/worker/gpu/sample/logprob.py",
    "v1/worker/gpu/sample/gumbel.py",
    "v1/sample/ops/topk_topp_triton.py",
    "model_executor/layers/mamba/ops/mamba_ssm.py",
    "model_executor/layers/mamba/ops/ssd_bmm.py",
    "model_executor/layers/mamba/ops/ssd_chunk_scan.py",
    "model_executor/layers/mamba/ops/ssd_chunk_state.py",
    "model_executor/layers/mamba/ops/ssd_state_passing.py",
    "model_executor/layers/mamba/ops/ssd_combined.py",
    "model_executor/layers/mamba/ops/layernorm_gated.py",
    "model_executor/layers/mamba/ops/triton_helpers.py",
    "utils/math_utils.py",
    "lora/ops/triton_ops/lora_shrink_op.py",
    "lora/ops/triton_ops/lora_expand_op.py",
    "lora/ops/triton_ops/kernel_utils.py",
    "lora/ops/triton_ops/utils.py",
    "model_executor/layers/fused_moe/fused_moe.py",
    "model_executor/determinism/batch_invariant.py",
    "kernels/triton/activation.py",
    "v1/attention/ops/triton_reshape_and_cache_flash.py",
    "v1/worker/gpu/sample/logit_bias.py",
    "v1/worker/gpu/sample/bad_words.py",
    "v1/sample/rejection_sampler.py",
    "model_executor/layers/lightning_attn.py",
    "model_executor/warmup/jit_warmup.py",
    "model_executor/warmup/jit_warmup_triton_helper.py",
    "utils/gpu_sync_debug.py",
]
DIRS = ["third_party/flash_linear_attention"]

INIT = '''"""vLLM shim: real Triton kernel modules + on-demand stubs for everything else (scripts/build_vllm_shim.py)."""
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()


class _Stub(metaclass=_StubMeta):
    def __init__(self, *a, **k):
        pass

    def __class_getitem__(cls, item):
        return cls

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return _Stub()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()

    def __getitem__(self, item):
        return _Stub()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (_Stub,), {})
        setattr(self, name, cls)
        return cls


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if not name.startswith("vllm."):
            return None
        rel = Path(*name.split(".")[1:])
        if (_HERE / rel).with_suffix(".py").exists() or (_HERE / rel / "__init__.py").exists() or (_HERE / rel).is_dir():
            return None
        return importlib.machinery.ModuleSpec(name, _Loader(), is_package=True)


sys.meta_path.append(_Finder())
'''

TORCH_UTILS = '''import torch


def is_quantized_kv_cache(kv_cache_dtype):  # vllm/utils/torch_utils.py
    return (kv_cache_dtype.startswith("fp8") or kv_cache_dtype.endswith("per_token_head")
            or kv_cache_dtype.startswith("nvfp4"))


def async_tensor_h2d(data, device=None, dtype=None, out=None):
    t = torch.as_tensor(data, dtype=dtype)
    if out is not None:
        out.copy_(t, non_blocking=False)
        return out
    return t.to(device)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from vllm import _Stub
    return type(name, (_Stub,), {})
'''

BACKEND_UTILS = '''PAD_SLOT_ID = -1  # values of vllm/v1/attention/backends/utils.py
NULL_BLOCK_ID = 0


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from vllm import _Stub
    return type(name, (_Stub,), {})
'''

PLATFORM_UTILS = '''import torch


def num_compute_units(device_id=None):
    return torch.cuda.get_device_properties(device_id if device_id is not None else 0).multi_processor_count


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from vllm import _Stub
    return type(name, (_Stub,), {})
'''

QUANT_UTILS = '''import torch

FP8_DTYPE = torch.float8_e4m3fn


def get_fp8_min_max(dtype=None):
    fi = torch.finfo(dtype or FP8_DTYPE)
    return float(fi.min), float(fi.max)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from vllm import _Stub
    return type(name, (_Stub,), {})
'''

TRITON_UTILS = '''import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

HAS_TRITON = True
LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453


def use_tensor_descriptor(*a, **k):
    return False
'''

ENVS = '''def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return False
'''

LOGGER = '''import logging


class _Logger(logging.LoggerAdapter):
    def __init__(self, name):
        super().__init__(logging.getLogger(name), {})

    def info_once(self, msg, *a, **k):
        self.info(msg, *a)

    def warning_once(self, msg, *a, **k):
        self.warning(msg, *a)

    def debug_once(self, msg, *a, **k):
        self.debug(msg, *a)


def init_logger(name):
    return _Logger(name)
'''

PLATFORMS = '''import torch


class _Platform:
    device_type = "cuda"

    def is_rocm(self):
        return False

    def is_cuda(self):
        return True

    def is_cuda_alike(self):
        return True

    def is_xpu(self):
        return False

    def is_cpu(self):
        return False

    def fp8_dtype(self):
        return torch.float8_e4m3fn

    def get_device_capability(self, device_id=0):
        return torch.cuda.get_device_capability(device_id)

    def has_device_capability(self, cap, device_id=0):
        major, minor = torch.cuda.get_device_capability(device_id)
        return (major * 10 + minor) >= (cap if isinstance(cap, int) else cap[0] * 10 + cap[1])

    def is_device_capability(self, cap, device_id=0):
        major, minor = torch.cuda.get_device_capability(device_id)
        return (major * 10 + minor) == (cap if isinstance(cap, int) else cap[0] * 10 + cap[1])

    def is_device_capability_family(self, fam, device_id=0):
        major, _ = torch.cuda.get_device_capability(device_id)
        return major * 10 == fam

    def __getattr__(self, name):
        return lambda *a, **k: False


current_platform = _Platform()
'''


def main():
    DST.mkdir(parents=True, exist_ok=True)  # files are overwritten in place (running jobs keep importing)
    (DST / "__init__.py").write_text(INIT)
    for rel, text in {"triton_utils/__init__.py": TRITON_UTILS, "envs.py": ENVS, "logger.py": LOGGER,
                      "platforms/__init__.py": PLATFORMS,
                      "utils/torch_utils.py": TORCH_UTILS,
                      "v1/attention/backends/utils.py": BACKEND_UTILS,
                      "utils/platform_utils.py": PLATFORM_UTILS,
                      "model_executor/layers/quantization/utils/quant_utils.py": QUANT_UTILS}.items():
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
        (DST / rel).write_text(text)
    # kv_cache_interface: only the enum
    src = (SRC / "v1/kv_cache_interface.py").read_text()
    start = src.index("class KVQuantMode(IntEnum):")
    end = src.index("\nclass ", start + 10) if "\nclass " in src[start + 10:] else len(src)
    body = src[start:end]
    (DST / "v1").mkdir(parents=True, exist_ok=True)
    (DST / "v1/kv_cache_interface.py").write_text("from enum import IntEnum\n\n\n" + body.split("\n\n\n")[0] + "\n")
    for rel in FILES:
        if not (SRC / rel).exists():
            print("missing", rel)
            continue
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / rel, DST / rel)
    for rel in DIRS:
        shutil.copytree(SRC / rel, DST / rel, dirs_exist_ok=True)
    # packages need __init__.py so the finder treats them as real directories
    for d in DST.rglob("*"):
        if d.is_dir() and not (d / "__init__.py").exists():
            (d / "__init__.py").write_text("")
    print(f"vllm shim at {DST.parent}: {len(FILES)} files, {len(DIRS)} directories")


if __name__ == "__main__":
    main()
