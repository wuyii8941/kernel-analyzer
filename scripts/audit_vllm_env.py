#!/usr/bin/env python
from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
from pathlib import Path


def main() -> None:
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    signature = inspect.signature(LLM)
    sampling_signature = inspect.signature(SamplingParams)
    capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    result = {
        "schema_version": "forkcert.vllm_env.v1",
        "python": os.sys.executable,
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": capability,
        "native_bf16_hardware": bool(capability and capability[0] >= 8),
        "v1_hardware_supported": bool(capability and capability[0] >= 8),
        "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
        "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "llm_has_logprobs_mode": "logprobs_mode" in signature.parameters,
        "sampling_has_prompt_logprobs": "prompt_logprobs" in sampling_signature.parameters,
        "sampling_has_logprobs_mode": "logprobs_mode" in sampling_signature.parameters,
    }
    out = Path("results/vllm_env_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
