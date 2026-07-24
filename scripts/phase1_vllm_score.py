#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any


def chosen_prompt_logprob(entry: Any, token_id: int) -> float:
    if entry is None:
        raise ValueError("prompt logprob entry is None for a scored token")
    value = entry.get(token_id) if hasattr(entry, "get") else None
    if value is None and hasattr(entry, "get"):
        value = entry.get(str(token_id))
    if value is None:
        raise KeyError(f"actual prompt token {token_id} missing from prompt_logprobs entry")
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def extract_response_rows(sample: dict[str, Any], request_output: Any) -> list[dict[str, Any]]:
    prompt_ids = [int(value) for value in sample["prompt_ids"]]
    response_ids = [int(value) for value in sample["response_ids"]]
    full_ids = prompt_ids + response_ids
    prompt_logprobs = request_output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(full_ids):
        raise ValueError(
            f"prompt_logprobs coverage mismatch: got {None if prompt_logprobs is None else len(prompt_logprobs)}, "
            f"expected {len(full_ids)}"
        )
    rows = []
    for token_index, token_id in enumerate(response_ids):
        full_position = len(prompt_ids) + token_index
        rows.append(
            {
                "case_id": str(sample["case_id"]),
                "token_index": token_index,
                "token_id": token_id,
                "logp": chosen_prompt_logprob(prompt_logprobs[full_position], token_id),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="One-process vLLM teacher-forcing prompt-logprob scorer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--canary-even-token-logit-shift", type=float, default=0.0)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"vLLM model directory does not exist: {model_path}")

    from vllm import LLM, SamplingParams

    samples = [json.loads(line) for line in Path(args.samples).read_text(encoding="utf-8").splitlines() if line][
        : args.max_samples
    ]
    if args.reverse_order:
        samples.reverse()
    prompts = [
        {"prompt_token_ids": [int(value) for value in sample["prompt_ids"] + sample["response_ids"]]}
        for sample in samples
    ]
    if args.canary_even_token_logit_shift:
        from vllm.model_executor.layers.sampler import Sampler

        original_forward = Sampler.forward

        def canary_forward(sampler, logits, sampling_metadata):
            shifted = logits.clone()
            shifted[:, 0::2] += args.canary_even_token_logit_shift
            return original_forward(sampler, shifted, sampling_metadata)

        Sampler.forward = canary_forward
    engine = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype=args.dtype,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
        enable_chunked_prefill=args.chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        min_tokens=0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=None,
        detokenize=False,
        seed=args.seed,
        logits_processors=None,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words=None,
    )
    outputs = engine.generate(prompts, params, use_tqdm=False)
    by_request = {str(output.request_id): output for output in outputs}
    if len(outputs) != len(samples):
        raise ValueError(f"vLLM output count mismatch: {len(outputs)} != {len(samples)}")

    rows = []
    # Offline LLM preserves request order, but assert through the returned prompt IDs
    # instead of relying only on that API behavior.
    for index, (sample, output) in enumerate(zip(samples, outputs, strict=True)):
        expected = [int(value) for value in sample["prompt_ids"] + sample["response_ids"]]
        observed = [int(value) for value in output.prompt_token_ids]
        if observed != expected:
            raise ValueError(f"request {index} prompt token mismatch")
        rows.extend(extract_response_rows(sample, output))

    metadata = {
        "schema_version": "forkcert.vllm_score.v1",
        "model": str(model_path),
        "vllm_version": importlib.metadata.version("vllm"),
        "torch_version": importlib.metadata.version("torch"),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
        "dtype": args.dtype,
        "seed": args.seed,
        "engine_enforce_eager": args.enforce_eager,
        "engine_chunked_prefill": args.chunked_prefill,
        "engine_max_num_batched_tokens": args.max_num_batched_tokens,
        "engine_max_num_seqs": args.max_num_seqs,
        "engine_gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend_env": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "canary_even_token_logit_shift": args.canary_even_token_logit_shift,
        "teacher_forcing_protocol": "full prompt+response token IDs scored through prompt_logprobs",
        "prompt_logprobs": 1,
        "temperature": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_p": 0.0,
        "min_tokens": 0,
        "custom_logits_processors": None,
        "top_p": 1.0,
        "top_k": -1,
        "generated_token_excluded": True,
        "raw_vs_processed_mode": (
            "v0.9.2 API has no selector; processor-free identity is audited separately in "
            "results/p1_vllm/raw_processed_identity.json"
        ),
        "requests": len(samples),
        "tokens": len(rows),
        "request_order": "reversed" if args.reverse_order else "original",
        "unused_request_index_size": len(by_request),
    }
    payload = {"metadata": metadata, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
