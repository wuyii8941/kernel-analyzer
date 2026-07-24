#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import struct
from pathlib import Path
from typing import Any


def chosen_logprob(entry: Any, token_id: int) -> float:
    value = entry.get(token_id) if hasattr(entry, "get") else None
    if value is None and hasattr(entry, "get"):
        value = entry.get(str(token_id))
    if value is None:
        raise KeyError(f"token {token_id} missing from prompt logprobs")
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def f32_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", float(value)) for value in values)


def identity_params(params: Any) -> dict[str, bool]:
    return {
        "temperature_one": float(params.temperature) == 1.0,
        "top_p_unrestricted": float(params.top_p) == 1.0,
        "top_k_unrestricted": int(params.top_k) in {-1, 0},
        "min_p_disabled": float(params.min_p) == 0.0,
        "presence_penalty_disabled": float(params.presence_penalty) == 0.0,
        "frequency_penalty_disabled": float(params.frequency_penalty) == 0.0,
        "repetition_penalty_disabled": float(params.repetition_penalty) == 1.0,
        "min_tokens_disabled": int(params.min_tokens) == 0,
        "custom_logits_processors_disabled": not params.logits_processors,
        "logit_bias_disabled": not params.logit_bias,
        "allowed_token_ids_disabled": params.allowed_token_ids is None,
        "bad_words_disabled": not params.bad_words,
    }


class SamplerIdentityAudit:
    def __init__(self, expect_identity: bool) -> None:
        self.expect_identity = expect_identity
        self.raw: list[float] = []
        self.processed: list[float] = []
        self.calls = 0
        self.parameter_checks: list[dict[str, bool]] = []
        self.input_dtypes: set[str] = set()

    def wrap(self, original):
        import torch
        from vllm.model_executor.layers.sampler import _get_next_prompt_tokens

        def audited_forward(sampler, logits, sampling_metadata):
            self.calls += 1
            self.input_dtypes.add(str(logits.dtype))
            group_raw: list[list[tuple[int, float]]] = []
            raw_logprobs = torch.log_softmax(logits.float(), dim=-1)
            for group in sampling_metadata.seq_groups:
                selected: list[tuple[int, float]] = []
                if group.is_prompt and group.sampling_params.prompt_logprobs is not None:
                    checks = identity_params(group.sampling_params)
                    self.parameter_checks.append(checks)
                    if self.expect_identity and not all(checks.values()):
                        failed = [key for key, passed in checks.items() if not passed]
                        raise AssertionError(f"non-identity sampling parameters: {failed}")
                    token_ids = list(_get_next_prompt_tokens(group))
                    query_indices = list(group.prompt_logprob_indices)
                    if len(token_ids) != len(query_indices):
                        raise AssertionError("prompt token/query index length mismatch")
                    for query_index, token_id in zip(query_indices, token_ids, strict=True):
                        selected.append((int(token_id), float(raw_logprobs[query_index, token_id].item())))
                group_raw.append(selected)

            output = original(sampler, logits, sampling_metadata)
            if output is None:
                raise AssertionError("sampler returned no output during identity audit")
            if len(output.outputs) != len(group_raw):
                raise AssertionError("sampler output/group count mismatch")
            for group_output, selected in zip(output.outputs, group_raw, strict=True):
                if not selected:
                    continue
                entries = group_output.prompt_logprobs
                if entries is None or len(entries) != len(selected):
                    raise AssertionError("processed prompt-logprob coverage mismatch")
                for entry, (token_id, raw_value) in zip(entries, selected, strict=True):
                    self.raw.append(raw_value)
                    self.processed.append(chosen_logprob(entry, token_id))
            return output

        return audited_forward

    def summary(self) -> dict[str, Any]:
        raw_bytes = f32_bytes(self.raw)
        processed_bytes = f32_bytes(self.processed)
        mismatches = sum(
            raw_bytes[index:index + 4] != processed_bytes[index:index + 4]
            for index in range(0, len(raw_bytes), 4)
        )
        max_abs = max((abs(a - b) for a, b in zip(self.raw, self.processed, strict=True)), default=0.0)
        checks = {
            key: all(item[key] for item in self.parameter_checks)
            for key in self.parameter_checks[0]
        } if self.parameter_checks else {}
        return {
            "sampler_calls": self.calls,
            "tokens_compared": len(self.raw),
            "input_logits_dtypes": sorted(self.input_dtypes),
            "processor_parameter_checks": checks,
            "all_processors_identity": bool(checks) and all(checks.values()),
            "bitwise_mismatch_count": mismatches,
            "max_abs_delta": max_abs,
            "raw_f32_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "processed_f32_sha256": hashlib.sha256(processed_bytes).hexdigest(),
            "bitwise_equal": raw_bytes == processed_bytes,
        }


def run_engine(args: argparse.Namespace, *, temperature: float, expect_identity: bool) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.sampler import Sampler

    samples = [json.loads(line) for line in Path(args.samples).read_text(encoding="utf-8").splitlines() if line][
        : args.max_samples
    ]
    prompts = [{"prompt_token_ids": sample["prompt_ids"] + sample["response_ids"]} for sample in samples]
    audit = SamplerIdentityAudit(expect_identity=expect_identity)
    original = Sampler.forward
    Sampler.forward = audit.wrap(original)
    try:
        engine = LLM(
            model=str(Path(args.model).resolve()),
            tokenizer=str(Path(args.model).resolve()),
            dtype="float16",
            seed=args.seed,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=True,
            enable_chunked_prefill=False,
        )
        params = SamplingParams(
            temperature=temperature,
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
        for start in range(0, len(prompts), args.request_batch_size):
            engine.generate(prompts[start:start + args.request_batch_size], params, use_tqdm=False)
    finally:
        Sampler.forward = original
    return audit.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit vLLM V0 raw versus identity-processed prompt logprobs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=1000)
    parser.add_argument("--request-batch-size", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.request_batch_size <= 0:
        raise ValueError("request-batch-size must be positive")

    identity = run_engine(args, temperature=1.0, expect_identity=True)
    payload = {
        "schema_version": "forkcert.vllm.raw_processed_identity.v1",
        "vllm_version": importlib.metadata.version("vllm"),
        "torch_version": importlib.metadata.version("torch"),
        "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
        "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "model": str(Path(args.model).resolve()),
        "samples": str(Path(args.samples).resolve()),
        "requested_samples": args.max_samples,
        "minimum_tokens": args.min_tokens,
        "identity": identity,
    }
    payload["passed"] = bool(
        identity["tokens_compared"] >= args.min_tokens
        and identity["all_processors_identity"]
        and identity["bitwise_equal"]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
