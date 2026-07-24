#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


def set_hash(values: list[int]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def common_uniforms(case_id: str, token_index: int, temperature: float, draws: int) -> list[float]:
    values = []
    for draw in range(draws):
        payload = f"forkcert-crn-v1|{case_id}|{token_index}|{temperature:.9g}|{draw}".encode()
        integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        values.append((integer + 0.5) / 2**64)
    return values


def inverse_cdf(ids, probabilities, uniforms: list[float]) -> tuple[list[int], list[list[float]]]:
    import torch

    normalized = probabilities.float() / probabilities.float().sum()
    cumulative = torch.cumsum(normalized, dim=-1).cpu()
    token_ids = ids.cpu()
    sampled = []
    intervals = []
    for uniform in uniforms:
        index = int(torch.searchsorted(cumulative, torch.tensor(uniform), right=False).item())
        index = min(index, cumulative.numel() - 1)
        lower = float(cumulative[index - 1].item()) if index else 0.0
        upper = float(cumulative[index].item())
        sampled.append(int(token_ids[index].item()))
        intervals.append([lower, upper])
    return sampled, intervals


def decisions(
    logits,
    cases: list[tuple[str, int, int]],
    *,
    top_k: int,
    top_p: float,
    temperature: float,
    draws: int,
) -> list[dict[str, Any]]:
    import torch

    scaled = logits.float() / temperature
    top_values, top_ids = torch.topk(scaled, k=top_k + 1, dim=-1, largest=True, sorted=True)
    probabilities = torch.softmax(scaled, dim=-1)
    sorted_prob, sorted_ids = torch.sort(probabilities, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_prob, dim=-1)
    output = []
    for index, (case_id, token_index, token_id) in enumerate(cases):
        k_ids_tensor = top_ids[index, :top_k]
        k_ids = [int(value) for value in k_ids_tensor.tolist()]
        cutoff = int(torch.searchsorted(cumulative[index], torch.tensor(top_p, device=logits.device)).item())
        cutoff = min(cutoff, cumulative.shape[1] - 1)
        p_ids_tensor = sorted_ids[index, :cutoff + 1]
        p_ids = [int(value) for value in p_ids_tensor.tolist()]
        uniforms = common_uniforms(case_id, token_index, temperature, draws)
        k_sampled, k_intervals = inverse_cdf(
            k_ids_tensor,
            torch.softmax(top_values[index, :top_k], dim=-1),
            uniforms,
        )
        p_sampled, p_intervals = inverse_cdf(p_ids_tensor, sorted_prob[index, :cutoff + 1], uniforms)
        before = float(cumulative[index, cutoff - 1].item()) if cutoff else 0.0
        at = float(cumulative[index, cutoff].item())
        output.append(
            {
                "case_id": case_id,
                "token_index": token_index,
                "token_id": token_id,
                "top_k_ids": k_ids,
                "top_k_hash": set_hash(k_ids),
                "top_k_margin_logit": float((top_values[index, top_k - 1] - top_values[index, top_k]).item()),
                "top_p_hash": set_hash(p_ids),
                "top_p_count": len(p_ids),
                "top_p_margin_probability": min(top_p - before, at - top_p),
                "common_uniforms": uniforms,
                "top_k_sampled_ids": k_sampled,
                "top_k_cdf_intervals": k_intervals,
                "top_p_sampled_ids": p_sampled,
                "top_p_cdf_intervals": p_intervals,
            }
        )
    return output


class Capture:
    def __init__(self, samples: list[dict], args: argparse.Namespace) -> None:
        self.args = args
        self.by_tokens = {
            tuple(int(value) for value in sample["prompt_ids"] + sample["response_ids"]): sample
            for sample in samples
        }
        self.rows: dict[tuple[str, int, int], dict] = {}
        self.formal_calls = 0
        self.logits_dtypes: set[str] = set()

    def wrap(self, original):
        def capture_forward(sampler, logits, sampling_metadata):
            self.logits_dtypes.add(str(logits.dtype))
            for group in sampling_metadata.seq_groups:
                if not group.is_prompt or group.sampling_params.prompt_logprobs is None:
                    continue
                seq_data = group.seq_data[group.seq_ids[0]]
                full_ids = tuple(int(value) for value in seq_data.prompt_token_ids)
                sample = self.by_tokens.get(full_ids)
                if sample is None:
                    continue
                self.formal_calls += 1
                computed_len = seq_data.get_num_computed_tokens()
                query_len = int(group.query_len)
                target_start = computed_len + 1
                target_end = min(computed_len + query_len + 1, len(full_ids))
                target_positions = list(range(target_start, target_end))
                query_indices = list(group.prompt_logprob_indices)
                if len(target_positions) != len(query_indices):
                    raise AssertionError("vLLM prompt position/query index mismatch")
                prompt_len = len(sample["prompt_ids"])
                selected_indices = []
                selected_cases = []
                for query_index, target_position in zip(query_indices, target_positions, strict=True):
                    if target_position < prompt_len:
                        continue
                    token_index = target_position - prompt_len
                    token_id = int(full_ids[target_position])
                    selected_indices.append(query_index)
                    selected_cases.append((str(sample["case_id"]), token_index, token_id))
                if selected_indices:
                    batch = logits[selected_indices]
                    for row in decisions(
                        batch,
                        selected_cases,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        temperature=self.args.temperature,
                        draws=self.args.draws,
                    ):
                        row["path"] = "vllm-fp16-v0-xformers"
                        key = (row["case_id"], row["token_index"], row["token_id"])
                        if key in self.rows and self.rows[key] != row:
                            raise AssertionError(f"non-identical duplicate sampling row: {key}")
                        self.rows[key] = row
            return original(sampler, logits, sampling_metadata)

        return capture_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture vLLM V0 full-vocabulary sampling decisions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--state-jsonl", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.temperature <= 0 or args.draws <= 0:
        raise ValueError("temperature and draws must be positive")

    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.sampler import Sampler

    case_ids = {json.loads(line)["case_id"] for line in Path(args.state_jsonl).read_text().splitlines() if line}
    samples = [
        row for row in (json.loads(line) for line in Path(args.samples).read_text().splitlines() if line)
        if row["case_id"] in case_ids
    ]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    capture = Capture(samples, args)
    original = Sampler.forward
    Sampler.forward = capture.wrap(original)
    try:
        engine = LLM(
            model=str(Path(args.model).resolve()),
            tokenizer=str(Path(args.model).resolve()),
            dtype="float16",
            seed=args.seed,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=512,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=True,
            enable_chunked_prefill=False,
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
        )
        for sample in samples:
            prompt = {"prompt_token_ids": sample["prompt_ids"] + sample["response_ids"]}
            engine.generate([prompt], params, use_tqdm=False)
    finally:
        Sampler.forward = original

    rows = [capture.rows[key] for key in sorted(capture.rows)]
    payload = {
        "metadata": {
            "schema_version": "forkcert.p1.vllm-sampling.v1",
            "vllm_version": importlib.metadata.version("vllm"),
            "torch_version": importlib.metadata.version("torch"),
            "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "samples": len(samples),
            "tokens": len(rows),
            "formal_sampler_calls": capture.formal_calls,
            "input_logits_dtypes": sorted(capture.logits_dtypes),
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "draws": args.draws,
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    if len(rows) != len(samples) * 128:
        raise SystemExit(f"unexpected vLLM sampling coverage: {len(rows)}")


if __name__ == "__main__":
    main()
