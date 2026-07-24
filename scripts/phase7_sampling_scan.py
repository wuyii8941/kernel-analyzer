#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    PathConfig,
    _encode_sample,
    attention_backend_context,
    cleanup_memory,
    configure_determinism,
    load_hf_path,
    precision_context,
)
from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import mean, percentile


def path_config(data: dict, key: str) -> PathConfig:
    item = data[key]
    return PathConfig(
        name=item["name"],
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        autocast_dtype=item.get("autocast_dtype"),
        device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False),
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=True,
        model_training_mode=item.get("model_training_mode", False),
        gradient_checkpointing=item.get("gradient_checkpointing", False),
    )


def set_hash(values: list[int]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def common_uniforms(case_id: str, token_index: int, temperature: float, draws: int) -> list[float]:
    values = []
    for draw in range(draws):
        payload = f"forkcert-crn-v1|{case_id}|{token_index}|{temperature:.9g}|{draw}".encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        values.append((integer + 0.5) / 2**64)
    return values


def inverse_cdf_samples(ids, probabilities, uniforms: list[float]) -> tuple[list[int], list[tuple[float, float]]]:
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
        intervals.append((lower, upper))
    return sampled, intervals


def decisions_for_sample(
    tokenizer, model, config: PathConfig, sample: dict, top_k: int, top_p: float, temperature: float, draws: int
) -> list[dict]:
    import torch

    encoded = _encode_sample(tokenizer, sample, config.device)
    input_ids = encoded["input_ids"]
    prompt_len = encoded["prompt_len"]
    response_len = len(encoded["response_ids"])
    with torch.inference_mode(), attention_backend_context(config), precision_context(config):
        logits = model(input_ids=input_ids).logits.float()[0, prompt_len - 1 : prompt_len - 1 + response_len]
        logits = logits / temperature
        top_values, top_ids = torch.topk(logits, k=top_k + 1, dim=-1, largest=True, sorted=True)
        probabilities = torch.softmax(logits, dim=-1)
        sorted_prob, sorted_ids = torch.sort(probabilities, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_prob, dim=-1)
    output = []
    for token_index in range(response_len):
        k_ids = [int(value) for value in top_ids[token_index, :top_k].tolist()]
        k_margin = float((top_values[token_index, top_k - 1] - top_values[token_index, top_k]).item())
        cutoff = int(torch.searchsorted(cumulative[token_index], torch.tensor(top_p, device=cumulative.device)).item())
        cutoff = min(cutoff, cumulative.shape[1] - 1)
        p_ids = [int(value) for value in sorted_ids[token_index, : cutoff + 1].tolist()]
        cumulative_at = float(cumulative[token_index, cutoff].item())
        cumulative_before = float(cumulative[token_index, cutoff - 1].item()) if cutoff > 0 else 0.0
        p_margin = min(top_p - cumulative_before, cumulative_at - top_p)
        uniforms = common_uniforms(str(sample["case_id"]), token_index, temperature, draws)
        k_sampled, k_intervals = inverse_cdf_samples(
            top_ids[token_index, :top_k], torch.softmax(top_values[token_index, :top_k].float(), dim=-1), uniforms
        )
        p_selected_ids = sorted_ids[token_index, : cutoff + 1]
        p_selected_prob = sorted_prob[token_index, : cutoff + 1]
        p_sampled, p_intervals = inverse_cdf_samples(p_selected_ids, p_selected_prob, uniforms)
        output.append(
            {
                "case_id": sample["case_id"],
                "token_index": token_index,
                "token_id": int(encoded["response_ids"][token_index]),
                "top_k_ids": k_ids,
                "top_k_hash": set_hash(k_ids),
                "top_k_margin_logit": k_margin,
                "top_p_ids": p_ids,
                "top_p_hash": set_hash(p_ids),
                "top_p_count": len(p_ids),
                "top_p_margin_probability": p_margin,
                "common_uniforms": uniforms,
                "top_k_sampled_ids": k_sampled,
                "top_k_cdf_intervals": k_intervals,
                "top_p_sampled_ids": p_sampled,
                "top_p_cdf_intervals": p_intervals,
            }
        )
    return output


def run_path(
    config: PathConfig, samples: list[dict], top_k: int, top_p: float, temperature: float, draws: int, warmup: bool
) -> list[list[dict]]:
    tokenizer, model = load_hf_path(config)
    try:
        if warmup:
            for sample in samples:
                decisions_for_sample(tokenizer, model, config, sample, top_k, top_p, temperature, draws)
        runs = []
        for _ in range(2):
            rows = []
            for sample in samples:
                rows.extend(decisions_for_sample(tokenizer, model, config, sample, top_k, top_p, temperature, draws))
            runs.append(rows)
        return runs
    finally:
        del model, tokenizer
        cleanup_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 top-k/top-p candidate-set fork scan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--draws-per-state", type=int, default=64)
    parser.add_argument("--out-jsonl", default="results/phase7_sampling_certificates.jsonl")
    parser.add_argument("--report", default="reports/phase7_sampling.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    samples = read_jsonl(args.samples)[: args.max_samples]
    ref_cfg = path_config(cfg, "path_ref")
    alt_cfg = path_config(cfg, "path_alt")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.draws_per_state <= 0:
        raise ValueError("draws-per-state must be positive")
    ref_runs = run_path(ref_cfg, samples, args.top_k, args.top_p, args.temperature, args.draws_per_state, warmup=False)
    alt_runs = run_path(alt_cfg, samples, args.top_k, args.top_p, args.temperature, args.draws_per_state, warmup=True)
    rows = []
    for ref0, ref1, alt0, alt1 in zip(ref_runs[0], ref_runs[1], alt_runs[0], alt_runs[1], strict=True):
        key = (ref0["case_id"], ref0["token_index"], ref0["token_id"])
        if any((row["case_id"], row["token_index"], row["token_id"]) != key for row in [ref1, alt0, alt1]):
            raise ValueError(f"sampling token alignment mismatch: {key}")
        top_k_draw_forks = [a != b for a, b in zip(ref0["top_k_sampled_ids"], alt0["top_k_sampled_ids"], strict=True)]
        top_p_draw_forks = [a != b for a, b in zip(ref0["top_p_sampled_ids"], alt0["top_p_sampled_ids"], strict=True)]
        rows.append(
            {
                "case_id": ref0["case_id"],
                "token_index": ref0["token_index"],
                "token_id": ref0["token_id"],
                "path_ref": ref_cfg.name,
                "path_alt": alt_cfg.name,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "top_k_margin_ref": ref0["top_k_margin_logit"],
                "top_p_margin_ref": ref0["top_p_margin_probability"],
                "top_k_ref_count": len(ref0["top_k_ids"]),
                "top_k_alt_count": len(alt0["top_k_ids"]),
                "top_p_ref_count": ref0["top_p_count"],
                "top_p_alt_count": alt0["top_p_count"],
                "top_k_symmetric_diff": len(set(ref0["top_k_ids"]) ^ set(alt0["top_k_ids"])),
                "top_p_symmetric_diff": len(set(ref0["top_p_ids"]) ^ set(alt0["top_p_ids"])),
                "top_k_actual_fork": ref0["top_k_hash"] != alt0["top_k_hash"],
                "top_p_actual_fork": ref0["top_p_hash"] != alt0["top_p_hash"],
                "candidate_set_semantics_note": "legacy *_actual_fork fields above mean candidate-set fork, not sampled-token fork",
                "draws_per_state": args.draws_per_state,
                "common_uniforms": ref0["common_uniforms"],
                "top_k_sampled_ref": ref0["top_k_sampled_ids"],
                "top_k_sampled_alt": alt0["top_k_sampled_ids"],
                "top_p_sampled_ref": ref0["top_p_sampled_ids"],
                "top_p_sampled_alt": alt0["top_p_sampled_ids"],
                "top_k_sampling_fork_draws": sum(top_k_draw_forks),
                "top_p_sampling_fork_draws": sum(top_p_draw_forks),
                "top_k_cdf_boundary_fork": any(top_k_draw_forks),
                "top_p_cdf_boundary_fork": any(top_p_draw_forks),
                "top_k_actual_sampling_fork": any(top_k_draw_forks),
                "top_p_actual_sampling_fork": any(top_p_draw_forks),
                "top_k_first_draw_sampling_fork": top_k_draw_forks[0],
                "top_p_first_draw_sampling_fork": top_p_draw_forks[0],
                "top_k_self_sampling_failures": sum(a != b for a, b in zip(ref0["top_k_sampled_ids"], ref1["top_k_sampled_ids"], strict=True)) + sum(a != b for a, b in zip(alt0["top_k_sampled_ids"], alt1["top_k_sampled_ids"], strict=True)),
                "top_p_self_sampling_failures": sum(a != b for a, b in zip(ref0["top_p_sampled_ids"], ref1["top_p_sampled_ids"], strict=True)) + sum(a != b for a, b in zip(alt0["top_p_sampled_ids"], alt1["top_p_sampled_ids"], strict=True)),
                "self_ref_top_k_fork": ref0["top_k_hash"] != ref1["top_k_hash"],
                "self_alt_top_k_fork": alt0["top_k_hash"] != alt1["top_k_hash"],
                "self_ref_top_p_fork": ref0["top_p_hash"] != ref1["top_p_hash"],
                "self_alt_top_p_fork": alt0["top_p_hash"] != alt1["top_p_hash"],
                "region": "unknown",
                "delta_bound_legal": None,
            }
        )
    write_jsonl(args.out_jsonl, rows)
    self_failures = sum(
        any(row[key] for key in ["self_ref_top_k_fork", "self_alt_top_k_fork", "self_ref_top_p_fork", "self_alt_top_p_fork"])
        for row in rows
    )
    summary = {
        "samples": len(samples),
        "tokens": len(rows),
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "draws_per_state": args.draws_per_state,
        "state_draw_trials": len(rows) * args.draws_per_state,
        "self_candidate_set_failures": self_failures,
        "top_k_actual_forks": sum(row["top_k_actual_fork"] for row in rows),
        "top_p_actual_forks": sum(row["top_p_actual_fork"] for row in rows),
        "top_k_sampling_fork_states": sum(row["top_k_actual_sampling_fork"] for row in rows),
        "top_p_sampling_fork_states": sum(row["top_p_actual_sampling_fork"] for row in rows),
        "top_k_sampling_fork_draws": sum(row["top_k_sampling_fork_draws"] for row in rows),
        "top_p_sampling_fork_draws": sum(row["top_p_sampling_fork_draws"] for row in rows),
        "top_k_first_draw_sampling_forks": sum(row["top_k_first_draw_sampling_fork"] for row in rows),
        "top_p_first_draw_sampling_forks": sum(row["top_p_first_draw_sampling_fork"] for row in rows),
        "sampling_self_failures": sum(row["top_k_self_sampling_failures"] + row["top_p_self_sampling_failures"] for row in rows),
        "top_k_min_margin": min(row["top_k_margin_ref"] for row in rows),
        "top_p_min_margin": min(row["top_p_margin_ref"] for row in rows),
        "top_p_count_mean_ref": mean([row["top_p_ref_count"] for row in rows]),
        "top_p_count_p99_ref": percentile([row["top_p_ref_count"] for row in rows], 99),
        "all_regions_unknown": True,
    }
    report = "\n".join(
        [
            "# Phase 7 Sampling Truncation Forks",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- same fixed response tokens: PASS",
            "- same checkpoint and MATH attention backend: PASS",
            "- warmed compile path: PASS",
            "- two candidate-set self runs per path: " + ("PASS" if self_failures == 0 else "FAIL"),
            "- theoretical legal bound available: FAIL; regions remain unknown",
            "",
            "## Delta Self Control",
            f"Candidate-set self mismatches: {self_failures}.",
            "",
            "## External Validity",
            "This scan uses the exact step-5 T4 FP16 snapshot. Temperature scaling is applied before truncation; generation-engine-specific processed-logit paths require separate replication.",
            "",
            "## Summary",
            markdown_table([summary], list(summary.keys())),
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if self_failures:
        raise SystemExit(37)


if __name__ == "__main__":
    main()
