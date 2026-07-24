#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    _encode_sample,
    attention_backend_context,
    cleanup_memory,
    configure_determinism,
    load_hf_path,
    precision_context,
)
from scripts.phase6_twin_training import path_config
from scripts.phase7_sampling_scan import common_uniforms, inverse_cdf_samples, set_hash
from scripts.phase9_mutation_catalog import MUTATIONS, selected_mutations


def probabilities_and_log_normalizer(logits: Any, reducer: str) -> tuple[Any, Any]:
    import torch

    values = logits.float()
    if reducer == "standard":
        log_normalizer = torch.logsumexp(values, dim=-1)
        probabilities = torch.exp(values - log_normalizer.unsqueeze(-1))
    elif reducer == "fp16_logsoftmax":
        with torch.autocast(device_type="cuda", enabled=False):
            log_probs = torch.nn.functional.log_softmax(logits.to(torch.float16), dim=-1)
        probabilities = torch.exp(log_probs.float())
        log_normalizer = values.max(dim=-1).values - log_probs.float().max(dim=-1).values
    elif reducer == "chunked_reverse":
        maximum = values.max(dim=-1, keepdim=True).values
        shifted = torch.exp(values - maximum)
        chunks = list(torch.tensor_split(shifted, 8, dim=-1))
        denominator = sum(
            (chunk.sum(dim=-1) for chunk in reversed(chunks)),
            torch.zeros_like(maximum[..., 0]),
        )
        log_normalizer = maximum[..., 0] + torch.log(denominator)
        probabilities = torch.exp(values - log_normalizer.unsqueeze(-1))
    else:
        raise ValueError(f"unsupported reducer: {reducer}")
    return probabilities, log_normalizer


def decisions_for_sample(
    tokenizer: Any,
    model: Any,
    config: Any,
    sample: dict[str, Any],
    reducer: str,
    top_k: int,
    top_p: float,
    temperature: float,
    draws: int,
) -> list[dict[str, Any]]:
    import torch

    encoded = _encode_sample(tokenizer, sample, config.device)
    prompt_len = encoded["prompt_len"]
    response_len = len(encoded["response_ids"])
    with torch.inference_mode(), attention_backend_context(config), precision_context(config):
        logits = model(input_ids=encoded["input_ids"]).logits[
            0, prompt_len - 1 : prompt_len - 1 + response_len
        ]
        logits = logits / temperature
        probabilities, log_normalizer = probabilities_and_log_normalizer(logits, reducer)
        top_values, top_ids = torch.topk(logits.float(), k=top_k + 1, dim=-1, largest=True, sorted=True)
        sorted_prob, sorted_ids = torch.sort(probabilities, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_prob, dim=-1)
    rows = []
    for token_index in range(response_len):
        uniforms = common_uniforms(str(sample["case_id"]), token_index, temperature, draws)
        k_ids = top_ids[token_index, :top_k]
        k_probabilities = probabilities[token_index].gather(0, k_ids)
        k_sampled, _ = inverse_cdf_samples(k_ids, k_probabilities, uniforms)
        cutoff = int(
            torch.searchsorted(cumulative[token_index], torch.tensor(top_p, device=cumulative.device)).item()
        )
        cutoff = min(cutoff, cumulative.shape[1] - 1)
        p_ids = sorted_ids[token_index, : cutoff + 1]
        p_probabilities = sorted_prob[token_index, : cutoff + 1]
        p_sampled, _ = inverse_cdf_samples(p_ids, p_probabilities, uniforms)
        rows.append(
            {
                "case_id": str(sample["case_id"]),
                "token_index": token_index,
                "token_id": int(encoded["response_ids"][token_index]),
                "log_normalizer": float(log_normalizer[token_index].item()),
                "top_k_hash": set_hash([int(value) for value in k_ids.tolist()]),
                "top_p_hash": set_hash([int(value) for value in p_ids.tolist()]),
                "top_k_sampled": k_sampled,
                "top_p_sampled": p_sampled,
            }
        )
    return rows


def run_twice(
    tokenizer: Any, model: Any, config: Any, samples: list[dict[str, Any]], reducer: str, args: Any
) -> list[list[dict[str, Any]]]:
    runs = []
    for _ in range(2):
        rows = []
        for sample in samples:
            rows.extend(
                decisions_for_sample(
                    tokenizer,
                    model,
                    config,
                    sample,
                    reducer,
                    args.top_k,
                    args.top_p,
                    args.temperature,
                    args.draws_per_state,
                )
            )
        runs.append(rows)
    return runs


def compare_rows(reference: list[dict[str, Any]], alternative: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for ref, alt in zip(reference, alternative, strict=True):
        key = (ref["case_id"], ref["token_index"], ref["token_id"])
        if key != (alt["case_id"], alt["token_index"], alt["token_id"]):
            raise ValueError(f"sampling alignment mismatch: {key}")
        output.append(
            {
                "case_id": ref["case_id"],
                "token_index": ref["token_index"],
                "token_id": ref["token_id"],
                "log_normalizer_delta": alt["log_normalizer"] - ref["log_normalizer"],
                "top_k_candidate_set_fork": alt["top_k_hash"] != ref["top_k_hash"],
                "top_p_candidate_set_fork": alt["top_p_hash"] != ref["top_p_hash"],
                "top_k_sampling_fork_draws": sum(
                    left != right for left, right in zip(ref["top_k_sampled"], alt["top_k_sampled"], strict=True)
                ),
                "top_p_sampling_fork_draws": sum(
                    left != right for left, right in zip(ref["top_p_sampled"], alt["top_p_sampled"], strict=True)
                ),
                "top_k_first_draw_sampling_fork": ref["top_k_sampled"][0] != alt["top_k_sampled"][0],
                "top_p_first_draw_sampling_fork": ref["top_p_sampled"][0] != alt["top_p_sampled"][0],
            }
        )
    return output


def self_failures(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> int:
    return sum(
        row["top_k_hash"] != other["top_k_hash"]
        or row["top_p_hash"] != other["top_p_hash"]
        or row["top_k_sampled"] != other["top_k_sampled"]
        or row["top_p_sampled"] != other["top_p_sampled"]
        or row["log_normalizer"] != other["log_normalizer"]
        for row, other in zip(first, second, strict=True)
    )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 12 Clipping-surviving Mutation Sampling",
        "",
        "## Objective",
        "",
        "Test whether mutations with zero clipping fork at the discovery state cross top-k/top-p sampling boundaries under common random numbers.",
        "",
        "## Controls",
        "",
        "- Same checkpoint, prompt/response token IDs, temperature and common uniform draws.",
        f"- Restricted to rollout batch `{summary['rollout_batch']}`, the exact batch used to establish zero clipping forks.",
        "- Two complete deterministic runs for clean and every mutation path.",
        "- Final-reduction mutations alter the probability normalization used by sampling rather than becoming no-op probes.",
        "",
        "## Results",
        "",
        "| Mutation | Canary max log-normalizer delta | top-k set forks | top-p set forks | top-k fork draws | top-p fork draws | top-k first draw | top-p first draw | Either first draw | Self failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["mutations"]:
        lines.append(
            f"| {row['mutation']} | {row['canary_max_abs_log_normalizer_delta']:.6g} | "
            f"{row['top_k_candidate_set_forks']} | {row['top_p_candidate_set_forks']} | "
            f"{row['top_k_sampling_fork_draws']} | {row['top_p_sampling_fork_draws']} | "
            f"{row['top_k_first_draw_sampling_forks']} | {row['top_p_first_draw_sampling_forks']} | "
            f"{row['first_draw_actual_sampling_forks']} | {row['self_failures']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            summary["claim_scope"],
            "",
            "## Artifacts",
            "",
            f"- `{summary['artifacts']['summary']}`",
            f"- `{summary['artifacts']['rows']}`",
            "- `scripts/phase12_mutation_sampling.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sampling scan for clipping-surviving mutations.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--mutation", action="append", choices=[item.name for item in MUTATIONS], required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--rollout-batch",
        type=int,
        default=1,
        help="Restrict sampling states to the rollout batch used by the clipping-survival catalog.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--draws-per-state", type=int, default=64)
    parser.add_argument("--out-dir", default="results/phase12_mutation_sampling")
    parser.add_argument("--report", default="reports/phase12_mutation_sampling.md")
    args = parser.parse_args()
    configure_determinism(0)
    config = path_config(load_config(args.config), "path_ref")
    eligible_samples = [
        sample
        for sample in read_jsonl(args.samples)
        if int(sample.get("metadata", {}).get("rollout_batch", -1)) == args.rollout_batch
    ]
    samples = eligible_samples[: args.max_samples]
    if not samples:
        raise ValueError(f"no samples for rollout batch {args.rollout_batch}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_hf_path(config)
    try:
        reference_runs = run_twice(tokenizer, model, config, samples, "standard", args)
        clean_self = self_failures(*reference_runs)
        summaries = []
        all_rows = []
        for mutation in selected_mutations(args.mutation):
            with ExitStack() as stack:
                changed = mutation.installer(model, stack)
                mutation_runs = run_twice(tokenizer, model, config, samples, mutation.reducer, args)
            rows = compare_rows(reference_runs[0], mutation_runs[0])
            failures = clean_self + self_failures(*mutation_runs)
            canary = max(abs(row["log_normalizer_delta"]) for row in rows)
            # Model-component mutations may preserve the log-normalizer while moving individual logits;
            # candidate or sampled-token differences are also valid execution canaries.
            decision_canary = any(
                row["top_k_candidate_set_fork"]
                or row["top_p_candidate_set_fork"]
                or row["top_k_sampling_fork_draws"]
                or row["top_p_sampling_fork_draws"]
                for row in rows
            )
            valid_canary = canary > 0.0 or decision_canary
            if not valid_canary:
                raise ValueError(f"sampling canary is zero for {mutation.name}")
            for row in rows:
                row.update({"schema_version": "forkcert.mutation_sampling_row.v1", "mutation": mutation.name})
            write_jsonl(out_dir / f"{mutation.name}.jsonl", rows)
            all_rows.extend(rows)
            summaries.append(
                {
                    "mutation": mutation.name,
                    "reducer": mutation.reducer,
                    "changed_modules": changed,
                    "canary_max_abs_log_normalizer_delta": canary,
                    "top_k_candidate_set_forks": sum(row["top_k_candidate_set_fork"] for row in rows),
                    "top_p_candidate_set_forks": sum(row["top_p_candidate_set_fork"] for row in rows),
                    "top_k_sampling_fork_draws": sum(row["top_k_sampling_fork_draws"] for row in rows),
                    "top_p_sampling_fork_draws": sum(row["top_p_sampling_fork_draws"] for row in rows),
                    "top_k_first_draw_sampling_forks": sum(
                        row["top_k_first_draw_sampling_fork"] for row in rows
                    ),
                    "top_p_first_draw_sampling_forks": sum(
                        row["top_p_first_draw_sampling_fork"] for row in rows
                    ),
                    "first_draw_actual_sampling_forks": sum(
                        row["top_k_first_draw_sampling_fork"] or row["top_p_first_draw_sampling_fork"] for row in rows
                    ),
                    "self_failures": failures,
                }
            )
        write_jsonl(out_dir / "all_rows.jsonl", all_rows)
        summary = {
            "schema_version": "forkcert.mutation_sampling_summary.v1",
            "samples": len(samples),
            "rollout_batch": args.rollout_batch,
            "tokens_per_mutation": len(reference_runs[0]),
            "draws_per_state": args.draws_per_state,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "mutations": summaries,
            "artifacts": {
                "summary": str(out_dir / "summary.json"),
                "rows": str(out_dir / "all_rows.jsonl"),
                "report": args.report,
            },
            "claim_scope": (
                "Artificial mutations with zero discovery-state clipping fork. Candidate-set and common-random-number "
                "sampled-token forks are semantic events; they do not establish reward or task-quality harm."
            ),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        Path(args.report).write_text(render_report(summary), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        del model, tokenizer
        cleanup_memory()


if __name__ == "__main__":
    main()
