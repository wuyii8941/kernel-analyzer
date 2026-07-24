#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import types
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    PathConfig,
    attention_backend_context,
    cleanup_memory,
    configure_determinism,
    load_hf_path,
    precision_context,
)
from scripts.phase6_twin_training import path_config
from scripts.phase8_case_attribution import make_batch


@dataclass(frozen=True)
class Mutation:
    name: str
    description: str
    mechanism: str
    installer: Callable[[Any, ExitStack], list[str]]
    reducer: str = "standard"


def named_by_class(model: Any, fragment: str) -> list[tuple[str, Any]]:
    needle = fragment.lower()
    return [(name, module) for name, module in model.named_modules() if needle in module.__class__.__name__.lower()]


def replace_forward(stack: ExitStack, module: Any, replacement: Callable[..., Any]) -> None:
    original = module.forward
    module.forward = types.MethodType(replacement, module)
    stack.callback(setattr, module, "forward", original)


def install_rmsnorm_no_upcast(model: Any, stack: ExitStack) -> list[str]:
    import torch

    changed = []
    for name, module in named_by_class(model, "RMSNorm"):
        def forward(this: Any, hidden_states: Any) -> Any:
            input_dtype = hidden_states.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                values = hidden_states.to(torch.float16)
                variance = values.pow(2).mean(-1, keepdim=True)
                normalized = values * torch.rsqrt(variance + this.variance_epsilon)
            return this.weight * normalized.to(input_dtype)

        replace_forward(stack, module, forward)
        changed.append(name)
    return changed


def install_rmsnorm_eps(model: Any, stack: ExitStack) -> list[str]:
    changed = []
    for name, module in named_by_class(model, "RMSNorm"):
        original = float(module.variance_epsilon)
        module.variance_epsilon = 1e-5
        stack.callback(setattr, module, "variance_epsilon", original)
        changed.append(name)
    return changed


def install_rmsnorm_nminus1(model: Any, stack: ExitStack) -> list[str]:
    import torch

    changed = []
    for name, module in named_by_class(model, "RMSNorm"):
        def forward(this: Any, hidden_states: Any) -> Any:
            input_dtype = hidden_states.dtype
            values = hidden_states.float()
            denominator = max(values.shape[-1] - 1, 1)
            variance = values.pow(2).sum(-1, keepdim=True) / denominator
            normalized = values * torch.rsqrt(variance + this.variance_epsilon)
            return this.weight * normalized.to(input_dtype)

        replace_forward(stack, module, forward)
        changed.append(name)
    return changed


def install_rotary_tail_unrotated(model: Any, stack: ExitStack) -> list[str]:
    import torch
    from transformers.models.qwen3 import modeling_qwen3

    original = modeling_qwen3.apply_rotary_pos_emb

    def rotate_half(x: Any) -> Any:
        midpoint = x.shape[-1] // 2
        return torch.cat((-x[..., midpoint:], x[..., :midpoint]), dim=-1)

    def mutated(q: Any, k: Any, cos: Any, sin: Any, unsqueeze_dim: int = 1) -> tuple[Any, Any]:
        cosine = cos.unsqueeze(unsqueeze_dim)
        sine = sin.unsqueeze(unsqueeze_dim)
        q_rotated = (q * cosine) + (rotate_half(q) * sine)
        k_rotated = (k * cosine) + (rotate_half(k) * sine)
        cutoff = 3 * q.shape[-1] // 4
        q_out = torch.cat((q_rotated[..., :cutoff], q[..., cutoff:]), dim=-1)
        k_out = torch.cat((k_rotated[..., :cutoff], k[..., cutoff:]), dim=-1)
        return q_out, k_out

    modeling_qwen3.apply_rotary_pos_emb = mutated
    stack.callback(setattr, modeling_qwen3, "apply_rotary_pos_emb", original)
    return ["transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb"]


def install_rotary_phase_fp16(model: Any, stack: ExitStack) -> list[str]:
    import torch

    changed = []
    for name, module in named_by_class(model, "RotaryEmbedding"):
        def forward(this: Any, x: Any, position_ids: Any) -> tuple[Any, Any]:
            inv = this.inv_freq[None, :, None].to(device=x.device, dtype=torch.float16)
            inv = inv.expand(position_ids.shape[0], -1, 1)
            positions = position_ids[:, None, :].to(dtype=torch.float16)
            with torch.autocast(device_type="cuda", enabled=False):
                freqs = (inv @ positions).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * this.attention_scaling
                sin = emb.sin() * this.attention_scaling
            return cos.to(x.dtype), sin.to(x.dtype)

        replace_forward(stack, module, forward)
        changed.append(name)
    return changed


def install_attention_scale(model: Any, stack: ExitStack) -> list[str]:
    changed = []
    for name, module in named_by_class(model, "Attention"):
        if not hasattr(module, "scaling"):
            continue
        original = float(module.scaling)
        module.scaling = original * 1.001
        stack.callback(setattr, module, "scaling", original)
        changed.append(name)
    return changed


def install_output_rounding(model: Any, stack: ExitStack, selector: Callable[[str, Any], bool]) -> list[str]:
    import torch

    selected = [(name, module) for name, module in model.named_modules() if selector(name, module)]
    changed = []
    for name, module in selected:
        original = module.forward

        def forward(this: Any, *args: Any, _original: Callable[..., Any] = original, **kwargs: Any) -> Any:
            output = _original(*args, **kwargs)

            def rounded(tensor: Any) -> Any:
                return tensor.to(torch.bfloat16).to(tensor.dtype)

            if isinstance(output, tuple):
                return (rounded(output[0]), *output[1:])
            if isinstance(output, list):
                return [rounded(output[0]), *output[1:]]
            return rounded(output)

        replace_forward(stack, module, forward)
        changed.append(name)
    return changed


def first_match(fragment: str) -> Callable[[str, Any], bool]:
    seen = False

    def selector(name: str, module: Any) -> bool:
        nonlocal seen
        if seen or fragment not in name:
            return False
        seen = True
        return True

    return selector


def install_q_proj_round(model: Any, stack: ExitStack) -> list[str]:
    return install_output_rounding(model, stack, first_match(".self_attn.q_proj"))


def install_attention_output_round(model: Any, stack: ExitStack) -> list[str]:
    return install_output_rounding(model, stack, first_match(".self_attn"))


def install_mlp_gate_round(model: Any, stack: ExitStack) -> list[str]:
    return install_output_rounding(model, stack, first_match(".mlp.gate_proj"))


def install_mlp_output_round(model: Any, stack: ExitStack) -> list[str]:
    return install_output_rounding(model, stack, first_match(".mlp"))


def install_embedding_round(model: Any, stack: ExitStack) -> list[str]:
    return install_output_rounding(model, stack, lambda name, _module: name.endswith("embed_tokens"))


def install_lm_head_round(model: Any, stack: ExitStack) -> list[str]:
    return install_output_rounding(model, stack, lambda name, _module: name == "lm_head")


def install_decoder_output_round(model: Any, stack: ExitStack) -> list[str]:
    layers = [(name, module) for name, module in model.named_modules() if name.endswith(".layers.0")]
    names = {name for name, _ in layers}
    return install_output_rounding(model, stack, lambda name, _module: name in names)


def install_none(model: Any, stack: ExitStack) -> list[str]:
    del model, stack
    return ["logprob_reducer"]


MUTATIONS = [
    Mutation("rmsnorm_no_upcast", "Compute RMSNorm variance in the incoming FP16 dtype.", "missing FP32 accumulation", install_rmsnorm_no_upcast),
    Mutation("rmsnorm_eps_wrong", "Use epsilon 1e-5 instead of the checkpoint value 1e-6.", "constant/configuration corruption", install_rmsnorm_eps),
    Mutation("rmsnorm_nminus1", "Divide the squared-norm reduction by n-1.", "reduction denominator off by one", install_rmsnorm_nminus1),
    Mutation("rotary_tail_unrotated", "Leave the final quarter of each rotary head unrotated.", "partial rotary-kernel write", install_rotary_tail_unrotated),
    Mutation("rotary_phase_fp16", "Compute RoPE phase and transcendental inputs in FP16.", "missing phase upcast", install_rotary_phase_fp16),
    Mutation("attention_scale_plus_0p1pct", "Increase every attention scale by 0.1%.", "miscompiled scalar constant", install_attention_scale),
    Mutation("q_projection_bf16_round", "Round the first-layer Q projection through BF16.", "unexpected projection materialization", install_q_proj_round),
    Mutation("attention_output_bf16_round", "Round the first attention block output through BF16.", "unexpected attention materialization", install_attention_output_round),
    Mutation("mlp_gate_bf16_round", "Round the first MLP gate projection through BF16.", "unexpected MLP materialization", install_mlp_gate_round),
    Mutation("mlp_output_bf16_round", "Round the first MLP output through BF16.", "unexpected fused-MLP output cast", install_mlp_output_round),
    Mutation("embedding_bf16_round", "Round token embeddings through BF16 before the decoder.", "wrong embedding output dtype", install_embedding_round),
    Mutation("decoder0_output_bf16_round", "Round decoder layer zero output through BF16.", "wrong residual-stream materialization", install_decoder_output_round),
    Mutation("lm_head_bf16_round", "Round lm_head logits through BF16 before normalization.", "wrong logits materialization", install_lm_head_round),
    Mutation("logsoftmax_fp16", "Evaluate final log-softmax on FP16 logits.", "missing final-logits upcast", install_none, reducer="fp16_logsoftmax"),
    Mutation("logsumexp_chunked_reverse", "Sum stabilized vocabulary exponentials in reversed chunks.", "different final reduction partition/order", install_none, reducer="chunked_reverse"),
]


def selected_mutations(names: list[str] | None) -> list[Mutation]:
    if not names:
        return MUTATIONS
    wanted = set(names)
    unknown = wanted - {mutation.name for mutation in MUTATIONS}
    if unknown:
        raise ValueError(f"unknown mutations: {sorted(unknown)}")
    return [mutation for mutation in MUTATIONS if mutation.name in wanted]


def response_logps(tokenizer: Any, model: Any, config: PathConfig, samples: list[dict[str, Any]], reducer: str) -> tuple[Any, list[int]]:
    import torch

    input_ids, attention_mask, max_prompt = make_batch(tokenizer, samples, config.device)
    response_ids = [[int(value) for value in sample["response_ids"]] for sample in samples]
    with torch.no_grad(), attention_backend_context(config), precision_context(config):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        if reducer == "standard":
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        elif reducer == "fp16_logsoftmax":
            with torch.autocast(device_type="cuda", enabled=False):
                log_probs = torch.nn.functional.log_softmax(logits.to(torch.float16), dim=-1)
            gathered = log_probs.float().gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        elif reducer == "chunked_reverse":
            values = logits.float()
            maximum = values.max(dim=-1, keepdim=True).values
            shifted = torch.exp(values - maximum)
            chunks = list(torch.tensor_split(shifted, 8, dim=-1))
            denominator = sum((chunk.sum(dim=-1) for chunk in reversed(chunks)), torch.zeros_like(maximum[..., 0]))
            log_denominator = maximum[..., 0] + torch.log(denominator)
            target_logits = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            gathered = target_logits - log_denominator
        else:
            raise ValueError(f"unsupported reducer: {reducer}")
    rows = [
        gathered[index, max_prompt - 1 : max_prompt - 1 + len(response)]
        for index, response in enumerate(response_ids)
    ]
    return torch.cat(rows).cpu(), [token for response in response_ids for token in response]


def clip_active(logp: float, old_logp: float, advantage_sign: int, eps: float) -> bool:
    ratio_log = logp - old_logp
    boundary = math.log1p(eps) if advantage_sign > 0 else math.log1p(-eps)
    return ratio_log > boundary if advantage_sign > 0 else ratio_log < boundary


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 9 Refined Artificial Mutation Evaluation",
        "",
        "## Objective",
        "",
        "Evaluate decision-level fork detection on a broader set of executed model mutations spanning normalization, RoPE, attention, MLP, materialization and final vocabulary reduction.",
        "",
        "## Claim Boundary",
        "",
        "These are artificial model-level mutations motivated by compiler/kernel failure mechanisms. They are neither historical bugs nor certified violations of a floating-point contract. A nonzero canary proves that the altered execution reached the measured logprobs; it does not prove realism of the mutation frequency.",
        "",
        "## Controls",
        "",
        f"- Frozen checkpoint and replay batch; `{summary['tokens_per_mutation']}` aligned response tokens per mutation.",
        f"- Baseline maximum mismatch against canonical eager certificates: `{summary['baseline_max_abs_delta_vs_certificate']:.6g}`.",
        f"- Baseline clipping-branch mismatches against canonical certificates: `{summary['baseline_branch_mismatches_vs_certificate']}`.",
        "- Each mutation is independently installed and restored on the same eager model object.",
        "- Mutations with zero observed output delta are marked invalid and excluded from `all_mutation_rows.jsonl`.",
        "",
        "## Results",
        "",
        "| Mutation | Mechanism | Modules | Canary max delta | p99 delta | Branch forks | Fork rate | Valid |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["mutations"]:
        lines.append(
            f"| {row['name']} | {row['mechanism']} | {row['changed_module_count']} | "
            f"{row['max_abs_delta']:.5g} | {row['p99_abs_delta']:.5g} | {row['branch_forks']} | "
            f"{row['fork_rate']:.5g} | {row['valid']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Fork rate measures whether an executed mutation changes the frozen PPO/GRPO clipping branch. Delta magnitude and fork rate remain separate signals: a large mutation can miss all boundaries, while a smaller directed change can cross one.",
            "",
            "Cross-format BF16 round-trip mutations are controlled sensitivity probes on T4 FP16, not claims about native BF16 kernel behavior.",
            "",
            "## Artifacts",
            "",
            "- `results/phase9_mutations/summary.json`",
            "- `results/phase9_mutations/all_mutation_rows.jsonl`",
            "- `results/phase9_mutations/<mutation>.jsonl`",
            "- `scripts/phase9_mutation_catalog.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a catalog of model-level numerical mutations.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--out-dir", default="results/phase9_mutations")
    parser.add_argument("--report", default="reports/phase9_mutation_evaluation.md")
    parser.add_argument("--mutation", action="append", choices=[mutation.name for mutation in MUTATIONS])
    parser.add_argument("--optimizer-step", type=int, default=5, help="Frozen online checkpoint state to evaluate.")
    parser.add_argument("--eps", type=float, default=0.2)
    args = parser.parse_args()

    configure_determinism(0)
    config_data = load_config(args.config)
    config = path_config(config_data, "path_ref")
    certificate_rows = [
        row
        for row in read_jsonl(args.certificates)
        if int(row.get("advantage_sign", 0)) != 0
        and int(row["metadata"]["phase1_metadata"]["online_state"]["optimizer_step"]) == args.optimizer_step
    ]
    if not certificate_rows:
        raise ValueError(f"no canonical certificates for optimizer step {args.optimizer_step}")
    rollout_batches = {
        int(row["metadata"]["phase1_metadata"]["online_state"]["rollout_batch"])
        for row in certificate_rows
    }
    if len(rollout_batches) != 1:
        raise ValueError(f"optimizer step maps to multiple rollout batches: {sorted(rollout_batches)}")
    rollout_batch = next(iter(rollout_batches))
    samples = [
        sample
        for sample in read_jsonl(args.samples)
        if int(sample["metadata"]["rollout_batch"]) == rollout_batch
    ]
    certificates = {
        (str(row["case_id"]), int(row["token_index"])): row
        for row in certificate_rows
    }
    expected_keys = [(str(sample["case_id"]), index) for sample in samples for index in range(len(sample["response_ids"]))]
    missing = [key for key in expected_keys if key not in certificates]
    if missing:
        raise ValueError(f"missing {len(missing)} canonical certificates, first={missing[0]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_hf_path(config)
    try:
        baseline, token_ids = response_logps(tokenizer, model, config, samples, "standard")
        baseline_values = baseline.tolist()
        certificate_values = [float(certificates[key]["logp_ref"]) for key in expected_keys]
        baseline_error = max(abs(a - b) for a, b in zip(baseline_values, certificate_values, strict=True))
        if baseline_error > 1e-5:
            raise ValueError(f"baseline does not reproduce canonical certificates: max_abs_delta={baseline_error}")
        baseline_branch_mismatches = sum(
            clip_active(
                value,
                float(certificates[key]["old_logp"]),
                int(certificates[key]["advantage_sign"]),
                float(certificates[key]["eps"]),
            )
            != bool(certificates[key]["clip_ref"])
            for key, value in zip(expected_keys, baseline_values, strict=True)
        )
        if baseline_branch_mismatches:
            raise ValueError(f"baseline changes {baseline_branch_mismatches} canonical clipping branches")

        all_rows = []
        mutation_summaries = []
        for mutation in selected_mutations(args.mutation):
            with ExitStack() as stack:
                changed = mutation.installer(model, stack)
                if not changed:
                    raise RuntimeError(f"mutation selected no implementation target: {mutation.name}")
                mutated, mutated_ids = response_logps(tokenizer, model, config, samples, mutation.reducer)
            if mutated_ids != token_ids:
                raise ValueError(f"token alignment changed under {mutation.name}")
            values = mutated.tolist()
            deltas = [abs(value - reference) for value, reference in zip(values, baseline_values, strict=True)]
            valid = max(deltas) > 0.0
            rows = []
            for key, token_id, logp_mutated, delta in zip(expected_keys, token_ids, values, deltas, strict=True):
                cert = certificates[key]
                mutated_clip = clip_active(
                    logp_mutated, float(cert["old_logp"]), int(cert["advantage_sign"]), float(cert["eps"])
                )
                row = {
                    "schema_version": "forkcert.mutation_row.v1",
                    "bug": mutation.name,
                    "mutation_description": mutation.description,
                    "mutation_mechanism": mutation.mechanism,
                    "case_id": key[0],
                    "token_index": key[1],
                    "token_id": token_id,
                    "logp_ref": float(cert["logp_ref"]),
                    "logp_mutated": logp_mutated,
                    "logprob_delta": delta,
                    "clip_ref": bool(cert["clip_ref"]),
                    "clip_mutated": mutated_clip,
                    "actual_clip_branch_fork": mutated_clip != bool(cert["clip_ref"]),
                    "advantage_sign": int(cert["advantage_sign"]),
                    "old_logp": float(cert["old_logp"]),
                    "eps": float(cert["eps"]),
                    "mutation_valid": valid,
                }
                rows.append(row)
            write_jsonl(out_dir / f"{mutation.name}.jsonl", rows)
            if valid:
                all_rows.extend(rows)
            forks = sum(row["actual_clip_branch_fork"] for row in rows)
            mutation_summaries.append(
                {
                    "name": mutation.name,
                    "description": mutation.description,
                    "mechanism": mutation.mechanism,
                    "changed_modules": changed,
                    "changed_module_count": len(changed),
                    "valid": valid,
                    "max_abs_delta": max(deltas),
                    "p50_abs_delta": percentile(deltas, 0.50),
                    "p99_abs_delta": percentile(deltas, 0.99),
                    "branch_forks": forks,
                    "fork_rate": forks / len(rows),
                }
            )
        write_jsonl(out_dir / "all_mutation_rows.jsonl", all_rows)
        summary = {
            "schema_version": "forkcert.mutation_catalog.v1",
            "contract": "Frozen eager checkpoint and token IDs; exactly one artificial execution mutation is installed per run.",
            "optimizer_step": args.optimizer_step,
            "rollout_batch": rollout_batch,
            "tokens_per_mutation": len(expected_keys),
            "baseline_max_abs_delta_vs_certificate": baseline_error,
            "baseline_branch_mismatches_vs_certificate": baseline_branch_mismatches,
            "valid_mutations": sum(row["valid"] for row in mutation_summaries),
            "invalid_mutations": sum(not row["valid"] for row in mutation_summaries),
            "mutations": mutation_summaries,
            "claim_scope": "Artificial model-level mutations for empirical oracle evaluation; not historical or certified bugs.",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        Path(args.report).write_text(render_report(summary), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        del model, tokenizer
        cleanup_memory()


if __name__ == "__main__":
    main()
