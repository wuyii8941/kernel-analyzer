#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.detector import clip_active
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
from forkcert.report import markdown_table, write_phase_report
from forkcert.twin_train import TwinStep, trajectory_summary


def path_config(data: dict[str, Any], key: str) -> PathConfig:
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
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
        rmsnorm_reference=item.get("rmsnorm_reference", False),
        rmsnorm_no_upcast=item.get("rmsnorm_no_upcast", False),
        rmsnorm_compile=item.get("rmsnorm_compile", False),
        materialize_bf16_outputs=item.get("materialize_bf16_outputs", False),
        materialization_dtype=item.get("materialization_dtype"),
        allow_bf16_reduced_precision_reduction=item.get("allow_bf16_reduced_precision_reduction"),
        allow_fp16_reduced_precision_reduction=item.get("allow_fp16_reduced_precision_reduction"),
        model_training_mode=item.get("model_training_mode", False),
        gradient_checkpointing=item.get("gradient_checkpointing", False),
    )


def validate_backend_only(ref: PathConfig, alt: PathConfig) -> tuple[bool, list[str]]:
    allowed = {"name", "compile_model", "attn_implementation", "attention_backend"}
    ref_data = asdict(ref)
    alt_data = asdict(alt)
    unexpected = [key for key in ref_data if key not in allowed and ref_data[key] != alt_data[key]]
    backend_changed = any(
        ref_data[key] != alt_data[key]
        for key in ["compile_model", "attn_implementation", "attention_backend"]
    )
    failures = [f"unexpected path difference: {key}" for key in unexpected]
    if not backend_changed:
        failures.append("no backend variable differs between twin paths")
    return not failures, failures


def raw_model(model):
    return getattr(model, "_orig_mod", model)


def select_trainable_parameters(model, scope: str) -> list[Any]:
    if scope == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    if scope != "lm_head":
        raise ValueError(f"unsupported trainable scope: {scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = raw_model(model).get_output_embeddings()
    if head is None:
        raise ValueError("model has no output embedding head")
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in head.parameters() if parameter.requires_grad]


def response_logps_with_grad(tokenizer, model, config: PathConfig, sample: dict[str, Any]):
    import torch

    encoded = _encode_sample(tokenizer, sample, config.device)
    input_ids = encoded["input_ids"]
    prompt_len = encoded["prompt_len"]
    with attention_backend_context(config), precision_context(config):
        logits = model(input_ids=input_ids).logits
        if config.logits_upcast_fp32:
            logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return gathered[0, prompt_len - 1 :], encoded["response_ids"]


def batch_response_logps_with_grad(tokenizer, model, config: PathConfig, samples: list[dict[str, Any]]):
    import torch

    prompt_ids = [[int(value) for value in sample["prompt_ids"]] for sample in samples]
    response_ids = [[int(value) for value in sample["response_ids"]] for sample in samples]
    max_prompt = max(len(values) for values in prompt_ids)
    max_response = max(len(values) for values in response_ids)
    pad_id = int(tokenizer.pad_token_id)
    batch = []
    masks = []
    for prompt, response in zip(prompt_ids, response_ids, strict=True):
        left = max_prompt - len(prompt)
        right = max_response - len(response)
        batch.append([pad_id] * left + prompt + response + [pad_id] * right)
        masks.append([0] * left + [1] * (len(prompt) + len(response)) + [0] * right)
    input_ids = torch.tensor(batch, dtype=torch.long, device=config.device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=config.device)
    with attention_backend_context(config), precision_context(config):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        if config.logits_upcast_fp32:
            logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        gathered = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    rows = [
        gathered[index, max_prompt - 1 : max_prompt - 1 + len(response)]
        for index, response in enumerate(response_ids)
    ]
    return torch.cat(rows), [token for response in response_ids for token in response]


def state_tensors(torch, states: list[dict[str, Any]], dtype, device):
    old = torch.tensor([float(row["old_logp"]) for row in states], dtype=dtype, device=device)
    advantages = torch.tensor([float(row["advantage"]) for row in states], dtype=dtype, device=device)
    return old, advantages


def ppo_loss(torch, logps, old, advantages, eps: float):
    ratio = torch.exp(logps - old)
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()


def trainable_divergence(torch, ref_parameters: list[Any], alt_parameters: list[Any]) -> tuple[float, float]:
    if len(ref_parameters) != len(alt_parameters):
        raise ValueError("twin trainable parameter counts differ")
    diff_sq = torch.zeros((), dtype=torch.float64, device=ref_parameters[0].device)
    ref_sq = torch.zeros_like(diff_sq)
    for ref, alt in zip(ref_parameters, alt_parameters, strict=True):
        if ref.shape != alt.shape:
            raise ValueError(f"twin trainable parameter shape mismatch: {ref.shape} vs {alt.shape}")
        ref_flat = ref.detach().reshape(-1)
        alt_flat = alt.detach().reshape(-1)
        for start in range(0, ref_flat.numel(), 1_048_576):
            ref_chunk = ref_flat[start : start + 1_048_576].float()
            alt_chunk = alt_flat[start : start + 1_048_576].float()
            diff = ref_chunk - alt_chunk
            diff_sq += torch.sum(diff * diff, dtype=torch.float64)
            ref_sq += torch.sum(ref_chunk * ref_chunk, dtype=torch.float64)
    divergence = float(torch.sqrt(diff_sq).item())
    ref_norm = float(torch.sqrt(ref_sq).item())
    return divergence, divergence / ref_norm if ref_norm > 0 else 0.0


def write_not_triggered(out_jsonl: str, out_summary: str, report: str) -> None:
    write_jsonl(out_jsonl, [])
    summary = {
        "status": "not_triggered",
        "reason": "Phase 4 contains no natural actual_fork; twin training is conditional by design.",
        "backend_only_difference": None,
        "exact_weight_divergence": None,
    }
    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_phase_report(
        report,
        title="Phase 6 Twin Training",
        confound_checklist={"natural_fork_trigger": False},
        delta_self_summary="Phase 1 self-consistency gate remains authoritative.",
        summary=summary["reason"],
        sections={"Summary": markdown_table([summary], list(summary.keys()))},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lockstep backend-only twin PPO training after natural forks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--rollout-jsonl", required=True)
    parser.add_argument("--phase4-certificates", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--measure-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument(
        "--trainable-scope",
        choices=["full", "lm_head"],
        default="full",
        help="Canonical evidence requires full; lm_head is a lower-memory debug mode.",
    )
    parser.add_argument("--out-jsonl", default="results/phase6_twin_trajectory.jsonl")
    parser.add_argument("--out-summary", default="results/phase6_twin_summary.json")
    parser.add_argument("--report", default="reports/phase6_twin.md")
    args = parser.parse_args()

    phase4 = read_jsonl(args.phase4_certificates)
    if not any(row.get("actual_fork") for row in phase4):
        write_not_triggered(args.out_jsonl, args.out_summary, args.report)
        print(json.dumps({"status": "not_triggered"}, indent=2))
        return
    if args.steps < 100:
        raise SystemExit("Twin training requires at least 100 optimizer steps for the coupling experiment.")
    if args.measure_every <= 0:
        raise SystemExit("--measure-every must be positive")

    import torch

    cfg = load_config(args.config)
    ref_cfg = path_config(cfg, "path_ref")
    alt_cfg = path_config(cfg, "path_alt")
    backend_only, backend_failures = validate_backend_only(ref_cfg, alt_cfg)
    if not backend_only:
        raise SystemExit("; ".join(backend_failures))
    configure_determinism(seed=int(cfg.get("seed", 0)))
    samples = read_jsonl(args.samples)
    fork_case_ids = {str(row["case_id"]) for row in phase4 if row.get("actual_fork")}
    rollout_groups: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        rollout_batch = int((sample.get("metadata") or {}).get("rollout_batch", -1))
        rollout_groups.setdefault(rollout_batch, []).append(sample)
    groups = list(rollout_groups.values())
    groups.sort(
        key=lambda group: (
            not any(str(sample["case_id"]) in fork_case_ids for sample in group),
            int((group[0].get("metadata") or {}).get("rollout_batch", -1)),
        )
    )
    rollout = {(str(row["case_id"]), int(row["token_index"])): row for row in read_jsonl(args.rollout_jsonl)}

    ref_tokenizer, ref_model = load_hf_path(ref_cfg)
    ref_parameters = select_trainable_parameters(ref_model, args.trainable_scope)
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in ref_model.parameters())
    trainable_bytes = sum(parameter.numel() * parameter.element_size() for parameter in ref_parameters)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    compile_reserve = 2 * 1024**3 if alt_cfg.compile_model else 512 * 1024**2
    required_increment = model_bytes + 2 * trainable_bytes + compile_reserve
    if free_bytes < required_increment:
        del ref_model, ref_tokenizer
        cleanup_memory()
        raise RuntimeError(
            "insufficient free GPU memory for lockstep twins before loading path_alt: "
            f"free={free_bytes}, estimated_increment={required_increment}, total={total_bytes}. "
            "Use a larger GPU or reduce the trainable scope; do not substitute a sequential approximation."
        )
    alt_tokenizer, alt_model = load_hf_path(alt_cfg)
    alt_parameters = select_trainable_parameters(alt_model, args.trainable_scope)
    ref_optimizer = torch.optim.SGD(ref_parameters, lr=args.lr)
    alt_optimizer = torch.optim.SGD(alt_parameters, lr=args.lr)
    initial_divergence, initial_relative = trainable_divergence(torch, ref_parameters, alt_parameters)
    if initial_divergence != 0.0:
        raise ValueError(f"twins do not start from identical trainable weights: {initial_divergence}")

    trajectory = [
        TwinStep(0, "initial", 0, 0.0, initial_divergence, initial_relative, False).to_json_dict()
    ]
    interval_had_fork = False
    with torch.no_grad():
        batch_response_logps_with_grad(ref_tokenizer, ref_model, ref_cfg, groups[0])
        batch_response_logps_with_grad(alt_tokenizer, alt_model, alt_cfg, groups[0])
    try:
        for step in range(1, args.steps + 1):
            group = groups[(step - 1) % len(groups)]
            states = [
                rollout[(str(sample["case_id"]), token_index)]
                for sample in group
                for token_index in range(len(sample.get("response_ids", [])))
            ]
            if not states:
                raise ValueError("rollout group has no aligned token states")

            ref_optimizer.zero_grad(set_to_none=True)
            ref_logps, ref_ids = batch_response_logps_with_grad(ref_tokenizer, ref_model, ref_cfg, group)
            old, advantages = state_tensors(torch, states, ref_logps.dtype, ref_logps.device)
            ref_loss = ppo_loss(torch, ref_logps, old, advantages, args.eps)
            ref_values = ref_logps.detach().float().cpu().tolist()
            ref_loss.backward()
            ref_optimizer.step()

            alt_optimizer.zero_grad(set_to_none=True)
            alt_logps, alt_ids = batch_response_logps_with_grad(alt_tokenizer, alt_model, alt_cfg, group)
            if ref_ids != alt_ids or ref_ids != [int(row["token_id"]) for row in states]:
                raise ValueError("twin token alignment mismatch in rollout batch")
            old_alt, advantages_alt = state_tensors(torch, states, alt_logps.dtype, alt_logps.device)
            alt_loss = ppo_loss(torch, alt_logps, old_alt, advantages_alt, args.eps)
            alt_values = alt_logps.detach().float().cpu().tolist()
            alt_loss.backward()
            alt_optimizer.step()

            fork_count = 0
            max_delta = 0.0
            for ref_logp, alt_logp, state in zip(ref_values, alt_values, states, strict=True):
                sign = int(state["advantage_sign"])
                max_delta = max(max_delta, abs(alt_logp - ref_logp))
                if sign == 0:
                    continue
                if clip_active(ref_logp, float(state["old_logp"]), sign, args.eps) != clip_active(
                    alt_logp, float(state["old_logp"]), sign, args.eps
                ):
                    fork_count += 1
            interval_had_fork = interval_had_fork or fork_count > 0
            divergence = relative = None
            measured_interval = None
            if step % args.measure_every == 0 or step == args.steps:
                divergence, relative = trainable_divergence(torch, ref_parameters, alt_parameters)
                measured_interval = interval_had_fork
                interval_had_fork = False
            trajectory.append(
                TwinStep(
                    step,
                    ",".join(str(sample["case_id"]) for sample in group),
                    fork_count,
                    max_delta,
                    divergence,
                    relative,
                    measured_interval,
                ).to_json_dict()
            )
    finally:
        del ref_optimizer, alt_optimizer, ref_model, alt_model, ref_tokenizer, alt_tokenizer
        cleanup_memory()

    write_jsonl(args.out_jsonl, trajectory)
    summary = {
        "status": "completed",
        "backend_only_difference": True,
        "exact_weight_divergence": True,
        "weight_scope": "full_model" if args.trainable_scope == "full" else "lm_head_debug",
        "optimizer": "SGD",
        "learning_rate": args.lr,
        "measure_every": args.measure_every,
        "model_parameter_bytes_each": model_bytes,
        "trainable_parameter_bytes_each": trainable_bytes,
        "pre_alt_free_gpu_bytes": free_bytes,
        "estimated_increment_gpu_bytes": required_increment,
        "path_ref": ref_cfg.name,
        "path_alt": alt_cfg.name,
        **trajectory_summary(trajectory),
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_phase_report(
        args.report,
        title="Phase 6 Twin Training",
        confound_checklist={
            "same_initial_weights": initial_divergence == 0.0,
            "same_seed_data_optimizer": True,
            "backend_only_difference": True,
            "fixed_token_alignment": True,
            "exact_trainable_weight_divergence": True,
        },
        delta_self_summary="Phase 1 self-consistency gate remains authoritative for backend attribution.",
        summary="Backend-only twins ran in lockstep; fork timestamps and exact trainable-weight divergence were recorded.",
        sections={"Summary": markdown_table([summary], list(summary.keys()))},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
