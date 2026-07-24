#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    PathConfig,
    attention_backend_context,
    cleanup_memory,
    configure_determinism,
    load_hf_path,
    precision_context,
)
from forkcert.grad import ppo_token_surrogate
from forkcert.report import markdown_table, write_phase_report


def proxy_grad_contrib(cert: dict, branch: str) -> float:
    """Cheap certificate-level proxy until full model autograd is run.

    For PPO clipping, a clipped token has zero gradient contribution from the
    policy-ratio branch; an unclipped token contributes proportionally to
    |advantage|. The certificate stores only advantage sign, so use 1.0 as the
    normalized unclipped contribution.
    """
    active = bool(cert[f"clip_{branch}"])
    return 0.0 if active else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 gradient contribution annotation for fork certificates.")
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--out-jsonl", default="results/phase6_grad_certificates.jsonl")
    parser.add_argument("--report", default="reports/phase6.md")
    parser.add_argument(
        "--proxy-only",
        action="store_true",
        help="Use branch-based normalized proxy instead of full autograd contribution.",
    )
    parser.add_argument("--samples", help="JSONL with prompt/response for full HF autograd mode.")
    parser.add_argument("--config", help="YAML path-pair config for full HF autograd mode.")
    parser.add_argument("--case-id", action="append", help="Restrict annotation to these actual-fork case IDs.")
    parser.add_argument("--token-index", type=int, action="append", help="Optional token-index restriction paired with --case-id.")
    args = parser.parse_args()

    rows = read_jsonl(args.certificates)
    if args.case_id:
        allowed_cases = set(args.case_id)
        allowed_tokens = set(args.token_index or [])
        rows = [
            row
            for row in rows
            if str(row.get("case_id")) in allowed_cases
            and (not allowed_tokens or int(row.get("token_index", -1)) in allowed_tokens)
        ]
    if args.proxy_only:
        annotated = annotate_proxy(rows)
        mode = "branch_proxy"
    else:
        if not args.samples or not args.config:
            raise SystemExit("Full autograd mode requires --samples and --config, or use --proxy-only.")
        annotated = annotate_hf_autograd(rows, args.samples, args.config)
        mode = "hf_autograd"
    write_jsonl(args.out_jsonl, annotated)

    fork_rows = [row for row in annotated if row.get("actual_fork")]
    summary = {
        "n_certificates": len(annotated),
        "n_actual_forks": len(fork_rows),
        "forks_with_nonzero_grad_diff": sum(1 for row in fork_rows if row["grad_contribution_diff"] > 0),
        "mode": mode,
    }
    write_phase_report(
        args.report,
        title="Phase 6 Gradient Contribution",
        confound_checklist={
            "actual_forks_present": bool(fork_rows),
            "full_autograd": mode == "hf_autograd",
            "proxy_matches_clipping_semantics": True,
        },
        delta_self_summary="Uses certificates generated after Phase 1 self-consistency checks.",
        summary="Gradient contribution fields were added to certificates.",
        sections={"Gradient Evidence": markdown_table([summary], list(summary.keys()))},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def annotate_proxy(rows: list[dict]) -> list[dict]:
    annotated = []
    for row in rows:
        item = dict(row)
        ref = proxy_grad_contrib(item, "ref")
        alt = proxy_grad_contrib(item, "alt")
        item["grad_contribution_ref"] = ref
        item["grad_contribution_alt"] = alt
        item["grad_contribution_diff"] = abs(alt - ref)
        item["grad_contribution_mode"] = "branch_proxy"
        annotated.append(item)
    return annotated


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


def batch_token_logprob_with_grad(tokenizer, model, config: PathConfig, samples: list[dict], cert: dict):
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
    target_batch = next(index for index, sample in enumerate(samples) if str(sample["case_id"]) == str(cert["case_id"]))
    full_position = max_prompt + int(cert["token_index"])
    with attention_backend_context(config), precision_context(config):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        if config.logits_upcast_fp32:
            logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits[target_batch, full_position - 1, :], dim=-1)
    return log_probs[input_ids[target_batch, full_position]]


def grad_norm_for_case(tokenizer, model, config: PathConfig, samples: list[dict], cert: dict, branch: str) -> float:
    import torch

    model.zero_grad(set_to_none=True)
    if config.compile_model:
        # Match the measured A4 protocol: compiled outputs are used only after
        # the graph has reached a warm state.
        with torch.no_grad():
            batch_token_logprob_with_grad(tokenizer, model, config, samples, cert)
    logp = batch_token_logprob_with_grad(tokenizer, model, config, samples, cert)
    expected_logp = float(cert[f"logp_{branch}"])
    observed_logp = float(logp.detach().item())
    if abs(observed_logp - expected_logp) > 1e-6:
        raise ValueError(
            f"snapshot logp mismatch for {branch}: observed={observed_logp}, certificate={expected_logp}"
        )
    old_logp = torch.tensor(float(cert["old_logp"]), dtype=logp.dtype, device=logp.device)
    rollout_advantage = (cert.get("metadata") or {}).get("rollout_advantage")
    advantage_value = float(rollout_advantage) if rollout_advantage is not None else float(cert["advantage_sign"])
    advantage = torch.tensor(advantage_value, dtype=logp.dtype, device=logp.device)
    loss = -ppo_token_surrogate(logp, old_logp, advantage, float(cert["eps"]))
    loss.backward()
    total = torch.zeros((), dtype=torch.float32, device=logp.device)
    for param in model.parameters():
        if param.grad is not None:
            total = total + torch.sum(param.grad.detach().float() ** 2)
    value = float(torch.sqrt(total).item())
    model.zero_grad(set_to_none=True)
    return value


def annotate_hf_autograd(rows: list[dict], samples_path: str, config_path: str) -> list[dict]:
    from forkcert.config import load_config

    cfg = load_config(config_path)
    samples = {str(row["case_id"]): row for row in read_jsonl(samples_path)}
    samples_by_rollout: dict[int, list[dict]] = {}
    for sample in samples.values():
        rollout = int((sample.get("metadata") or {}).get("rollout_batch", -1))
        samples_by_rollout.setdefault(rollout, []).append(sample)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    ref_cfg = path_config(cfg, "path_ref")
    alt_cfg = path_config(cfg, "path_alt")
    annotated = [dict(row) for row in rows]
    fork_indices = [index for index, row in enumerate(annotated) if row.get("actual_fork")]
    for item in annotated:
        item["grad_contribution_mode"] = "hf_autograd" if item.get("actual_fork") else "not_applicable"
    if not fork_indices:
        return annotated

    ref_tokenizer, ref_model = load_hf_path(ref_cfg)
    ref_values: dict[int, float] = {}
    for index in fork_indices:
        item = annotated[index]
        sample = samples.get(str(item["case_id"]))
        if sample is None:
            raise KeyError(f"missing prompt/response sample for actual fork case_id={item['case_id']}")
        rollout = int((sample.get("metadata") or {}).get("rollout_batch", -1))
        ref_values[index] = grad_norm_for_case(
            ref_tokenizer, ref_model, ref_cfg, samples_by_rollout[rollout], item, "ref"
        )
    del ref_model
    del ref_tokenizer
    cleanup_memory()

    alt_tokenizer, alt_model = load_hf_path(alt_cfg)
    alt_values: dict[int, float] = {}
    for index in fork_indices:
        item = annotated[index]
        sample = samples.get(str(item["case_id"]))
        if sample is None:
            raise KeyError(f"missing prompt/response sample for actual fork case_id={item['case_id']}")
        rollout = int((sample.get("metadata") or {}).get("rollout_batch", -1))
        alt_values[index] = grad_norm_for_case(
            alt_tokenizer, alt_model, alt_cfg, samples_by_rollout[rollout], item, "alt"
        )
    del alt_model
    del alt_tokenizer
    cleanup_memory()

    for index in fork_indices:
        item = annotated[index]
        ref = ref_values[index]
        alt = alt_values[index]
        item["grad_contribution_ref"] = ref
        item["grad_contribution_alt"] = alt
        item["grad_contribution_diff"] = abs(alt - ref)
    return annotated


if __name__ == "__main__":
    main()
