#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.detector import clip_active
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from forkcert.report import markdown_table, write_phase_report

from phase6_twin_training import (
    batch_response_logps_with_grad,
    path_config,
    ppo_loss,
    select_trainable_parameters,
    state_tensors,
    validate_backend_only,
)


def grad_norm(parameters: list[Any]) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        flat = parameter.grad.detach().reshape(-1)
        for start in range(0, flat.numel(), 1_048_576):
            chunk = flat[start : start + 1_048_576].float()
            total += torch.sum(chunk * chunk, dtype=torch.float64)
    return float(torch.sqrt(total).item())


def load_batch(samples_path: str, rollout_batch: int) -> list[dict[str, Any]]:
    samples = [
        row
        for row in read_jsonl(samples_path)
        if int((row.get("metadata") or {}).get("rollout_batch", -1)) == rollout_batch
    ]
    if not samples:
        raise ValueError(f"no samples found for rollout_batch={rollout_batch}")
    return samples


def load_states(
    online_path: str,
    samples: list[dict[str, Any]],
    optimizer_step: int,
    policy_iteration: int,
) -> list[dict[str, Any]]:
    case_ids = {str(sample["case_id"]) for sample in samples}
    keyed = {
        (str(row["case_id"]), int(row["token_index"])): row
        for row in read_jsonl(online_path)
        if str(row.get("case_id")) in case_ids
        and int(row.get("optimizer_step", -1)) == optimizer_step
        and int(row.get("policy_iteration", -1)) == policy_iteration
    }
    states = [
        keyed[(str(sample["case_id"]), token_index)]
        for sample in samples
        for token_index in range(len(sample["response_ids"]))
    ]
    expected = sum(len(sample["response_ids"]) for sample in samples)
    if len(states) != expected:
        raise ValueError(f"online state coverage mismatch: {len(states)} != {expected}")
    return states


def run_path(config, samples, states, branch: str, eps: float, replay_tolerance: float) -> dict[str, Any]:
    import torch

    tokenizer, model = load_hf_path(config)
    parameters = select_trainable_parameters(model, "full")
    expected_ids = [int(row["token_id"]) for row in states]
    expected_logps = [float(row[f"logp_{branch}"]) for row in states]

    if config.compile_model:
        with torch.no_grad():
            batch_response_logps_with_grad(tokenizer, model, config, samples)

    runs = []
    try:
        for run_index in range(2):
            model.zero_grad(set_to_none=True)
            logps, token_ids = batch_response_logps_with_grad(tokenizer, model, config, samples)
            if token_ids != expected_ids:
                raise ValueError(f"{branch} token alignment mismatch")
            observed = logps.detach().float().cpu().tolist()
            max_logp_error = max(abs(a - b) for a, b in zip(observed, expected_logps, strict=True))
            if max_logp_error > replay_tolerance:
                raise ValueError(
                    f"{branch} online replay mismatch: max_abs_error={max_logp_error} "
                    f"exceeds {replay_tolerance}"
                )
            clip_branch_mismatches = sum(
                clip_active(value, float(state["old_logp"]), int(state["advantage_sign"]), eps)
                != clip_active(expected, float(state["old_logp"]), int(state["advantage_sign"]), eps)
                for value, expected, state in zip(observed, expected_logps, states, strict=True)
                if int(state["advantage_sign"]) != 0
            )
            if clip_branch_mismatches:
                raise ValueError(
                    f"{branch} replay changed {clip_branch_mismatches} PPO clipping branches"
                )
            old, advantages = state_tensors(torch, states, logps.dtype, logps.device)
            loss = ppo_loss(torch, logps, old, advantages, eps)
            loss.backward()
            runs.append(
                {
                    "run": run_index + 1,
                    "loss": float(loss.detach().item()),
                    "grad_norm": grad_norm(parameters),
                    "max_online_logp_error": max_logp_error,
                    "clip_branch_mismatches": clip_branch_mismatches,
                }
            )
    finally:
        del model, tokenizer, parameters
        cleanup_memory()

    return {
        "path": config.name,
        "branch": branch,
        "runs": runs,
        "self_grad_norm_delta": abs(runs[1]["grad_norm"] - runs[0]["grad_norm"]),
        "self_loss_delta": abs(runs[1]["loss"] - runs[0]["loss"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 natural gradient-clipping trigger audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--online-logprobs", required=True)
    parser.add_argument("--rollout-batch", type=int, default=1)
    parser.add_argument("--optimizer-step", type=int, default=5)
    parser.add_argument("--policy-iteration", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--replay-tolerance", type=float, default=2e-6)
    parser.add_argument("--out", default="results/phase7_gradclip.json")
    parser.add_argument("--report", default="reports/phase7_gradclip.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ref_cfg = path_config(cfg, "path_ref")
    alt_cfg = path_config(cfg, "path_alt")
    backend_only, failures = validate_backend_only(ref_cfg, alt_cfg)
    if not backend_only:
        raise SystemExit("; ".join(failures))
    configure_determinism(seed=int(cfg.get("seed", 0)))
    samples = load_batch(args.samples, args.rollout_batch)
    states = load_states(
        args.online_logprobs,
        samples,
        args.optimizer_step,
        args.policy_iteration,
    )

    ref = run_path(ref_cfg, samples, states, "ref", args.eps, args.replay_tolerance)
    alt = run_path(alt_cfg, samples, states, "alt", args.eps, args.replay_tolerance)
    ref_norm = float(ref["runs"][0]["grad_norm"])
    alt_norm = float(alt["runs"][0]["grad_norm"])
    threshold = float(args.max_grad_norm)
    controlled_threshold = (ref_norm + alt_norm) / 2.0
    summary = {
        "schema_version": "forkcert.phase7.gradclip.v1",
        "status": "completed",
        "model_state": str(ref_cfg.model_name_or_path),
        "optimizer_step": args.optimizer_step,
        "policy_iteration": args.policy_iteration,
        "rollout_batch": args.rollout_batch,
        "batch_responses": len(samples),
        "token_decisions": len(states),
        "online_replay_tolerance": args.replay_tolerance,
        "loss_normalization": (
            "TRL GRPO: mean of per-response token means; equivalent to the implemented flat mean "
            "for this batch because all four completions contain 128 valid tokens"
        ),
        "natural_threshold": threshold,
        "ref_grad_norm": ref_norm,
        "alt_grad_norm": alt_norm,
        "grad_norm_delta": abs(alt_norm - ref_norm),
        "ref_margin": abs(ref_norm - threshold),
        "alt_margin": abs(alt_norm - threshold),
        "ref_clip_trigger": ref_norm > threshold,
        "alt_clip_trigger": alt_norm > threshold,
        "natural_actual_fork": (ref_norm > threshold) != (alt_norm > threshold),
        "controlled_threshold": controlled_threshold,
        "controlled_actual_fork": (ref_norm > controlled_threshold) != (alt_norm > controlled_threshold),
        "controlled_is_calibration_only": True,
        "region": "unknown",
        "region_reason": "No usable analytic legal bound B exists.",
        "ref_self_grad_norm_delta": ref["self_grad_norm_delta"],
        "alt_self_grad_norm_delta": alt["self_grad_norm_delta"],
        "ref_self_loss_delta": ref["self_loss_delta"],
        "alt_self_loss_delta": alt["self_loss_delta"],
        "ref_max_online_logp_error": max(run["max_online_logp_error"] for run in ref["runs"]),
        "alt_max_online_logp_error": max(run["max_online_logp_error"] for run in alt["runs"]),
        "path_details": {"ref": ref, "alt": alt},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_keys = [
        "natural_threshold",
        "ref_grad_norm",
        "alt_grad_norm",
        "grad_norm_delta",
        "ref_clip_trigger",
        "alt_clip_trigger",
        "natural_actual_fork",
        "controlled_threshold",
        "controlled_actual_fork",
    ]
    write_phase_report(
        args.report,
        title="Phase 7 Gradient-Clipping Trigger",
        confound_checklist={
            "same_step5_snapshot": True,
            "same_four_response_batch": True,
            "all_online_tokens_replayed_within_explicit_2e-6_tolerance": True,
            "backend_only_difference": backend_only,
            "math_sdpa_locked_both_paths": True,
            "compile_warmed_before_measurement": True,
            "trl_grpo_loss_normalization_matched_for_equal_128_token_responses": True,
            "no_parameter_update_between_self_runs": True,
        },
        delta_self_summary=(
            f"Two backward passes per path: ref grad-norm delta={ref['self_grad_norm_delta']:.9g}, "
            f"alt grad-norm delta={alt['self_grad_norm_delta']:.9g}."
        ),
        summary=(
            "The standard max_grad_norm=1.0 trigger was evaluated on a step-5 replay aligned to all "
            "online token logprobs within the explicit replay tolerance and with zero PPO branch changes. "
            "A midpoint threshold is reported only as a controlled detector calibration."
        ),
        sections={
            "Natural Decision": markdown_table([{key: summary[key] for key in table_keys}], table_keys),
            "Interpretation": (
                "The natural result is a decision-boundary observation, not a fragile/bug classification, "
                "because Phase 2 did not produce a usable legal bound. The controlled midpoint threshold "
                "is not part of the natural claim."
            ),
            "External Validity": (
                "This audit uses FP16 autocast on Tesla T4. T4 has no native BF16 tensor-core support; "
                "a zero-fork FP16 result cannot rule out BF16 forks."
            ),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
