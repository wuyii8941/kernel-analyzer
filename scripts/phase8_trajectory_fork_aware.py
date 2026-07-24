#!/usr/bin/env python
"""
Arm D: compile path with fork-aware gradient mask.

Tokens whose clipping margin is smaller than the empirical compile-eager
delta threshold have their per-token objective detached, so their
gradient does not depend on which side of the boundary compile lands on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from forkcert.config import load_config
from forkcert.detector import clip_active
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from scripts.phase6_twin_training import (
    batch_response_logps_with_grad,
    path_config,
    raw_model,
    state_tensors,
)
from scripts.phase8_matched_step import select_fork_batch


def fork_aware_surrogate_loss(torch, logps, old, advantages, eps: float, margin_threshold: float):
    ratio = torch.exp(logps - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    objectives = torch.minimum(unclipped, clipped)

    log_ratio = logps - old
    pos_boundary = math.log1p(eps)
    neg_boundary = math.log1p(-eps)
    signed_margin = torch.where(
        advantages > 0,
        log_ratio - pos_boundary,
        log_ratio - neg_boundary,
    )
    near_boundary = signed_margin.abs() < margin_threshold
    objectives = torch.where(near_boundary, objectives.detach(), objectives)

    return -objectives.mean(), ratio, clipped_ratio, near_boundary


def main() -> None:
    parser = argparse.ArgumentParser(description="Arm D: compile with fork-aware gradient mask.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--states", default="data/phase6_step5_replay_dump.jsonl")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--margin-threshold", type=float, default=0.015,
                        help="Detach gradient for tokens with |margin| < this value. Default ~delta_p99.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.steps < 20:
        raise ValueError("trajectory must include the pre-registered 20-step endpoint")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_determinism(0)
    import torch
    from torch._inductor import config as inductor_config, metrics

    cert = next(
        row for row in read_jsonl(args.certificates)
        if row.get("actual_fork") and row["case_id"] == args.case_id and int(row["token_index"]) == args.token_index
    )
    samples, states, target = select_fork_batch(read_jsonl(args.samples), read_jsonl(args.states), cert)
    config = load_config(args.config)
    cfg = path_config(config, "path_alt")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_inductor_cache"

    torch._dynamo.reset()
    metrics.reset()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())

    tokenizer, model = load_hf_path(cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=args.lr)

    with torch.no_grad():
        batch_response_logps_with_grad(tokenizer, model, cfg, samples)

    expected_ids = [int(row["token_id"]) for row in states]
    trajectory = []
    checkpoints = {1, 5, 20}

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logps, ids = batch_response_logps_with_grad(tokenizer, model, cfg, samples)
        if ids != expected_ids:
            raise ValueError("trajectory token alignment mismatch")
        old, advantages = state_tensors(torch, states, logps.dtype, logps.device)
        loss, _, _, near_mask = fork_aware_surrogate_loss(
            torch, logps, old, advantages, float(cert["eps"]), args.margin_threshold
        )
        target_grad = float(torch.autograd.grad(loss, logps, retain_graph=True)[0][target].item())
        values = logps.detach().float().cpu().tolist()
        branch_count = sum(
            clip_active(value, float(state["old_logp"]), int(state["advantage_sign"]), float(cert["eps"]))
            for value, state in zip(values, states, strict=True)
        )
        masked_count = int(near_mask.sum().item())
        loss.backward()
        grad_square = torch.zeros((), dtype=torch.float64, device=logps.device)
        for p in parameters:
            if p.grad is not None:
                grad_square += p.grad.detach().double().square().sum()
        optimizer.step()
        row = {
            "step": step,
            "loss": float(loss.detach().item()),
            "full_gradient_norm": float(torch.sqrt(grad_square).item()),
            "target_logp": values[target],
            "target_loss_gradient": target_grad,
            "target_clip_active": clip_active(
                values[target], float(states[target]["old_logp"]),
                int(states[target]["advantage_sign"]), float(cert["eps"]),
            ),
            "batch_clip_active_count": branch_count,
            "fork_aware_masked_count": masked_count,
        }
        trajectory.append(row)
        if step in checkpoints:
            checkpoint = out_dir / f"step_{step:02d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            raw_model(model).save_pretrained(checkpoint, safe_serialization=True)

    payload = {
        "schema_version": "forkcert.trajectory_arm.v1",
        "arm": "D_fork_aware",
        "fork_id": f"clip-step{cert['metadata']['phase1_metadata']['online_state']['optimizer_step']}-{args.case_id}-t{args.token_index}",
        "path": cfg.name,
        "inductor_patch": None,
        "margin_threshold": args.margin_threshold,
        "steps": args.steps,
        "learning_rate": args.lr,
        "generated_kernel_count": int(metrics.generated_kernel_count),
        "trajectory": trajectory,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "trajectory"}, indent=2, sort_keys=True))
    del optimizer, model, tokenizer
    cleanup_memory()


if __name__ == "__main__":
    main()
