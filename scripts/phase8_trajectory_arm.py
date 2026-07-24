#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from forkcert.config import load_config
from forkcert.detector import clip_active
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from scripts.phase6_twin_training import batch_response_logps_with_grad, path_config, raw_model, state_tensors
from scripts.phase8_matched_step import select_fork_batch, selected_surrogate_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="One arm of a matched 1/5/20-step fork counterfactual trajectory.")
    parser.add_argument("--arm", choices=["A_reference", "B_alternative", "C_fusion_repair"], required=True)
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--states", default="data/phase6_step5_replay_dump.jsonl")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--repair-max-fusion-size", type=int, default=2)
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

    cert = next(row for row in read_jsonl(args.certificates) if row.get("actual_fork") and row["case_id"] == args.case_id and int(row["token_index"]) == args.token_index)
    samples, states, target = select_fork_batch(read_jsonl(args.samples), read_jsonl(args.states), cert)
    config = load_config(args.config)
    key = "path_ref" if args.arm == "A_reference" else "path_alt"
    cfg = path_config(config, key)
    patch = {"max_fusion_size": args.repair_max_fusion_size} if args.arm == "C_fusion_repair" else None
    load_cfg = replace(cfg, compile_model=False) if patch is not None else cfg
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_inductor_cache"
    if cfg.compile_model:
        torch._dynamo.reset()
        metrics.reset()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
    tokenizer, model = load_hf_path(load_cfg)
    context = inductor_config.patch(patch) if patch is not None else nullcontext()
    context.__enter__()
    if patch is not None:
        model = torch.compile(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=args.lr)
    if cfg.compile_model:
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
        loss, _, _ = selected_surrogate_loss(torch, logps, old, advantages, float(cert["eps"]), target, None)
        target_grad = float(torch.autograd.grad(loss, logps, retain_graph=True)[0][target].item())
        values = logps.detach().float().cpu().tolist()
        branch_count = sum(
            clip_active(value, float(state["old_logp"]), int(state["advantage_sign"]), float(cert["eps"]))
            for value, state in zip(values, states, strict=True)
        )
        loss.backward()
        grad_square = torch.zeros((), dtype=torch.float64, device=logps.device)
        for parameter in parameters:
            if parameter.grad is not None:
                grad_square += parameter.grad.detach().double().square().sum()
        optimizer.step()
        row = {
            "step": step, "loss": float(loss.detach().item()), "full_gradient_norm": float(torch.sqrt(grad_square).item()),
            "target_logp": values[target], "target_loss_gradient": target_grad,
            "target_clip_active": clip_active(values[target], float(states[target]["old_logp"]), int(states[target]["advantage_sign"]), float(cert["eps"])),
            "batch_clip_active_count": branch_count,
        }
        trajectory.append(row)
        if step in checkpoints:
            checkpoint = out_dir / f"step_{step:02d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            raw_model(model).save_pretrained(checkpoint, safe_serialization=True)
    context.__exit__(None, None, None)
    payload = {
        "schema_version": "forkcert.trajectory_arm.v1", "arm": args.arm,
        "fork_id": f"clip-step{cert['metadata']['phase1_metadata']['online_state']['optimizer_step']}-{args.case_id}-t{args.token_index}",
        "path": cfg.name, "inductor_patch": patch, "steps": args.steps, "learning_rate": args.lr,
        "generated_kernel_count": int(metrics.generated_kernel_count) if cfg.compile_model else None,
        "trajectory": trajectory,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "trajectory"}, indent=2, sort_keys=True))
    del optimizer, model, tokenizer
    cleanup_memory()


if __name__ == "__main__":
    main()
