#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import PathConfig, cleanup_memory, configure_determinism, load_hf_path
from scripts.phase6_twin_training import (
    batch_response_logps_with_grad,
    path_config,
    raw_model,
    state_tensors,
)


def selected_surrogate_loss(torch: Any, logps: Any, old: Any, advantages: Any, eps: float, target: int, forced_clip: bool | None):
    ratio = torch.exp(logps - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    objectives = torch.minimum(unclipped, clipped)
    if forced_clip is not None:
        forced = clipped[target] if forced_clip else unclipped[target]
        objectives = torch.cat((objectives[:target], forced.unsqueeze(0), objectives[target + 1 :]))
    return -objectives.mean(), ratio, clipped_ratio


def select_fork_batch(samples: list[dict[str, Any]], states: list[dict[str, Any]], cert: dict[str, Any]):
    sample_by_case = {str(row["case_id"]): row for row in samples}
    online_state = cert["metadata"]["phase1_metadata"]["online_state"]
    selected_states = [
        row for row in states
        if int(row.get("optimizer_step", -1)) == int(online_state["optimizer_step"])
        and row.get("state") == "pre_minibatch"
        and int(row.get("rollout_batch", -1)) == int(online_state["rollout_batch"])
    ]
    selected_states.sort(key=lambda row: (str(row["case_id"]), int(row["token_index"])))
    cases = {str(row["case_id"]) for row in selected_states}
    selected_samples = [row for row in samples if str(row["case_id"]) in cases]
    # The model batch order defines the flattened token order. Rebuild states in
    # exactly that order instead of relying on lexical sorting above.
    state_map = {(str(row["case_id"]), int(row["token_index"])): row for row in selected_states}
    aligned = [
        state_map[(str(sample["case_id"]), token_index)]
        for sample in selected_samples
        for token_index in range(len(sample["response_ids"]))
    ]
    target = next(
        index for index, row in enumerate(aligned)
        if str(row["case_id"]) == str(cert["case_id"]) and int(row["token_index"]) == int(cert["token_index"])
    )
    return selected_samples, aligned, target


def run_arm(name: str, cfg: PathConfig, samples: list[dict[str, Any]], states: list[dict[str, Any]], target: int, cert: dict[str, Any], forced_clip: bool | None, lr: float, out_dir: Path, inductor_patch: dict[str, Any] | None = None):
    import torch

    configure_determinism(0)
    cache_dir = None
    if cfg.compile_model:
        torch._dynamo.reset()
        cache_dir = out_dir / "_inductor_cache" / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
        from torch._inductor import metrics
        metrics.reset()
    load_cfg = replace(cfg, compile_model=False) if inductor_patch is not None and cfg.compile_model else cfg
    tokenizer, model = load_hf_path(load_cfg)
    patch_context = nullcontext()
    if inductor_patch is not None and cfg.compile_model:
        from torch._inductor import config as inductor_config
        patch_context = inductor_config.patch(inductor_patch)
    patch_context.__enter__()
    if inductor_patch is not None and cfg.compile_model:
        model = torch.compile(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=lr)
    if cfg.compile_model:
        with torch.no_grad():
            batch_response_logps_with_grad(tokenizer, model, cfg, samples)
    optimizer.zero_grad(set_to_none=True)
    logps, token_ids = batch_response_logps_with_grad(tokenizer, model, cfg, samples)
    expected_ids = [int(row["token_id"]) for row in states]
    if token_ids != expected_ids:
        raise ValueError(f"{name}: token alignment mismatch")
    old, advantages = state_tensors(torch, states, logps.dtype, logps.device)
    loss, ratio, clipped_ratio = selected_surrogate_loss(torch, logps, old, advantages, float(cert["eps"]), target, forced_clip)
    target_logp_grad = float(torch.autograd.grad(loss, logps, retain_graph=True)[0][target].item())
    loss.backward()
    grad_square = torch.zeros((), dtype=torch.float64, device=logps.device)
    for parameter in parameters:
        if parameter.grad is not None:
            grad_square += parameter.grad.detach().double().square().sum()
    grad_norm = float(torch.sqrt(grad_square).item())
    optimizer.step()
    patch_context.__exit__(None, None, None)
    arm_dir = out_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    raw_model(model).save_pretrained(arm_dir, safe_serialization=True)
    result = {
        "arm": name,
        "path": cfg.name,
        "compile_model": cfg.compile_model,
        "forced_target_clip_branch": forced_clip,
        "inductor_patch": inductor_patch,
        "loss": float(loss.detach().item()),
        "batch_tokens": int(logps.numel()),
        "target_flat_index": target,
        "target_logp": float(logps[target].detach().item()),
        "target_old_logp": float(old[target].item()),
        "target_advantage": float(advantages[target].item()),
        "target_ratio": float(ratio[target].detach().item()),
        "target_clipped_ratio": float(clipped_ratio[target].detach().item()),
        "target_logp_loss_gradient": target_logp_grad,
        "full_model_gradient_norm": grad_norm,
        "weights_dir": str(arm_dir),
        "inductor_cache_dir": str(cache_dir) if cache_dir is not None else None,
        "generated_kernel_count": int(metrics.generated_kernel_count) if cfg.compile_model else None,
    }
    del optimizer, model, tokenizer
    cleanup_memory()
    return result


def weight_files(directory: Path) -> list[Path]:
    index = directory / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text(encoding="utf-8"))
        return [directory / name for name in sorted(set(payload["weight_map"].values()))]
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {directory}")
    return files


def state_index(directory: Path) -> dict[str, Path]:
    from safetensors import safe_open
    result = {}
    for path in weight_files(directory):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                result[key] = path
    return result


def state_distance(left: Path, right: Path) -> dict[str, float]:
    import torch
    from safetensors import safe_open

    li, ri = state_index(left), state_index(right)
    if li.keys() != ri.keys():
        raise ValueError("saved state keys differ")
    diff_sq = 0.0
    left_sq = 0.0
    for key in sorted(li):
        with safe_open(li[key], framework="pt", device="cpu") as lh, safe_open(ri[key], framework="pt", device="cpu") as rh:
            a, b = lh.get_tensor(key).float(), rh.get_tensor(key).float()
        diff_sq += float(torch.sum((a - b).double().square()).item())
        left_sq += float(torch.sum(a.double().square()).item())
    distance = math.sqrt(diff_sq)
    norm = math.sqrt(left_sq)
    return {"l2": distance, "relative_l2": distance / norm if norm else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B/C matched-step counterfactual for one natural clipping fork.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--states", default="data/phase6_step5_replay_dump.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out-dir", default="results/matched_step/clip-step5-grpo_000001_2817771126c0-t80")
    parser.add_argument("--out", default="results/matched_step/clip-step5-grpo_000001_2817771126c0-t80.json")
    parser.add_argument("--repair-max-fusion-size", type=int, help="Use a real compiled fusion-partition repair for arm C instead of forcing the branch.")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    cfg_data = load_config(args.config)
    cert = next(row for row in read_jsonl(args.certificates) if row.get("actual_fork") and str(row["case_id"]) == args.case_id and int(row["token_index"]) == args.token_index)
    samples, states, target = select_fork_batch(read_jsonl(args.samples), read_jsonl(args.states), cert)
    ref_cfg = path_config(cfg_data, "path_ref")
    alt_cfg = path_config(cfg_data, "path_alt")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion_patch = {"max_fusion_size": args.repair_max_fusion_size} if args.repair_max_fusion_size is not None else None
    arms = [
        run_arm("A_reference", ref_cfg, samples, states, target, cert, None, args.lr, out_dir),
        run_arm("B_alternative", alt_cfg, samples, states, target, cert, None, args.lr, out_dir),
        run_arm(
            "C_fusion_repair" if fusion_patch else "C_branch_repair",
            alt_cfg, samples, states, target, cert,
            None if fusion_patch else bool(cert["clip_ref"]), args.lr, out_dir, fusion_patch,
        ),
    ]
    c_name = "C_fusion_repair" if fusion_patch else "C_branch_repair"
    distances = {
        "A_B": state_distance(out_dir / "A_reference", out_dir / "B_alternative"),
        "A_C": state_distance(out_dir / "A_reference", out_dir / c_name),
        "B_C": state_distance(out_dir / "B_alternative", out_dir / c_name),
    }
    ab = distances["A_B"]["l2"]
    payload = {
        "schema_version": "forkcert.matched_step.v1",
        "fork_id": f"clip-step{cert['metadata']['phase1_metadata']['online_state']['optimizer_step']}-{args.case_id}-t{args.token_index}",
        "intervention": (
            f"C preserves torch.compile and changes only Inductor max_fusion_size to {args.repair_max_fusion_size}."
            if fusion_patch else
            "C preserves alternative numeric path and forces only the target token clipping branch to the audited reference branch."
        ),
        "controls": {"same_checkpoint": True, "same_batch": True, "same_token_ids": True, "same_old_logp_advantage": True, "same_fresh_sgd_state": True, "seed": 0, "learning_rate": args.lr},
        "arms": arms,
        "distances": distances,
        "causal_recovery_ratio_A_C_over_A_B": distances["A_C"]["l2"] / ab if ab else None,
        "interpretation_scope": (
            "Fusion-partition causal effect; max_fusion_size identifies a responsible compile optimization class, not a unique source operator."
            if fusion_patch else
            "Decision-branch causal effect; this does not identify the responsible compiled operator."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "distances": distances, "recovery_ratio": payload["causal_recovery_ratio_A_C_over_A_B"]}, indent=2))


if __name__ == "__main__":
    main()
