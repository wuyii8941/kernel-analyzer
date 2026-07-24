#!/usr/bin/env python
"""Matched one-step A/B/C branch repair for the frozen Qwen GRPO event."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

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
from scripts.phase8_matched_step import selected_surrogate_loss, state_distance


class Audit:
    def __init__(self) -> None:
        self.compiles = 0
        self.invocations = 0
        self.hashes: list[str] = []
        self.nodes: list[int] = []


def tracking_backend(audit: Audit) -> Callable[..., Any]:
    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]):
        audit.compiles += 1
        audit.hashes.append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        audit.nodes.append(sum(1 for _ in graph_module.graph.nodes))
        compiled = inductor(graph_module, example_inputs)

        def counted(*args: Any):
            audit.invocations += 1
            return compiled(*args)

        return counted

    return backend


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_batch(samples: list[dict[str, Any]], states: list[dict[str, Any]], event: dict[str, Any]):
    step = int(event["optimizer_step"])
    rollout = int(event["rollout_batch"])
    selected_states = [
        row for row in states
        if int(row.get("optimizer_step", -1)) == step
        and int(row.get("rollout_batch", -1)) == rollout
        and str(row.get("state")) == "pre_minibatch"
    ]
    cases = {str(row["case_id"]) for row in selected_states}
    selected_samples = [row for row in samples if str(row["case_id"]) in cases]
    state_map = {
        (str(row["case_id"]), int(row["token_index"])): row for row in selected_states
    }
    aligned = [
        state_map[(str(sample["case_id"]), token_index)]
        for sample in selected_samples
        for token_index in range(len(sample["response_ids"]))
    ]
    target = next(
        index for index, row in enumerate(aligned)
        if str(row["case_id"]) == str(event["case_id"])
        and int(row["token_index"]) == int(event["token_index"])
    )
    if len(selected_samples) != 4 or len(aligned) != 512:
        raise ValueError("frozen follow-up requires four 128-token responses")
    return selected_samples, aligned, target


def run_arm(
    name: str,
    cfg: Any,
    samples: list[dict[str, Any]],
    states: list[dict[str, Any]],
    target: int,
    event: dict[str, Any],
    forced_clip: bool | None,
    out_dir: Path,
) -> dict[str, Any]:
    import torch

    configure_determinism(20260720)
    audit = Audit()
    load_cfg = replace(cfg, compile_model=False) if cfg.compile_model else cfg
    tokenizer, model = load_hf_path(load_cfg)
    if cfg.compile_model:
        torch._dynamo.reset()
        model = torch.compile(model, backend=tracking_backend(audit))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=1e-5, momentum=0.0, foreach=False)
    if cfg.compile_model:
        with torch.no_grad():
            batch_response_logps_with_grad(tokenizer, model, cfg, samples)
    optimizer.zero_grad(set_to_none=True)
    before = audit.invocations
    logps, token_ids = batch_response_logps_with_grad(tokenizer, model, cfg, samples)
    scored_invocations = audit.invocations - before
    expected_ids = [int(row["token_id"]) for row in states]
    if token_ids != expected_ids:
        raise ValueError(f"{name}: token alignment mismatch")
    old, advantages = state_tensors(torch, states, logps.dtype, logps.device)
    loss, ratio, clipped_ratio = selected_surrogate_loss(
        torch, logps, old, advantages, 0.2, target, forced_clip
    )
    selected_gradient = float(torch.autograd.grad(loss, logps, retain_graph=True)[0][target])
    loss.backward()
    square = torch.zeros((), dtype=torch.float64, device=logps.device)
    for parameter in parameters:
        if parameter.grad is not None:
            square += parameter.grad.detach().double().square().sum()
    gradient_norm = float(torch.sqrt(square))
    optimizer.step()
    torch.cuda.synchronize()
    arm_dir = out_dir / name
    arm_dir.mkdir(parents=True, exist_ok=False)
    raw_model(model).save_pretrained(arm_dir, safe_serialization=True)
    result = {
        "arm": name,
        "compiled": bool(cfg.compile_model),
        "forced_target_clip": forced_clip,
        "candidate_identity_valid": (not cfg.compile_model) or scored_invocations > 0,
        "compile_audit": {
            "backend_compiles": audit.compiles,
            "runtime_invocations": audit.invocations,
            "scored_invocations": scored_invocations,
            "graph_hashes": audit.hashes,
            "graph_nodes": audit.nodes,
        },
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "target_flat_index": target,
        "target_logp": float(logps[target].detach()),
        "target_old_logp": float(old[target]),
        "target_advantage": float(advantages[target]),
        "target_ratio": float(ratio[target].detach()),
        "target_clipped_ratio": float(clipped_ratio[target].detach()),
        "target_logp_loss_gradient": selected_gradient,
        "weights_dir": str(arm_dir.resolve()),
    }
    del optimizer, model, tokenizer, logps
    gc.collect()
    cleanup_memory()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--reconstruction-online", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    evaluation_path = Path(args.evaluation)
    evaluation = json.loads(evaluation_path.read_text())
    event = evaluation["first_event_for_one_step_followup"]
    samples_path, states_path = Path(args.samples), Path(args.states)
    samples, states, target = select_batch(
        read_jsonl(samples_path), read_jsonl(states_path), event
    )
    reconstruction_rows = read_jsonl(args.reconstruction_online)
    replay = next(
        row for row in reconstruction_rows
        if int(row["optimizer_step"]) == int(event["optimizer_step"])
        and str(row["case_id"]) == str(event["case_id"])
        and int(row["token_index"]) == int(event["token_index"])
    )
    reconstruction_exact = (
        float(replay["old_logp"]) == float(event["old_logp"])
        and float(replay["logp_ref"]) == float(event["logp_ref"])
        and float(replay["logp_alt"]) == float(event["logp_alt"])
        and int(replay["advantage_sign"]) == int(event["advantage_sign"])
        and bool(replay["candidate_identity_valid"])
        and float(replay["delta_self_ref"]) == float(replay["delta_self_alt"]) == 0.0
    )
    if not reconstruction_exact:
        raise RuntimeError("selected state did not reconstruct exactly")
    cfg = load_config(args.config)
    ref_cfg, alt_cfg = path_config(cfg, "path_ref"), path_config(cfg, "path_alt")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    arms = [
        run_arm("A_reference", ref_cfg, samples, states, target, event, None, out_dir),
        run_arm("B_candidate", alt_cfg, samples, states, target, event, None, out_dir),
        run_arm("C_branch_repair", alt_cfg, samples, states, target, event, bool(event["ref_clip"]), out_dir),
    ]
    by_name = {arm["arm"]: arm for arm in arms}
    bc_integrity = (
        by_name["B_candidate"]["candidate_identity_valid"]
        and by_name["C_branch_repair"]["candidate_identity_valid"]
        and by_name["B_candidate"]["target_logp"] == by_name["C_branch_repair"]["target_logp"]
        and by_name["B_candidate"]["target_old_logp"] == by_name["C_branch_repair"]["target_old_logp"]
        and by_name["B_candidate"]["target_advantage"] == by_name["C_branch_repair"]["target_advantage"]
    )
    distances = {
        "A_B": state_distance(out_dir / "A_reference", out_dir / "B_candidate"),
        "A_C": state_distance(out_dir / "A_reference", out_dir / "C_branch_repair"),
        "B_C": state_distance(out_dir / "B_candidate", out_dir / "C_branch_repair"),
    }
    denominator = distances["A_B"]["l2"]
    payload = {
        "schema_version": "forkcert.qwen3-grpo-one-step-branch-repair.v0.1",
        "contract": str(Path(__file__).with_name("QWEN3_GRPO_ONE_STEP_BRANCH_REPAIR_CONTRACT_V0_1_2026-07-17.md").resolve()),
        "event": event,
        "reconstruction_exact": reconstruction_exact,
        "intervention_integrity_valid": bc_integrity,
        "numerical_correctness": "UNINSTANTIATED",
        "arms": arms,
        "distances": distances,
        "residual_ratio_A_C_over_A_B": distances["A_C"]["l2"] / denominator if denominator else None,
        "artifacts": {
            "evaluation_sha256": sha256_file(evaluation_path),
            "samples_sha256": sha256_file(samples_path),
            "states_sha256": sha256_file(states_path),
            "reconstruction_online_sha256": sha256_file(Path(args.reconstruction_online)),
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "reconstruction_exact": reconstruction_exact,
        "intervention_integrity_valid": bc_integrity,
        "distances": distances,
        "residual_ratio": payload["residual_ratio_A_C_over_A_B"],
    }, indent=2))


if __name__ == "__main__":
    main()
