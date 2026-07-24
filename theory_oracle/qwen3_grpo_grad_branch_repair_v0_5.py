#!/usr/bin/env python
"""Controlled one-step branch repair for the frozen grad-context Qwen event."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from accelerate.utils.operations import convert_outputs_to_fp32

from forkcert.config import load_config
from forkcert.detector import clip_active
from forkcert.io import read_jsonl
from forkcert.logprob_runner import configure_determinism, load_hf_path
from scripts.phase6_twin_training import path_config, raw_model, state_tensors
from theory_oracle.qwen3_grpo_branch_repair_oracle import Audit, select_batch, tracking_backend
from theory_oracle.qwen3_grpo_branch_repair_oracle_v0_2 import (
    trainer_response_logps_with_grad,
)


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def branch_loss(
    torch: Any,
    logps: Any,
    old: Any,
    advantages: Any,
    epsilon: float,
    target: int,
    forced_reference_clip: bool | None,
) -> tuple[Any, Any, Any]:
    ratio = torch.exp(logps - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    objectives = torch.minimum(unclipped, clipped)
    if forced_reference_clip is not None:
        if forced_reference_clip:
            boundary = 1.0 + epsilon if float(advantages[target]) > 0.0 else 1.0 - epsilon
            replacement = logps[target] * 0.0 + boundary * advantages[target]
        else:
            replacement = unclipped[target]
        objectives = torch.cat(
            (objectives[:target], replacement.unsqueeze(0), objectives[target + 1 :])
        )
    return -objectives.mean(), ratio, clipped_ratio


def run_arm(
    *,
    name: str,
    cfg: Any,
    samples: list[dict[str, Any]],
    states: list[dict[str, Any]],
    target: int,
    event: dict[str, Any],
    expected_tensor_hash: str,
    forced_reference_clip: bool | None,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    configure_determinism(20260720)
    audit = Audit()
    tokenizer, model = load_hf_path(replace(cfg, compile_model=False))
    model.forward = convert_outputs_to_fp32(model.forward)
    if cfg.compile_model:
        torch._dynamo.reset()
        model = torch.compile(model, backend=tracking_backend(audit))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=1e-5, momentum=0.0, foreach=False)

    call_hashes: list[str] = []
    call_invocations: list[int] = []
    token_ids = None
    logps = None
    call_count = 3 if cfg.compile_model else 2
    for call_index in range(call_count):
        before = audit.invocations
        current, current_ids = trainer_response_logps_with_grad(tokenizer, model, cfg, samples)
        call_invocations.append(audit.invocations - before)
        call_hashes.append(tensor_sha256(current))
        if call_index + 1 == call_count:
            logps, token_ids = current, current_ids
        else:
            del current
    if logps is None or token_ids is None:
        raise RuntimeError(f"{name}: scorer produced no measured tensor")
    expected_ids = [int(row["token_id"]) for row in states]
    if token_ids != expected_ids:
        raise RuntimeError(f"{name}: token alignment mismatch")
    if any(value != expected_tensor_hash for value in call_hashes):
        raise RuntimeError(
            f"{name}: full scorer hash mismatch {call_hashes} != {expected_tensor_hash}"
        )
    expected_logp = float(event["logp_ref"] if name == "A_reference" else event["logp_alt"])
    target_logp = float(logps[target].detach())
    if target_logp != expected_logp:
        raise RuntimeError(f"{name}: target anchor mismatch {target_logp} != {expected_logp}")

    old, advantages = state_tensors(torch, states, logps.dtype, logps.device)
    optimizer.zero_grad(set_to_none=True)
    loss, ratio, clipped_ratio = branch_loss(
        torch, logps, old, advantages, 0.2, target, forced_reference_clip
    )
    target_gradient = float(torch.autograd.grad(loss, logps, retain_graph=True)[0][target])
    loss.backward()
    square = torch.zeros((), dtype=torch.float64, device=logps.device)
    for parameter in parameters:
        if parameter.grad is not None:
            square += parameter.grad.detach().double().square().sum()
    gradient_norm = float(torch.sqrt(square))
    optimizer.step()
    torch.cuda.synchronize()

    arm_dir = output_dir / name
    arm_dir.mkdir(parents=True, exist_ok=False)
    raw_model(model).save_pretrained(arm_dir, safe_serialization=True)
    sign = int(event["advantage_sign"])
    selected_clip = clip_active(target_logp, float(old[target]), sign, 0.2)
    result = {
        "arm": name,
        "compiled": bool(cfg.compile_model),
        "forced_reference_clip": forced_reference_clip,
        "candidate_identity_valid": (not cfg.compile_model)
        or all(value > 0 for value in call_invocations),
        "compile_audit": {
            "backend_compiles": audit.compiles,
            "runtime_invocations": audit.invocations,
            "per_call_runtime_invocations": call_invocations,
            "graph_hashes": audit.hashes,
            "graph_nodes": audit.nodes,
        },
        "scorer_call_sha256": call_hashes,
        "scorer_self_exact": len(set(call_hashes)) == 1,
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "target_flat_index": target,
        "target_logp": target_logp,
        "target_old_logp": float(old[target]),
        "target_advantage": float(advantages[target]),
        "target_ratio": float(ratio[target].detach()),
        "target_clipped_ratio": float(clipped_ratio[target].detach()),
        "target_clip": selected_clip,
        "target_logp_loss_gradient": target_gradient,
        "weights_dir": str(arm_dir.resolve()),
    }
    del optimizer, model, tokenizer, logps
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--state-audit", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    out = Path(args.out)
    try:
        evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
        event = evaluation["first_stable_event_for_one_step_followup"]
        samples, states, target = select_batch(
            read_jsonl(args.samples), read_jsonl(args.states), event
        )
        state_record = next(
            row
            for row in read_jsonl(args.state_audit)
            if str(row["state_id"]) == str(event["state_id"])
        )
        cfg = load_config(args.config)
        ref_cfg, alt_cfg = path_config(cfg, "path_ref"), path_config(cfg, "path_alt")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
        arms = [
            run_arm(
                name="A_reference", cfg=ref_cfg, samples=samples, states=states,
                target=target, event=event,
                expected_tensor_hash=state_record["ref_first_sha256"],
                forced_reference_clip=None, output_dir=out_dir,
            ),
            run_arm(
                name="B_candidate", cfg=alt_cfg, samples=samples, states=states,
                target=target, event=event,
                expected_tensor_hash=state_record["alt_first_sha256"],
                forced_reference_clip=None, output_dir=out_dir,
            ),
            run_arm(
                name="C_branch_repair", cfg=alt_cfg, samples=samples, states=states,
                target=target, event=event,
                expected_tensor_hash=state_record["alt_first_sha256"],
                forced_reference_clip=bool(event["ref_clip"]), output_dir=out_dir,
            ),
        ]
        by_name = {arm["arm"]: arm for arm in arms}
        a, b, c = by_name["A_reference"], by_name["B_candidate"], by_name["C_branch_repair"]
        integrity = (
            b["scorer_call_sha256"] == c["scorer_call_sha256"]
            and b["compile_audit"]["graph_hashes"] == c["compile_audit"]["graph_hashes"]
            and b["compile_audit"]["graph_nodes"] == c["compile_audit"]["graph_nodes"]
            and a["target_clip"] is bool(event["ref_clip"])
            and b["target_clip"] is bool(event["alt_clip"])
            and c["target_logp_loss_gradient"] != b["target_logp_loss_gradient"]
            and b["target_logp_loss_gradient"] == 0.0
            and c["target_logp_loss_gradient"] != 0.0
        )
        if not integrity:
            raise RuntimeError("branch treatment integrity gate failed")
        payload = {
            "schema_version": "forkcert.qwen3-grpo-grad-branch-repair.v0.5",
            "status": "MECHANICALLY_VALID_PENDING_INDEPENDENT_DISTANCE_AUDIT",
            "event": event,
            "target_flat_index": target,
            "arms": arms,
            "intervention_integrity_valid": True,
            "optimizer_probe": {"kind": "SGD", "lr": 1e-5, "momentum": 0.0, "foreach": False},
            "compiler_correctness": "NO CLAIM",
            "natural_training_update_effect": "NOT CLAIMED",
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as error:
        payload = {
            "schema_version": "forkcert.qwen3-grpo-grad-branch-repair.v0.5",
            "status": "INVALID",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
