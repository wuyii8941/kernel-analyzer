#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.detector import clip_active
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from scripts.phase6_twin_training import path_config, raw_model, state_tensors
from scripts.phase8_case_attribution import make_batch
from scripts.phase8_matched_step import select_fork_batch, selected_surrogate_loss
from scripts.phase9_mutation_catalog import MUTATIONS


def replay_batch_hash(samples: list[dict[str, Any]], states: list[dict[str, Any]]) -> str:
    payload = {
        "samples": [
            {
                "case_id": str(sample["case_id"]),
                "prompt_ids": [int(value) for value in sample["prompt_ids"]],
                "response_ids": [int(value) for value in sample["response_ids"]],
            }
            for sample in samples
        ],
        "states": [
            {
                "case_id": str(state["case_id"]),
                "token_index": int(state["token_index"]),
                "token_id": int(state["token_id"]),
                "old_logp": float(state["old_logp"]),
                "advantage": float(state["advantage"]),
            }
            for state in states
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def batch_logps_with_grad(tokenizer: Any, model: Any, config: Any, samples: list[dict[str, Any]], reducer: str):
    import torch
    from forkcert.logprob_runner import attention_backend_context, precision_context

    input_ids, attention_mask, max_prompt = make_batch(tokenizer, samples, config.device)
    response_ids = [[int(value) for value in sample["response_ids"]] for sample in samples]
    with attention_backend_context(config), precision_context(config):
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
            denominator = sum(
                (chunk.sum(dim=-1) for chunk in reversed(chunks)),
                torch.zeros_like(maximum[..., 0]),
            )
            log_denominator = maximum[..., 0] + torch.log(denominator)
            target_logits = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            gathered = target_logits - log_denominator
        else:
            raise ValueError(f"unsupported reducer: {reducer}")
    rows = [
        gathered[index, max_prompt - 1 : max_prompt - 1 + len(response)]
        for index, response in enumerate(response_ids)
    ]
    return torch.cat(rows), [token for response in response_ids for token in response]


def expected_step1_logps(
    mutation: str | None,
    states: list[dict[str, Any]],
    certificates_path: str,
    mutation_rows_path: str,
) -> list[float]:
    keys = [(str(row["case_id"]), int(row["token_index"])) for row in states]
    if mutation is None:
        rows = {
            (str(row["case_id"]), int(row["token_index"])): float(row["logp_ref"])
            for row in read_jsonl(certificates_path)
        }
    else:
        rows = {
            (str(row["case_id"]), int(row["token_index"])): float(row["logp_mutated"])
            for row in read_jsonl(mutation_rows_path)
            if str(row["bug"]) == mutation
        }
    missing = [key for key in keys if key not in rows]
    if missing:
        raise ValueError(f"step-1 canary is missing {len(missing)} rows, first={missing[0]}")
    return [rows[key] for key in keys]


def main() -> None:
    mutation_names = [mutation.name for mutation in MUTATIONS]
    parser = argparse.ArgumentParser(description="One clean or mutation arm for the zero-clipping-fork trajectory audit.")
    parser.add_argument("--mutation", choices=mutation_names)
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--states", default="data/phase6_step5_replay_dump.jsonl")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--mutation-rows", default="results/phase9_mutations_gated/all_mutation_rows.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--step1-tolerance", type=float, default=1e-5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    configure_determinism(0)
    import torch

    mutation = next((item for item in MUTATIONS if item.name == args.mutation), None)
    cert = next(
        row
        for row in read_jsonl(args.certificates)
        if row.get("actual_fork")
        and str(row["case_id"]) == args.case_id
        and int(row["token_index"]) == args.token_index
    )
    samples, states, target = select_fork_batch(read_jsonl(args.samples), read_jsonl(args.states), cert)
    config = path_config(load_config(args.config), "path_ref")
    tokenizer, model = load_hf_path(config)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=args.lr)
    expected_ids = [int(row["token_id"]) for row in states]
    expected_logps = expected_step1_logps(args.mutation, states, args.certificates, args.mutation_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {1, min(5, args.steps), args.steps}
    trajectory = []
    with ExitStack() as stack:
        changed_modules = mutation.installer(model, stack) if mutation else []
        reducer = mutation.reducer if mutation else "standard"
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logps, token_ids = batch_logps_with_grad(tokenizer, model, config, samples, reducer)
            if token_ids != expected_ids:
                raise ValueError("mutation trajectory token alignment mismatch")
            values = logps.detach().float().cpu().tolist()
            if step == 1:
                canary_delta = max(abs(left - right) for left, right in zip(values, expected_logps, strict=True))
                if canary_delta > args.step1_tolerance:
                    raise ValueError(
                        f"step-1 mutation canary mismatch for {args.mutation or 'clean'}: {canary_delta}"
                    )
            else:
                canary_delta = None
            old, advantages = state_tensors(torch, states, logps.dtype, logps.device)
            loss, _, _ = selected_surrogate_loss(
                torch, logps, old, advantages, float(cert["eps"]), target, None
            )
            branches = [
                clip_active(value, float(state["old_logp"]), int(state["advantage_sign"]), float(cert["eps"]))
                if int(state["advantage_sign"]) != 0
                else None
                for value, state in zip(values, states, strict=True)
            ]
            target_grad = float(torch.autograd.grad(loss, logps, retain_graph=True)[0][target].item())
            loss.backward()
            gradient_square = torch.zeros((), dtype=torch.float64, device=logps.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    gradient_square += parameter.grad.detach().double().square().sum()
            optimizer.step()
            trajectory.append(
                {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "full_gradient_norm": float(torch.sqrt(gradient_square).item()),
                    "target_logp": values[target],
                    "target_loss_gradient": target_grad,
                    "step1_canary_max_abs_delta": canary_delta,
                    "logps": values,
                    "clip_active": branches,
                }
            )
            if step in checkpoints:
                checkpoint = out_dir / f"step_{step:02d}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                raw_model(model).save_pretrained(checkpoint, safe_serialization=True)
    payload = {
        "schema_version": "forkcert.mutation_trajectory_arm.v1",
        "arm": "clean_reference" if mutation is None else f"mutation_{mutation.name}",
        "mutation": mutation.name if mutation else None,
        "mutation_description": mutation.description if mutation else None,
        "mutation_mechanism": mutation.mechanism if mutation else None,
        "changed_modules": changed_modules,
        "reducer": reducer,
        "steps": args.steps,
        "learning_rate": args.lr,
        "batch_tokens": len(states),
        "replay": {
            "config": args.config,
            "checkpoint_path": config.model_name_or_path,
            "samples_path": args.samples,
            "states_path": args.states,
            "certificates_path": args.certificates,
            "mutation_rows_path": args.mutation_rows,
            "optimizer_step": int(states[0]["optimizer_step"]),
            "rollout_batch": int(states[0]["rollout_batch"]),
            "case_ids": [str(sample["case_id"]) for sample in samples],
            "target_case_id": args.case_id,
            "target_token_index": args.token_index,
            "batch_sha256": replay_batch_hash(samples, states),
        },
        "checkpoint_steps": sorted(checkpoints),
        "trajectory": trajectory,
        "claim_scope": (
            "Measures continuous update divergence and clipping-branch divergence on one frozen replay batch; "
            "parameter distance is not task-level harm or semantic equivalence."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "trajectory"}, indent=2, sort_keys=True))
    del optimizer, model, tokenizer
    cleanup_memory()


if __name__ == "__main__":
    main()
