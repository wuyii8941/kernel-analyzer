#!/usr/bin/env python3
"""Run a paired candidate/repair trajectory and measure loss separation.

This runner is deliberately downstream of the 4096-step direct-direction
test.  It does not decide whether a case contains persistent bias.  It asks a
separate question: once that bias has been established, do candidate and
repair trajectories separate in parameter space and in held-out loss?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver  # noqa: E402


CONFIGS = {
    "qwen_lmhead_dx": {
        "case_id": "qwen_seq128_lmhead_dx",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "architecture": "qwen",
        "carrier": "model.norm.weight",
        "learning_rate": 1e-4,
        "sequence_length": 128,
        "contrast": "compiled BF16 lm_head dX MM vs FP32 MM + BF16 ABI cast",
    },
    "liger_fused_ce": {
        "case_id": "liger_fused_ce_t128",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "architecture": "qwen",
        "carrier": "model.embed_tokens.weight",
        "learning_rate": 1e-4,
        "sequence_length": 128,
        "contrast": "Liger fused CE BF16 dW accumulation vs FP32 accumulation",
    },
}


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "population_std": float(array.std())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--validation-states", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.steps < args.eval_every or args.steps % args.eval_every:
        raise ValueError("steps must contain complete evaluation intervals")
    config = CONFIGS[args.case]
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    needed = args.train_states + args.validation_states
    if states is None or len(states) < needed:
        raise RuntimeError(f"input bank needs at least {needed} states")
    train = states[: args.train_states]
    validation = states[args.train_states : needed]
    train_ids = [str(row.get("state_id", row.get("sequence_id", i))) for i, row in enumerate(train)]
    validation_ids = [
        str(row.get("state_id", row.get("sequence_id", args.train_states + i)))
        for i, row in enumerate(validation)
    ]
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("training and validation state IDs overlap")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    configure_candidate_runtime(24_000)
    model = load_model(config["architecture"], Path(config["model"]), device)
    model.eval()
    named = dict(model.named_parameters())
    carrier = named[config["carrier"]]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == config["carrier"])

    candidate: Any | None = None
    modules: list[Any] = []
    candidate_loss: Any | None = None
    repair_loss: Any | None = None
    if args.case == "liger_fused_ce":
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        if carrier.untyped_storage().data_ptr() != model.lm_head.weight.untyped_storage().data_ptr():
            raise RuntimeError("Liger carrier is not tied to lm_head")
        candidate_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=None,
        ).to(device)
        repair_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
        ).to(device)
    else:
        start = len(PyCodeCache.modules)
        candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
        warm = torch.tensor([train[0]["token_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        candidate(warm).backward()
        torch.cuda.synchronize(device)
        modules = list(PyCodeCache.modules[start:])

    initial = carrier.detach().float().clone()
    candidate_master = initial.clone()
    repair_master = initial.clone()
    candidate_m = torch.zeros_like(initial)
    candidate_v = torch.zeros_like(initial)
    repair_m = torch.zeros_like(initial)
    repair_v = torch.zeros_like(initial)

    def gradient(
        master: torch.Tensor, state: dict[str, Any], *, repair: bool, seed: int
    ) -> tuple[str, float, torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            carrier.copy_(master.to(carrier.dtype))
        tokens = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        local: dict[str, Any]
        if args.case == "liger_fused_ce":
            hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
            labels = torch.nn.functional.pad(tokens, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
            module = repair_loss if repair else candidate_loss
            loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
            loss.backward()
            local = {"changed_coordinates": -1, "endpoint_error_l2": None}
        else:
            observer = None
            if repair:
                observer = ShapeObserver(
                    modules,
                    "fp32",
                    [],
                    left_shape=(config["sequence_length"], 151936),
                    right_shape=(151936, 2048),
                )
            if observer is None:
                loss = candidate(tokens)
                loss.backward()
                local = {"changed_coordinates": 0, "endpoint_error_l2": None}
            else:
                with observer:
                    loss = candidate(tokens)
                    loss.backward()
                if observer.calls != 1:
                    raise RuntimeError("lm_head repair missed the exact endpoint")
                local = {"changed_coordinates": -1, "endpoint_error_l2": observer.changed_l2}
        torch.cuda.synchronize(device)
        if carrier.grad is None or not torch.isfinite(carrier.grad).all():
            raise RuntimeError("carrier gradient is absent or nonfinite")
        return tensor_digest(loss), float(loss.detach().float().item()), carrier.grad.detach().float().clone(), local

    def checkpoint(step: int) -> dict[str, Any]:
        candidate_losses: list[float] = []
        repair_losses: list[float] = []
        for offset, state in enumerate(validation):
            seed = 90_000 + step * args.validation_states + offset
            _, loss_c, _, _ = gradient(candidate_master, state, repair=False, seed=seed)
            _, loss_r, _, _ = gradient(repair_master, state, repair=True, seed=seed)
            candidate_losses.append(loss_c)
            repair_losses.append(loss_r)
        gap = float(np.mean(candidate_losses) - np.mean(repair_losses))
        return {
            "step": step,
            "candidate_loss": float(np.mean(candidate_losses)),
            "repair_loss": float(np.mean(repair_losses)),
            "loss_gap_candidate_minus_repair": gap,
            "absolute_loss_gap": abs(gap),
            "parameter_distance_l2": float(torch.linalg.vector_norm(candidate_master - repair_master)),
            "candidate_from_initial_l2": float(torch.linalg.vector_norm(candidate_master - initial)),
            "repair_from_initial_l2": float(torch.linalg.vector_norm(repair_master - initial)),
        }

    checkpoints = [checkpoint(0)]
    rows: list[dict[str, Any]] = []
    for index in range(args.steps):
        step = index + 1
        state_index = index % args.train_states
        state = train[state_index]
        seed = 60_000 + step
        digest_c, loss_c, grad_c, _ = gradient(candidate_master, state, repair=False, seed=seed)
        digest_r, loss_r, grad_r, local = gradient(repair_master, state, repair=True, seed=seed)
        if step == 1 and args.case == "qwen_lmhead_dx" and digest_c != digest_r:
            raise RuntimeError("backward-only repair changed the initial forward loss")
        update_c, candidate_m, candidate_v = adam_delta(
            grad_c, candidate_m, candidate_v, step,
            learning_rate=config["learning_rate"], beta1=0.9, beta2=0.95,
        )
        update_r, repair_m, repair_v = adam_delta(
            grad_r, repair_m, repair_v, step,
            learning_rate=config["learning_rate"], beta1=0.9, beta2=0.95,
        )
        candidate_master.add_(update_c)
        repair_master.add_(update_r)
        rows.append({
            "step": step,
            "state_id": train_ids[state_index],
            "candidate_loss": loss_c,
            "repair_loss": loss_r,
            "loss_gap_candidate_minus_repair": loss_c - loss_r,
            "parameter_distance_l2": float(torch.linalg.vector_norm(candidate_master - repair_master)),
            "candidate_update_l2": float(torch.linalg.vector_norm(update_c)),
            "repair_update_l2": float(torch.linalg.vector_norm(update_r)),
            "endpoint_changed_coordinates": local["changed_coordinates"],
        })
        if step % args.eval_every == 0:
            value = checkpoint(step)
            checkpoints.append(value)
            if not args.quiet:
                print(json.dumps({"event": "PAIRED_LOSS_CHECKPOINT", "case": args.case, **value}), flush=True)
        del grad_c, grad_r, update_c, update_r

    recent = rows[-min(512, len(rows)) :]
    final = checkpoints[-1]
    recent_loss_gap_mean = float(np.mean([
        row["loss_gap_candidate_minus_repair"] for row in recent
    ]))
    payload = {
        "schema": "kernel-analyzer-declared-paired-loss-v1",
        "status": "COMPLETE_PAIRED_LOSS_AUDIT",
        "case_id": config["case_id"],
        "protocol": {
            "input_bank": str(args.input_bank.resolve()),
            "training_state_ids": train_ids,
            "validation_state_ids": validation_ids,
            "training_schedule": "repeat frozen training-state order",
            "steps": args.steps,
            "validation_every_steps": args.eval_every,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": config["learning_rate"],
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "initial_moments": "ZERO_THEN_EVOLVED_SEPARATELY_PER_ARM",
            },
            "candidate": config["contrast"].split(" vs ")[0],
            "repair": " vs ".join(config["contrast"].split(" vs ")[1:]),
            "carrier": config["carrier"],
            "other_parameters_frozen": True,
        },
        "checkpoints": checkpoints,
        "final": final,
        "last_512_train_steps": {
            "candidate_loss": mean_std([float(row["candidate_loss"]) for row in recent]),
            "repair_loss": mean_std([float(row["repair_loss"]) for row in recent]),
            "paired_loss_gap": mean_std([float(row["loss_gap_candidate_minus_repair"]) for row in recent]),
        },
        "train_rows": rows,
        "loss_separation_observed": bool(
            final["parameter_distance_l2"] > 0.0
            and (
                final["absolute_loss_gap"] > 0.0
                or abs(recent_loss_gap_mean) > 0.0
            )
        ),
        "claim_boundary": (
            "A controlled one-carrier candidate/repair trajectory. Nonzero paired held-out "
            "loss and parameter separation is a functional consequence signal, not evidence "
            "of different converged solutions or full-parameter training degradation. Persistent "
            "bias must be established independently by the 4096-step direct-direction test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "PAIRED_LOSS_COMPLETE",
        "case": args.case,
        "steps": args.steps,
        "final_parameter_distance_l2": final["parameter_distance_l2"],
        "final_loss_gap": final["loss_gap_candidate_minus_repair"],
        "last_512_loss_gap_mean": recent_loss_gap_mean,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
