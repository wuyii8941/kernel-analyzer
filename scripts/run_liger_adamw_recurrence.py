#!/usr/bin/env python3
"""Run a 32-step four-arm AdamW recurrence for the Liger fused-CE repair.

Only the tied embedding/lm-head parameter evolves.  At every step candidate
and FP32-accumulator repair are evaluated at both live states.  This keeps the
direct operator effect separate from feedback caused by earlier divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from kernel_analyzer.persistence_property import CompleteTreeGramPath  # noqa: E402
from kernel_analyzer.trajectory_persistence import OrderedVectorPath  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402


TIED = "model.embed_tokens.weight"


def digest_tensor(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument(
        "--design", type=Path,
        default=ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, choices=(2, 32), default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")

    from transformers import AutoModelForCausalLM
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    design = json.loads(args.design.read_text(encoding="utf-8"))
    states = list(design["records"][:args.steps])
    if len(states) != args.steps:
        raise RuntimeError("Liger state bank is shorter than requested")
    state_ids = [str(row["sequence_id"]) for row in states]
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    parameter = dict(model.named_parameters())[TIED]
    if parameter.untyped_storage().data_ptr() != model.lm_head.weight.untyped_storage().data_ptr():
        raise RuntimeError("Liger carrier is not tied to lm_head")
    candidate_loss = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=None,
    ).to(device)
    repair_loss = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
    ).to(device)

    def gradient(master: torch.Tensor, state: dict[str, Any], *, repair: bool, seed: int) -> tuple[torch.Tensor, dict[str, str]]:
        with torch.no_grad():
            parameter.copy_(master.to(device=device, dtype=parameter.dtype))
        ids = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        observed: list[torch.Tensor] = []
        hidden.register_hook(lambda value: observed.append(value.detach().clone()))
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        module = repair_loss if repair else candidate_loss
        loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
        loss_value = loss.detach().clone()
        loss.backward()
        if len(observed) != 1 or parameter.grad is None:
            raise RuntimeError("Liger branch missed the terminal VJP or tied gradient")
        controls = {
            "loss": digest_tensor(loss_value),
            "hidden": digest_tensor(hidden),
            "dH": digest_tensor(observed[0]),
            "labels": digest_tensor(labels),
        }
        result = parameter.grad.detach().float().cpu().clone()
        del ids, hidden, labels, loss, loss_value, observed
        torch.cuda.empty_cache()
        return result, controls

    initial = parameter.detach().float().cpu().clone()
    candidate_master = initial.clone(); repair_master = initial.clone()
    candidate_m = torch.zeros_like(initial); candidate_v = torch.zeros_like(initial)
    repair_m = torch.zeros_like(initial); repair_v = torch.zeros_like(initial)
    local_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=max(1, args.steps // 2))
    feedback_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=max(1, args.steps // 2))
    actual_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=max(1, args.steps // 2))
    local_gram = None
    if args.steps == 32:
        local_gram = CompleteTreeGramPath(total_steps=32, max_resident_bytes=64 * 1024**3)
    rows = []
    for index, state in enumerate(states):
        step = index + 1; seed = 61000 + index
        before = candidate_master - repair_master
        gc_c, control_cc = gradient(candidate_master, state, repair=False, seed=seed)
        gr_c, control_rc = gradient(candidate_master, state, repair=True, seed=seed)
        gc_r, control_cr = gradient(repair_master, state, repair=False, seed=seed)
        gr_r, control_rr = gradient(repair_master, state, repair=True, seed=seed)
        if control_cc != control_rc or control_cr != control_rr:
            raise RuntimeError(f"Liger repair changed forward or terminal dH at step {step}")
        raw_uc_c, next_cm, next_cv = adam_delta(gc_c, candidate_m, candidate_v, step, learning_rate=args.learning_rate)
        raw_ur_c, _, _ = adam_delta(gr_c, candidate_m, candidate_v, step, learning_rate=args.learning_rate)
        raw_uc_r, _, _ = adam_delta(gc_r, repair_m, repair_v, step, learning_rate=args.learning_rate)
        raw_ur_r, next_rm, next_rv = adam_delta(gr_r, repair_m, repair_v, step, learning_rate=args.learning_rate)
        next_candidate = candidate_master + raw_uc_c; next_repair = repair_master + raw_ur_r
        uc_c = next_candidate - candidate_master
        ur_c = (candidate_master + raw_ur_c) - candidate_master
        uc_r = (repair_master + raw_uc_r) - repair_master
        ur_r = next_repair - repair_master
        after = next_candidate - next_repair
        local = 0.5 * ((uc_c - ur_c) + (uc_r - ur_r))
        feedback = 0.5 * ((uc_c - uc_r) + (ur_c - ur_r))
        actual = after - before
        residual = actual - local - feedback
        relative = float(torch.linalg.vector_norm(residual)) / max(float(torch.linalg.vector_norm(actual)), 1e-30)
        local_path.add(local); feedback_path.add(feedback); actual_path.add(actual)
        if local_gram is not None:
            local_gram.add({"tied": local})
        rows.append({
            "step": step, "state_id": state_ids[index],
            "local_l2": float(torch.linalg.vector_norm(local)),
            "feedback_l2": float(torch.linalg.vector_norm(feedback)),
            "actual_l2": float(torch.linalg.vector_norm(actual)),
            "recurrence_relative": relative,
        })
        candidate_master = next_candidate; repair_master = next_repair
        candidate_m, candidate_v = next_cm, next_cv; repair_m, repair_v = next_rm, next_rv
        print(json.dumps({"event": "LIGER_ADAMW_RECURRENCE_STEP", **rows[-1]}), flush=True)
        del gc_c, gr_c, gc_r, gr_r, uc_c, ur_c, uc_r, ur_r, local, feedback, actual, residual

    statistics: dict[str, Any] = {
        "local_stream": local_path.finalize(),
        "feedback_stream": feedback_path.finalize(),
        "actual_stream": actual_path.finalize(),
    }
    if local_gram is not None:
        statistics["local_complete_gram"] = local_gram.finalize(
            state_ids=state_ids, sign_flip_draws=4000, seed=20260831,
        )
    payload = {
        "schema": "kernel-analyzer-liger-adamw-four-arm-recurrence-v1",
        "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
        "case_id": "liger_fused_ce_t128", "state_ids": state_ids,
        "steps": args.steps, "carrier": TIED,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0},
        "statistics": statistics, "rows": rows,
        "max_recurrence_relative": max(row["recurrence_relative"] for row in rows),
        "claim_boundary": "Four real candidate/repair evaluations per step on the tied carrier under AdamW. Only this carrier evolves; this is not full-parameter training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "LIGER_ADAMW_RECURRENCE_COMPLETE", "status": payload["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
