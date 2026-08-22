#!/usr/bin/env python3
"""Measure Liger fused-CE error at endpoint, gradient, and update stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from scripts.capture_liger_bias_formation_v21 import LigerEndpointObserver  # noqa: E402


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def curve(vectors: list[torch.Tensor]) -> list[dict[str, float | int]]:
    total = torch.zeros_like(vectors[0], dtype=torch.float64)
    energy = 0.0
    out = []
    for i, v in enumerate(vectors, 1):
        x = v.double()
        total.add_(x)
        energy += float(torch.dot(x.reshape(-1), x.reshape(-1)))
        if i in (2, 4, 8, 16, 32):
            den = math.sqrt(max(energy, 0.0))
            out.append({"horizon": i, "resultant_l2": float(torch.linalg.vector_norm(total)), "path_rms_l2": den, "coherence_amplification": float(torch.linalg.vector_norm(total)) / max(den, 1e-30)})
    return out


def branch(model, module, ids, master, parameter, repair: bool):
    with torch.no_grad():
        parameter.copy_(master.to(parameter.dtype))
    model.zero_grad(set_to_none=True)
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    observed = []
    hidden.register_hook(lambda g: observed.append(g.detach().clone()))
    labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
    observer = LigerEndpointObserver(module, "REPAIR") if repair else LigerEndpointObserver(module, "CANDIDATE")
    with observer:
        loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
        loss_digest = hashlib.sha256(loss.detach().cpu().numpy().tobytes()).hexdigest()
        loss.backward()
    if len(observed) != 1 or observer.endpoint_vector is None or parameter.grad is None:
        raise RuntimeError("Liger branch did not expose endpoint, dH, and tied gradient")
    return loss_digest, torch.from_numpy(observer.endpoint_vector).float(), parameter.grad.detach().float().cpu().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")
    from transformers import AutoModelForCausalLM
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    design = json.loads((ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json").read_text())
    protocol = json.loads((ROOT / "results/trajectory/liger_protocol.json").read_text())
    records = {row["sequence_id"]: row for row in design["records"]}
    states = [records[x] for x in protocol["trajectory"]["state_order"]]
    if len(states) != 32:
        raise RuntimeError("Liger reference trajectory must contain 32 states")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained("/data1/tzh/models/Qwen/Qwen3-1.7B", dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).to(device).eval()
    parameter = model.model.embed_tokens.weight
    module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=None).to(device)
    repair_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=torch.float32).to(device)
    reference = parameter.detach().float().cpu().clone()
    local, gradient, update, rows = [], [], [], []
    for i, state in enumerate(states):
        ids = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        loss_c, endpoint_c, grad_c = branch(model, module, ids, reference, parameter, False)
        loss_r, endpoint_r, grad_r = branch(model, repair_module, ids, reference, parameter, True)
        if loss_c != loss_r:
            raise RuntimeError(f"repair changed Liger loss at state {i}")
        e = endpoint_c - endpoint_r
        g = grad_c - grad_r
        u = -args.learning_rate * g
        local.append(e); gradient.append(g); update.append(u)
        reference.add_((-args.learning_rate * grad_r).cpu())
        rows.append({"step": i + 1, "state_id": str(state["sequence_id"]), "local_l2": float(torch.linalg.vector_norm(e.double())), "gradient_l2": float(torch.linalg.vector_norm(g.double())), "effective_update_l2": float(torch.linalg.vector_norm(u.double()))})
        print(json.dumps({"event": "LIGER_THREE_STAGE_STEP", **rows[-1]}), flush=True)
        del ids, endpoint_c, endpoint_r, grad_c, grad_r, e, g, u
        torch.cuda.empty_cache()
    payload = {"schema": "kernel-analyzer-liger-three-stage-reference-v1", "status": "COMPLETE_ORDERED_32_STATE_REFERENCE", "case_id": "liger_fused_ce_t128", "state_count": 32, "learning_rate": args.learning_rate, "reference_trajectory": "repair-gradient SGD master; candidate and repair evaluated at identical pre-step tied-embedding state", "stages": {"operator_output_error": {"coherence_curve": curve(local)}, "parameter_gradient_error": {"coherence_curve": curve(gradient)}, "effective_update_error": {"coherence_curve": curve(update)}}, "rows": rows, "claim_boundary": "One ordered reference trajectory and one tied carrier; not full-parameter training."}
    payload["result_sha256"] = sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
