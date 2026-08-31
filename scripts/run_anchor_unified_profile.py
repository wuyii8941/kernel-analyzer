#!/usr/bin/env python3
"""Unified 16+16 local/gradient/AdamW profiles for Liger and Phi anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src"), str(ROOT / "scripts")]

from analyze_three_mechanism_profiles import _profile  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.targeted_external_intervention import _count_sketch  # noqa: E402


def _small_or_sketch(value: torch.Tensor) -> np.ndarray:
    flat = value.detach().float().reshape(-1)
    if flat.numel() <= 4096:
        return flat.cpu().numpy().copy()
    return _count_sketch(flat, dimension=4096).numpy()


class CompactLigerEndpoint:
    def __init__(self, module: Any) -> None:
        self.module = module
        self.original = None
        self.vector: np.ndarray | None = None
        self.calls = 0

    def __enter__(self):
        import liger_kernel.ops.fused_linear_cross_entropy as fused
        self.original = fused.fused_linear_cross_entropy_forward
        original = self.original

        def wrapped(*args: Any, **kwargs: Any):
            result = original(*args, **kwargs)
            endpoint = result[4]
            if endpoint is None:
                raise RuntimeError("Liger fused dW endpoint is absent")
            self.vector = _small_or_sketch(endpoint)
            self.calls += 1
            return result

        fused.fused_linear_cross_entropy_forward = wrapped
        return self

    def __exit__(self, *unused):
        del unused
        import liger_kernel.ops.fused_linear_cross_entropy as fused
        fused.fused_linear_cross_entropy_forward = self.original
        if self.calls != 1:
            raise RuntimeError(f"Liger endpoint executed {self.calls} times")


def run_liger(device: torch.device) -> dict:
    from transformers import AutoModelForCausalLM
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    design = json.loads((ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json").read_text())
    protocol = json.loads((ROOT / "results/trajectory/liger_protocol.json").read_text())
    records = {row["sequence_id"]: row for row in design["records"]}
    states = [records[x] for x in protocol["trajectory"]["state_order"]]
    if len(states) != 32:
        raise RuntimeError("Liger profile requires 32 frozen states")
    torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    model = AutoModelForCausalLM.from_pretrained(
        "/data1/tzh/models/Qwen/Qwen3-1.7B", dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    parameter = model.model.embed_tokens.weight
    candidate_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=None).to(device)
    repair_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=torch.float32).to(device)
    master = parameter.detach().float().clone()
    first = torch.zeros_like(master); second = torch.zeros_like(master)
    stages = {name: ([], []) for name in ("LOCAL", "GRADIENT", "ADAMW_UPDATE")}
    state_ids = []

    def branch(state: dict, module: Any):
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        observed = []
        hidden.register_hook(lambda g: observed.append(g.detach()))
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        observer = CompactLigerEndpoint(module)
        with observer:
            loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
            loss_value = loss.detach().float().cpu().numpy().tobytes()
            loss.backward()
        if parameter.grad is None or len(observed) != 1 or observer.vector is None:
            raise RuntimeError("Liger branch did not expose the declared F+B path")
        return hashlib.sha256(loss_value).hexdigest(), observer.vector, parameter.grad.detach().float().clone()

    for index, state in enumerate(states):
        loss_c, endpoint_c, grad_c = branch(state, candidate_module)
        loss_r, endpoint_r, grad_r = branch(state, repair_module)
        if loss_c != loss_r:
            raise RuntimeError(f"Liger repair changed forward loss at state {index}")
        update_c, _, _ = adam_delta(grad_c, first, second, index + 1, learning_rate=1e-4, beta1=0.9, beta2=0.95)
        update_r, next_first, next_second = adam_delta(grad_r, first, second, index + 1, learning_rate=1e-4, beta1=0.9, beta2=0.95)
        stages["LOCAL"][0].append(endpoint_c - endpoint_r); stages["LOCAL"][1].append(endpoint_r)
        stages["GRADIENT"][0].append(_small_or_sketch(grad_c - grad_r)); stages["GRADIENT"][1].append(_small_or_sketch(grad_r))
        stages["ADAMW_UPDATE"][0].append(_small_or_sketch(update_c - update_r)); stages["ADAMW_UPDATE"][1].append(_small_or_sketch(update_r))
        master.add_(update_r); first, second = next_first, next_second
        state_ids.append(str(state["sequence_id"]))
        print(json.dumps({"event": "LIGER_UNIFIED_PROFILE_STATE", "step": index + 1}), flush=True)
        del grad_c, grad_r, update_c, update_r
        torch.cuda.empty_cache()
    return {"case_id": "liger_fused_ce_t128", "state_ids": state_ids,
            "optimizer": {"name": "AdamW", "lr": 1e-4, "betas": [0.9, 0.95], "initial_moments": "ZERO_THEN_REPAIR_EVOLVED"},
            "stages": {name: _profile(*values, seed=20260901 + i) for i, (name, values) in enumerate(stages.items())}}


def run_phi(device: torch.device) -> dict:
    from torch._inductor.codecache import PyCodeCache

    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = list(bank.get("states", bank.get("records")))
    if len(states) != 32:
        raise RuntimeError("Phi profile requires 32 frozen states")
    configure_candidate_runtime(24000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device).eval()
    parameter = model.model.norm.weight
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    master = parameter.detach().float().clone()
    first = torch.zeros_like(master); second = torch.zeros_like(master)
    stages = {name: ([], []) for name in ("LOCAL", "GRADIENT", "ADAMW_UPDATE")}
    state_ids = []

    def branch(state: dict, repair: bool):
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
        if observer is None:
            loss = candidate(ids); loss.backward()
        else:
            with observer:
                loss = candidate(ids); loss.backward()
        torch.cuda.synchronize(device)
        if parameter.grad is None:
            raise RuntimeError("Phi carrier gradient is absent")
        return tensor_digest(loss), parameter.grad.detach().float().clone(), observer

    for index, state in enumerate(states):
        loss_c, grad_c, _ = branch(state, False)
        loss_r, grad_r, observer = branch(state, True)
        if loss_c != loss_r or observer is None or observer.local_vector is None or observer.repair_vector is None:
            raise RuntimeError(f"Phi exact repair failed at state {index}")
        update_c, _, _ = adam_delta(grad_c, first, second, index + 1, learning_rate=1e-4, beta1=0.9, beta2=0.95)
        update_r, next_first, next_second = adam_delta(grad_r, first, second, index + 1, learning_rate=1e-4, beta1=0.9, beta2=0.95)
        stages["LOCAL"][0].append(np.asarray(observer.local_vector, dtype=np.float32)); stages["LOCAL"][1].append(np.asarray(observer.repair_vector, dtype=np.float32))
        stages["GRADIENT"][0].append(_small_or_sketch(grad_c - grad_r)); stages["GRADIENT"][1].append(_small_or_sketch(grad_r))
        stages["ADAMW_UPDATE"][0].append(_small_or_sketch(update_c - update_r)); stages["ADAMW_UPDATE"][1].append(_small_or_sketch(update_r))
        master.add_(update_r); first, second = next_first, next_second
        state_ids.append(str(state.get("state_id", index)))
        print(json.dumps({"event": "PHI_UNIFIED_PROFILE_STATE", "step": index + 1}), flush=True)
        del grad_c, grad_r, update_c, update_r
        torch.cuda.empty_cache()
    return {"case_id": "phi4_seq64_lmhead_dx", "state_ids": state_ids,
            "optimizer": {"name": "AdamW", "lr": 1e-4, "betas": [0.9, 0.95], "initial_moments": "ZERO_THEN_REPAIR_EVOLVED"},
            "stages": {name: _profile(*values, seed=20260911 + i) for i, (name, values) in enumerate(stages.items())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("liger", "phi"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    result = run_liger(device) if args.case == "liger" else run_phi(device)
    payload = {"schema": "kernel-analyzer-anchor-unified-profile-v1", "status": "COMPLETE",
               "split": {"calibration": result["state_ids"][:16], "confirmation": result["state_ids"][16:]}, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
