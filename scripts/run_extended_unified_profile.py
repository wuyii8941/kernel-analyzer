#!/usr/bin/env python3
"""Unified 16+16 profiles for Qwen lm-head, Qwen v_proj, and Mamba in_proj."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from analyze_three_mechanism_profiles import _profile  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_qwen128_vproj_repair import VProjRepair  # noqa: E402
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver  # noqa: E402
from scripts.targeted_external_intervention import _count_sketch  # noqa: E402


CONFIG = {
    "qwen_lmhead": {
        "case_id": "qwen_seq128_lmhead_dx",
        "architecture": "qwen",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "bank": "results/coverage/qwen_seq128_input_bank.json",
        "carrier": "model.norm.weight",
        "lr": 1e-4,
        "kind": "LMHEAD",
    },
    "qwen_vproj": {
        "case_id": "qwen128_vproj_output",
        "architecture": "qwen",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "bank": "results/coverage/qwen_seq128_input_bank.json",
        "carrier": "model.layers.0.self_attn.v_proj.weight",
        "lr": 1e-5,
        "kind": "VPROJ",
        "output_shape": [128, 1024],
        "match_index": 2,
    },
    "mamba_inproj": {
        "case_id": "mamba_seq64_in_proj",
        "architecture": "mamba",
        "model": "/data1/tzh/models/state-spaces/mamba-130m-hf",
        "bank": "results/coverage/mamba_seq64_input_bank.json",
        "carrier": "backbone.layers.0.mixer.in_proj.weight",
        "lr": 1e-5,
        "kind": "INPROJ",
        "target_sha": "9c03ef3fc9b93005efed225a176c3e97efa91d33af2fae0b27cbb28e695c3cee",
    },
}


def compact(value: torch.Tensor | np.ndarray) -> np.ndarray:
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() <= 4096:
        return flat.cpu().numpy().copy()
    return _count_sketch(flat, dimension=4096).numpy()


def tokens(state: dict[str, Any]) -> list[int]:
    return state.get("token_ids", state.get("input_ids"))


def state_id(state: dict[str, Any], index: int) -> str:
    return str(state.get("state_id", state.get("sequence_id", index)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CONFIG), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = CONFIG[args.case]
    protocol = json.loads((ROOT / "results/property/extended_unified_profiles_v1/protocol.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_NEW_RESULTS":
        raise RuntimeError("protocol was not frozen")
    bank = json.loads((ROOT / config["bank"]).read_text())
    states = list(bank.get("states", bank.get("records")))
    if len(states) != 32:
        raise RuntimeError("unified profile requires exactly 32 frozen states")

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    configure_candidate_runtime(24_000)
    model = load_model(config["architecture"], Path(config["model"]), device).eval()
    parameter = dict(model.named_parameters())[config["carrier"]]
    if config["kind"] != "INPROJ":
        for name, candidate_parameter in model.named_parameters():
            candidate_parameter.requires_grad_(name == config["carrier"])
    start = len(PyCodeCache.modules)
    compiled = torch.compile(
        LossStep(model), backend="inductor", fullgraph=False, dynamic=False,
    )
    warm = torch.tensor([tokens(states[0])], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); compiled(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])

    master = parameter.detach().float().clone()
    first = torch.zeros_like(master); second = torch.zeros_like(master)
    stages = {name: ([], []) for name in ("LOCAL", "GRADIENT", "ADAMW_UPDATE")}
    ids: list[str] = []

    def branch(state: dict[str, Any], repair: bool):
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        model.zero_grad(set_to_none=True)
        value = torch.tensor([tokens(state)], dtype=torch.long, device=device)
        observer = None
        if repair:
            if config["kind"] == "LMHEAD":
                observer = ShapeObserver(
                    modules, "fp32", [], left_shape=(128, 151936),
                    right_shape=(151936, 2048),
                )
            elif config["kind"] == "VPROJ":
                observer = VProjRepair(
                    modules, "REPAIR_FP32_CAST_BF16", None,
                    expected_output_shape=tuple(config["output_shape"]),
                    expected_match_index=int(config["match_index"]),
                )
            else:
                observer = VProjRepair(
                    modules, "REPAIR_FP32_CAST_BF16", config["target_sha"],
                )
        if observer is None:
            loss = compiled(value); loss.backward()
        else:
            with observer:
                loss = compiled(value); loss.backward()
        torch.cuda.synchronize(device)
        if parameter.grad is None:
            raise RuntimeError("declared carrier gradient is absent")
        return tensor_digest(loss), parameter.grad.detach().float().clone(), observer

    for index, state in enumerate(states):
        loss_c, grad_c, _ = branch(state, False)
        loss_r, grad_r, observer = branch(state, True)
        if observer is None or observer.local_vector is None or observer.repair_vector is None:
            raise RuntimeError(f"exact repair emitted no local vectors at state {index}")
        if config["kind"] == "LMHEAD" and loss_c != loss_r:
            raise RuntimeError("backward-only repair changed forward loss")
        update_c, _, _ = adam_delta(
            grad_c, first, second, index + 1, learning_rate=config["lr"], beta1=0.9, beta2=0.95,
        )
        update_r, next_first, next_second = adam_delta(
            grad_r, first, second, index + 1, learning_rate=config["lr"], beta1=0.9, beta2=0.95,
        )
        stages["LOCAL"][0].append(compact(observer.local_vector))
        stages["LOCAL"][1].append(compact(observer.repair_vector))
        stages["GRADIENT"][0].append(compact(grad_c - grad_r))
        stages["GRADIENT"][1].append(compact(grad_r))
        stages["ADAMW_UPDATE"][0].append(compact(update_c - update_r))
        stages["ADAMW_UPDATE"][1].append(compact(update_r))
        master.add_(update_r); first, second = next_first, next_second
        ids.append(state_id(state, index))
        print(json.dumps({"event": "EXTENDED_UNIFIED_STATE", "case": args.case, "step": index + 1}), flush=True)
        del grad_c, grad_r, update_c, update_r
        torch.cuda.empty_cache()

    payload = {
        "schema": "kernel-analyzer-extended-unified-profile-v1",
        "status": "COMPLETE",
        "case_id": config["case_id"],
        "implementation_kind": config["kind"],
        "optimizer": {
            "name": "AdamW", "lr": config["lr"], "betas": [0.9, 0.95],
            "initial_moments": "ZERO_THEN_REPAIR_EVOLVED",
        },
        "state_ids": ids,
        "split": {"calibration": ids[:16], "confirmation": ids[16:]},
        "stages": {
            name: _profile(*values, seed=20260921 + offset)
            for offset, (name, values) in enumerate(stages.items())
        },
        "claim_boundary": "32 matched natural states; short-state stagewise evidence, not long-run convergence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
