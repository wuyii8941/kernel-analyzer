#!/usr/bin/env python3
"""Paired 32-step q_proj live-weight trajectory for the layer-23 key F+B repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_SRC = REPO / "archive" / "round1_code" / "src"
for path in (OLD_SRC, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.long_horizon_trigger import atomic_json, build_model, load_eval_states, load_milestone, under_root
from scripts import evolving_triton_observation as observation


PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280)
COLUMNS = slice(1664, 1792)
REGIONS = ["forward:1352", "backward:156", "backward:158", "backward:159", "backward:160", "backward:164"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--campaign", type=Path, default=Path(
        "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/triton_online_reference_campaign_v1.json"
    ))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/final/l23_key_live_weight_adamw.json"))
    parser.add_argument("--checkpoint-step", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps != 32 or args.learning_rate != 1e-5:
        raise ValueError("frozen protocol requires 32 steps and learning rate 1e-5")

    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    campaign_path = under_root(args.campaign, "campaign")
    direction_path = under_root(args.direction, "direction")
    output_path = under_root(args.output, "output")
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/tile_causal_compile")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from transformers import AutoTokenizer
    from forkcert.generated_triton_reference_observer import GeneratedTritonReferenceObserver

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    bank = json.loads(bank_path.read_text())
    milestone = next(row for row in bank["milestones"] if int(row["step"]) == args.checkpoint_step)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    states, evaluation = load_eval_states(tokenizer, 1024, 40, device)
    states = states[8:40]
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)
    parameter = dict(model.named_parameters())[PARAMETER]
    initial_master = parameter.detach().float().clone()
    masters = {"default": initial_master.clone(), "repair": initial_master.clone()}
    moments = {
        arm: {"m": torch.zeros_like(initial_master), "v": torch.zeros_like(initial_master)}
        for arm in ("default", "repair")
    }
    direction = torch.load(direction_path, map_location=device, weights_only=False)["direction"].float()
    direction_norm = direction.norm()

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__(); self.subject = subject
        def forward(self, input_ids, labels):
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm = candidate(*states[0]); warm.backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])
    warmed_symbols = observation.discover_all_triton_symbols(modules)
    full_rows = json.loads(campaign_path.read_text())["rows"]
    by_id = {row["region_id"]: row for row in full_rows}
    symbols = {by_id[region]["symbol"] for region in REGIONS}
    selected = [row for row in full_rows if row["symbol"] in symbols]
    campaign_rows, unmatched = observation.remap_campaign_to_warmed_symbols(
        selected, warmed_symbols, allow_extra_same_stem=True
    )
    if not set(REGIONS) <= {row["region_id"] for row in campaign_rows}:
        raise RuntimeError("complete key F+B regions are absent after exact remapping")

    def set_master(value):
        with torch.no_grad():
            parameter.copy_(value.to(dtype=parameter.dtype))

    def run(inputs, repair):
        model.zero_grad(set_to_none=True)
        if repair:
            observer = GeneratedTritonReferenceObserver(
                modules=modules, campaign_rows=campaign_rows, sequence=1024,
                intervene_region_ids=REGIONS,
            )
            with observer:
                loss = candidate(*inputs); loss.backward()
            if set(observer.summary()["intervention"]["observed_region_ids"]) != set(REGIONS):
                raise RuntimeError("complete key F+B intervention was not observed")
        else:
            loss = candidate(*inputs); loss.backward()
        torch.cuda.synchronize(device)
        return float(loss.detach().float().cpu()), parameter.grad.detach().float().clone()

    records = []
    for step, inputs in enumerate(states):
        same_weight = {}
        selected_gradients = {}
        for arm in ("default", "repair"):
            set_master(masters[arm])
            baseline_loss, baseline_gradient = run(inputs, False)
            repaired_loss, repaired_gradient = run(inputs, True)
            tile_delta = baseline_gradient[ROWS, COLUMNS] - repaired_gradient[ROWS, COLUMNS]
            same_weight[arm] = {
                "baseline_loss": baseline_loss,
                "repaired_loss": repaired_loss,
                "gradient_removal_projection": float(torch.dot(tile_delta.reshape(-1), direction.reshape(-1)) / direction_norm),
                "gradient_removal_l2": float(tile_delta.norm()),
            }
            selected_gradients[arm] = baseline_gradient if arm == "default" else repaired_gradient
        with torch.no_grad():
            beta1, beta2, epsilon = 0.9, 0.95, 1e-8
            for arm in ("default", "repair"):
                gradient = selected_gradients[arm]
                moments[arm]["m"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                moments[arm]["v"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                m_hat = moments[arm]["m"] / (1.0 - beta1 ** (step + 1))
                v_hat = moments[arm]["v"] / (1.0 - beta2 ** (step + 1))
                masters[arm].addcdiv_(m_hat, v_hat.sqrt().add_(epsilon), value=-args.learning_rate)
        master_delta = masters["repair"][ROWS, COLUMNS] - masters["default"][ROWS, COLUMNS]
        default_bf16 = masters["default"][ROWS, COLUMNS].to(torch.bfloat16).float()
        repair_bf16 = masters["repair"][ROWS, COLUMNS].to(torch.bfloat16).float()
        materialized_delta = repair_bf16 - default_bf16
        records.append({
            "step": step + 1,
            "state_index": step + 8,
            "token_sha256": evaluation["token_sha256"][step + 8],
            "same_weight": same_weight,
            "fp32_master_projection": float(torch.dot(master_delta.reshape(-1), direction.reshape(-1)) / direction_norm),
            "fp32_master_l2": float(master_delta.norm()),
            "bf16_materialized_projection": float(torch.dot(materialized_delta.reshape(-1), direction.reshape(-1)) / direction_norm),
            "bf16_materialized_nonzero": int(torch.count_nonzero(materialized_delta)),
        })
        print(json.dumps({"step": step + 1, "master_projection": records[-1]["fp32_master_projection"]}), flush=True)

    result = {
        "schema": "kernel-analyzer-l23-key-live-weight-v1",
        "status": "COMPLETE",
        "checkpoint_step": args.checkpoint_step,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0},
        "updated_parameter": PARAMETER,
        "other_parameters_updated": False,
        "arms": {"default": "Inductor", "repair": "Inductor with complete layer23 key F+B reference replacement"},
        "region_ids": REGIONS,
        "state_indices": list(range(8, 40)),
        "candidate_restoration_sham": "results/final/l23_key_forward_backward_sham_s8.json",
        "unmatched_warmed_symbols": unmatched,
        "records": records,
        "gates": {
            "same_initial_fp32_master": True,
            "same_state_order": True,
            "same_weight_baseline_and_repair_measured_each_arm_each_step": True,
            "only_q_proj_live_weight_updated": True,
            "final_fp32_master_divergence_nonzero": records[-1]["fp32_master_l2"] > 0.0,
            "bf16_live_weight_feedback_observed": any(row["bf16_materialized_nonzero"] > 0 for row in records),
            "tensor_values_saved": False,
        },
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"output": str(output_path), "final": records[-1]}, sort_keys=True))


if __name__ == "__main__":
    main()
