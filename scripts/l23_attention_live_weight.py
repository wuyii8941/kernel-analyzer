#!/usr/bin/env python3
"""Paired 32-step live-weight trajectory for the complete S_bwd/K repair."""

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


PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280)
COLUMNS = slice(1664, 1792)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/final/l23_attention_live_weight.json"))
    parser.add_argument("--checkpoint-step", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps != 32 or args.learning_rate != 1e-5:
        raise ValueError("frozen protocol requires 32 steps and learning rate 1e-5")

    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
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
    import torch.nn.functional as F
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.select_algorithm import extern_kernels
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    device = torch.device(args.device)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    bank = json.loads(bank_path.read_text())
    milestone = next(row for row in bank["milestones"] if int(row["step"]) == args.checkpoint_step)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    all_states, evaluation = load_eval_states(tokenizer, 1024, 40, device)
    states = all_states[8:40]
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)
    parameter = dict(model.named_parameters())[PARAMETER]
    initial_master = parameter.detach().float().clone()
    masters = {arm: initial_master.clone() for arm in ("default", "repair")}
    moments = {
        arm: {"m": torch.zeros_like(initial_master), "v": torch.zeros_like(initial_master)}
        for arm in masters
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
    matches = []
    for module in modules:
        path = Path(module.__file__)
        source = path.read_text()
        if "bmm_76]" in source and "mm_267" in source:
            matches.append((path, source))
    if len(matches) != 1:
        raise RuntimeError(f"expected one bmm_76 backward source, got {len(matches)}")
    source_path, source = matches[0]
    marker = source.index("bmm_76]")
    call_start = source.rfind("def call(", 0, marker)
    target_ordinal = source[call_start:marker].count("extern_kernels.bmm(")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    original_bmm = extern_kernels.bmm
    original_attention = modeling_qwen3.eager_attention_forward
    eager_capture = {}

    def captured_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
        raw_scores = torch.matmul(query, key_states.transpose(2, 3))
        if module.layer_idx == 23:
            eager_capture["K"] = key_states.detach().reshape(16, 1024, 128).clone()
            def raw_hook(gradient):
                eager_capture["S"] = gradient.detach().reshape(16, 1024, 1024).clone()
                return gradient
            raw_scores.register_hook(raw_hook)
        weights = raw_scores * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :key_states.shape[-2]]
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output, weights

    def set_master(value):
        with torch.no_grad():
            parameter.copy_(value.to(parameter.dtype))

    def eager_operands(inputs):
        eager_capture.clear()
        model.zero_grad(set_to_none=True)
        modeling_qwen3.eager_attention_forward = captured_attention
        try:
            loss = model(input_ids=inputs[0], labels=inputs[1], use_cache=False, return_dict=False)[0]
            loss.backward()
            torch.cuda.synchronize(device)
        finally:
            modeling_qwen3.eager_attention_forward = original_attention
        if set(eager_capture) != {"S", "K"}:
            raise RuntimeError("eager S/K capture failed")
        return float(loss.detach().float().cpu()), eager_capture["S"], eager_capture["K"]

    def candidate_gradient(inputs, replacement=None):
        model.zero_grad(set_to_none=True)
        loss = candidate(*inputs)
        if replacement is None:
            loss.backward(); torch.cuda.synchronize(device)
            return float(loss.detach().float().cpu()), parameter.grad.detach().float().clone()
        counter = {"bmm": 0}; observed = {"target": False}
        def wrapped_bmm(*values, **kwargs):
            ordinal = counter["bmm"]; counter["bmm"] += 1
            if ordinal != target_ordinal:
                return original_bmm(*values, **kwargs)
            left, right = replacement
            out = kwargs.get("out")
            if tuple(out.shape) != (16, 1024, 128):
                raise RuntimeError("bmm_76 output shape mismatch")
            observed["target"] = True
            return original_bmm(left, right, out=out)
        extern_kernels.bmm = wrapped_bmm
        try:
            loss.backward(); torch.cuda.synchronize(device)
        finally:
            extern_kernels.bmm = original_bmm
        if not observed["target"]:
            raise RuntimeError("bmm_76 repair was not observed")
        return float(loss.detach().float().cpu()), parameter.grad.detach().float().clone()

    records = []
    for step, inputs in enumerate(states):
        same_weight = {}; selected = {}
        for arm in ("default", "repair"):
            set_master(masters[arm])
            eager_loss, reference_s, reference_k = eager_operands(inputs)
            baseline_loss, baseline_gradient = candidate_gradient(inputs)
            repaired_loss, repaired_gradient = candidate_gradient(inputs, (reference_s, reference_k))
            tile_delta = baseline_gradient[ROWS, COLUMNS] - repaired_gradient[ROWS, COLUMNS]
            same_weight[arm] = {
                "eager_loss": eager_loss,
                "baseline_loss": baseline_loss,
                "repaired_loss": repaired_loss,
                "gradient_removal_projection": float(torch.dot(tile_delta.reshape(-1), direction.reshape(-1)) / direction_norm),
                "gradient_removal_l2": float(tile_delta.norm()),
            }
            selected[arm] = baseline_gradient if arm == "default" else repaired_gradient
        with torch.no_grad():
            beta1, beta2, epsilon = 0.9, 0.95, 1e-8
            for arm in masters:
                gradient = selected[arm]
                moments[arm]["m"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                moments[arm]["v"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                m_hat = moments[arm]["m"] / (1.0 - beta1 ** (step + 1))
                v_hat = moments[arm]["v"] / (1.0 - beta2 ** (step + 1))
                masters[arm].addcdiv_(m_hat, v_hat.sqrt().add_(epsilon), value=-args.learning_rate)
        master_delta = masters["repair"][ROWS, COLUMNS] - masters["default"][ROWS, COLUMNS]
        materialized_delta = (
            masters["repair"][ROWS, COLUMNS].to(torch.bfloat16).float()
            - masters["default"][ROWS, COLUMNS].to(torch.bfloat16).float()
        )
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
        "schema": "kernel-analyzer-l23-attention-live-weight-v1",
        "status": "COMPLETE",
        "checkpoint_step": args.checkpoint_step,
        "steps": args.steps,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0},
        "updated_parameter": PARAMETER,
        "other_parameters_updated": False,
        "repair_boundary": "actual bmm_76: G_q = S_bwd @ K",
        "binding": {"target_ordinal": target_ordinal, "generated_source_sha256": source_sha256},
        "state_indices": list(range(8, 40)),
        "records": records,
        "gates": {
            "same_initial_fp32_master": True,
            "same_state_order": True,
            "same_weight_baseline_and_repair_measured_each_arm_each_step": True,
            "only_q_proj_live_weight_updated": True,
            "final_fp32_master_divergence_nonzero": records[-1]["fp32_master_l2"] > 0,
            "bf16_live_weight_feedback_observed": any(row["bf16_materialized_nonzero"] > 0 for row in records),
            "tensor_values_saved": False,
        },
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"output": str(output_path), "final": records[-1]}, sort_keys=True))


if __name__ == "__main__":
    main()
