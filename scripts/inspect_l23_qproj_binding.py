#!/usr/bin/env python3
"""Record which generated matrix products actually write layer-23 q_proj.grad."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/l23_binding_inspection")

import torch
from torch._dynamo.backends.registry import lookup_backend
from torch._inductor.codecache import PyCodeCache
from torch._inductor.select_algorithm import extern_kernels
from transformers import AutoTokenizer

from scripts.long_horizon_trigger import build_model, load_eval_states, load_milestone, under_root


PARAMETER = "model.layers.23.self_attn.q_proj.weight"


def main() -> None:
    model_path = under_root(Path("/data1/tzh/models/Qwen/Qwen3-1.7B"), "model")
    bank_path = under_root(Path("results/final/long_horizon_bank.json"), "bank")
    output_path = under_root(Path("results/property/root_cause_closure_v1/l23_binding_inspection.json"), "output")
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    bank = json.loads(bank_path.read_text())
    milestone = next(row for row in bank["milestones"] if int(row["step"]) == 1024)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    all_states, _ = load_eval_states(tokenizer, 1024, 9, device)
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__()
            self.subject = subject

        def forward(self, input_ids, labels):
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    candidate = torch.compile(LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm = candidate(*all_states[8])
    warm.backward()
    torch.cuda.synchronize(device)

    q_weight = dict(model.named_parameters())[PARAMETER]
    q_storage = int(q_weight.untyped_storage().data_ptr())
    original_mm = extern_kernels.mm
    original_bmm = extern_kernels.bmm
    mm_rows = []
    bmm_rows = []

    def storage_ptr(value):
        try:
            return int(value.untyped_storage().data_ptr())
        except Exception:
            return None

    def wrapped_mm(*values, **kwargs):
        ordinal = len(mm_rows)
        left, right = values[:2]
        out = kwargs.get("out")
        row = {
            "ordinal": ordinal,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "out_shape": list(out.shape) if out is not None else None,
            "left_data_ptr_matches": int(left.data_ptr()) == int(q_weight.data_ptr()),
            "right_data_ptr_matches": int(right.data_ptr()) == int(q_weight.data_ptr()),
            "left_storage_matches": storage_ptr(left) == q_storage,
            "right_storage_matches": storage_ptr(right) == q_storage,
        }
        mm_rows.append(row)
        return original_mm(*values, **kwargs)

    def wrapped_bmm(*values, **kwargs):
        ordinal = len(bmm_rows)
        left, right = values[:2]
        out = kwargs.get("out")
        bmm_rows.append({
            "ordinal": ordinal,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "out_shape": list(out.shape) if out is not None else None,
        })
        return original_bmm(*values, **kwargs)

    extern_kernels.mm = wrapped_mm
    extern_kernels.bmm = wrapped_bmm
    try:
        model.zero_grad(set_to_none=True)
        loss = candidate(*all_states[8])
        loss.backward()
        torch.cuda.synchronize(device)
    finally:
        extern_kernels.mm = original_mm
        extern_kernels.bmm = original_bmm

    payload = {
        "schema": "kernel-analyzer-l23-qproj-binding-inspection-v1",
        "parameter": PARAMETER,
        "parameter_shape": list(q_weight.shape),
        "parameter_data_ptr": int(q_weight.data_ptr()),
        "matrix_calls": mm_rows,
        "batched_matrix_calls": bmm_rows,
        "gradient_norm": float(q_weight.grad.float().norm()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output_path.relative_to(REPO)), "mm_calls": len(mm_rows), "bmm_calls": len(bmm_rows)}))


if __name__ == "__main__":
    main()
