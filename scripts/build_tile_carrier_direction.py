#!/usr/bin/env python3
"""Freeze the confirmed layer-23 q_proj tile direction from calibration states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.long_horizon_trigger import build_model, load_eval_states, run_backward, under_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = under_root(args.model, "model")
    output = under_root(args.output, "output")
    output.parent.mkdir(parents=True, exist_ok=True)
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
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    states, evaluation = load_eval_states(tokenizer, 1024, 8, device)
    model = build_model(model_path, device)
    parameter_name = "model.layers.23.self_attn.q_proj.weight"

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__()
            self.subject = subject

        def forward(self, input_ids, labels):
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    candidate = torch.compile(LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False)
    warm_loss, _ = run_backward(model, states[0], candidate)
    direction = None
    rows = []
    for state_id, inputs in enumerate(states):
        eager_loss, eager = run_backward(model, inputs)
        candidate_loss, compiled = run_backward(model, inputs, candidate)
        delta = (compiled[parameter_name].float() - eager[parameter_name].float())[1152:1280, 1664:1792]
        direction = torch.zeros_like(delta) if direction is None else direction
        direction.add_(delta)
        rows.append({"state_id": state_id, "eager_loss": eager_loss, "candidate_loss": candidate_loss})
    direction.div_(len(states))
    payload = {
        "schema": "kernel-analyzer-frozen-tile-direction-v1",
        "parameter": parameter_name,
        "row_start": 1152,
        "row_stop": 1280,
        "column_start": 1664,
        "column_stop": 1792,
        "direction": direction.cpu(),
        "direction_l2": float(direction.norm()),
        "calibration": {"checkpoint_step": 0, "evaluation": evaluation, "candidate_repeats": 1},
        "candidate": {"backend": "Inductor", "fullgraph": True, "dtype": "bfloat16", "tf32": False},
        "warm_loss": warm_loss,
        "loss_rows": rows,
        "source_confirmation": "results/final/structured_carrier_confirmation.json",
    }
    metadata = {key: value for key, value in payload.items() if key != "direction"}
    payload["metadata_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "direction_l2": payload["direction_l2"]}, sort_keys=True))


if __name__ == "__main__":
    main()
