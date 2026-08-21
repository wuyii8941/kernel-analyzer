#!/usr/bin/env python3
"""Two-state eager/Inductor resource and execution preflight for a TCMP text cell."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor/tcmp_allop_v1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, Mistral3ForConditionalGeneration


VRAM_LIMIT = 44 * 1024**3


class LossStep(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=values, labels=values, use_cache=False, return_dict=True)
        if output.loss is None:
            raise RuntimeError("model returned no teacher-forced loss")
        return output.loss


def finite_gradients(model: torch.nn.Module) -> tuple[bool, int, float]:
    present = 0
    square = 0.0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        present += 1
        value = parameter.grad.detach().float()
        finite = finite and bool(torch.isfinite(value).all())
        square += float(torch.sum(value.double() * value.double()).item())
    return finite, present, square**0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.model, args.input_bank, args.output.parent):
        resolved = path.resolve()
        if Path("/data1/tzh") not in (resolved, *resolved.parents):
            raise ValueError("TCMP preflight paths must remain under /data1/tzh")
    bank = json.loads(args.input_bank.read_text())
    rows = [row for row in bank["states"] if row["role"] == "ENGINEERING"]
    if len(rows) != 2:
        raise RuntimeError("preflight requires exactly two engineering states")
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device_index)
    torch.manual_seed(20260820); torch.cuda.manual_seed_all(20260820)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_type = json.loads((args.model / "config.json").read_text()).get("model_type")
    model_class = (
        Mistral3ForConditionalGeneration if model_type == "mistral3"
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager", trust_remote_code=False,
    ).to(device).train()
    model.config.use_cache = False
    eager = LossStep(model)
    compiled = torch.compile(eager, backend="inductor", fullgraph=False, dynamic=False)
    measurements = []
    start = time.monotonic()
    for state_index, row in enumerate(rows):
        values = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        for arm, runner in (("BF16_EAGER", eager), ("BF16_INDUCTOR", compiled)):
            torch.manual_seed(20260820 + state_index)
            torch.cuda.manual_seed_all(20260820 + state_index)
            model.zero_grad(set_to_none=True)
            loss = runner(values); loss.backward(); torch.cuda.synchronize(device)
            finite, present, norm = finite_gradients(model)
            measurements.append({
                "state_id": row["state_id"], "arm": arm,
                "loss": float(loss.detach()), "loss_finite": bool(torch.isfinite(loss)),
                "gradients_finite": finite, "parameter_gradients_present": present,
                "gradient_l2": norm,
            })
            del loss
        del values
    peak_allocated = torch.cuda.max_memory_allocated(device_index)
    peak_reserved = torch.cuda.max_memory_reserved(device_index)
    payload = {
        "schema": "kernel-analyzer-tcmp-text-cell-preflight-v1",
        "cell_id": args.cell_id,
        "status": "READY" if (
            all(row["loss_finite"] and row["gradients_finite"] for row in measurements)
            and peak_reserved <= VRAM_LIMIT
        ) else "RESOURCE_OR_EXECUTION_BLOCKED",
        "model": str(args.model.resolve()), "input_bank": str(args.input_bank.resolve()),
        "measurements": measurements,
        "resources": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "vram_limit_bytes": VRAM_LIMIT,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_seconds": time.monotonic() - start,
        },
        "scientific_verdict_emitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": payload["status"], **payload["resources"]}))
    del compiled, eager, model
    gc.collect(); torch.cuda.empty_cache()
    if payload["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
