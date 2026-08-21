#!/usr/bin/env python3
"""Exact-gradient closure for positive Gemma-4 random recall reach screens."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from scripts.generated_fp32_observer import GeneratedFP32Observer
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules
from scripts.run_generated_fp32_screen import load_model


def digest(value: torch.Tensor) -> str:
    result = hashlib.sha256()
    result.update(str(value.dtype).encode()); result.update(repr(tuple(value.shape)).encode())
    flat = value.detach().reshape(-1)
    for start in range(0, flat.numel(), 1 << 22):
        chunk = flat[start:start + (1 << 22)].contiguous().cpu()
        result.update(chunk.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def gradients(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: "NONE" if parameter.grad is None else digest(parameter.grad)
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--runtime-release", type=Path, required=True)
    parser.add_argument("--reach-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    targets = [
        row for row in json.loads(args.reach_screen.read_text())["results"]
        if row["parameter_reach_screen"]
    ]
    bank = json.loads(args.input_bank.read_text())
    state = next(row for row in bank["states"] if row["role"] == "ENGINEERING")
    device = torch.device(args.device); configure_candidate_runtime(20260821)
    model = load_model("gemma4", args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(values).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.runtime_release / "capture.json").read_text())
    observed = [
        hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()
        for module, _ in wrapper_modules(modules)
    ]
    if observed != [row["sha256"] for row in capture["modules"]]:
        raise RuntimeError("runtime wrapper bytes differ from frozen release")
    with gzip.open(args.runtime_release / "campaign.json.gz", "rt") as handle:
        campaign = {row["region_id"]: row for row in json.load(handle)["rows"]}
    seed = 20260821
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True); candidate(values).backward(); torch.cuda.synchronize(device)
    baseline = gradients(model)
    results = []
    for target in targets:
        row = campaign[target["region_id"]]
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = GeneratedFP32Observer(
            modules=modules, campaign_rows=[row],
            repair_targets={row["region_id"]: [target["endpoint"]]}, allow_unlisted_calls=True,
        )
        with observer: candidate(values).backward()
        torch.cuda.synchronize(device)
        repaired = gradients(model)
        changed = sorted(name for name in baseline if baseline[name] != repaired[name])
        results.append({**target, "exact_changed_parameter_gradients": changed,
                        "exact_parameter_reachable": bool(changed)})
        print(json.dumps({"event": "GEMMA4_EXACT_REACH", "region": row["region_id"],
                          "changed": len(changed)}), flush=True)
    payload = {
        "schema": "kernel-analyzer-gemma4-random-recall-exact-reach-v1",
        "status": "COMPLETE_EXACT_PARAMETER_GRADIENT_DIGESTS",
        "results": results,
        "claim_boundary": "Exact reach only; formation and persistence require independent populations.",
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
