#!/usr/bin/env python3
"""Complete-coordinate formation for exact-positive Gemma-4 recall candidates."""

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


def square(value: torch.Tensor) -> float:
    result = 0.0; flat = value.reshape(-1)
    for start in range(0, flat.numel(), 1 << 22):
        norm = float(torch.linalg.vector_norm(flat[start:start + (1 << 22)]).item())
        result += norm * norm
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--runtime-release", type=Path, required=True)
    parser.add_argument("--exact-reach", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    reach = json.loads(args.exact_reach.read_text())["results"]
    states = [row for row in json.loads(args.input_bank.read_text())["states"] if row["role"] == "CONFIRMATION"]
    device = torch.device(args.device); configure_candidate_runtime(20260821)
    model = load_model("gemma4", args.model, device); parameters = dict(model.named_parameters())
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    bank = json.loads(args.input_bank.read_text())
    warm = torch.tensor([bank["states"][0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.runtime_release / "capture.json").read_text())
    observed = [hashlib.sha256(Path(m.__file__).resolve().read_bytes()).hexdigest() for m, _ in wrapper_modules(modules)]
    if observed != [row["sha256"] for row in capture["modules"]]: raise RuntimeError("wrapper mismatch")
    with gzip.open(args.runtime_release / "campaign.json.gz", "rt") as handle:
        campaign = {row["region_id"]: row for row in json.load(handle)["rows"]}
    outputs = []
    for target in reach:
        changed = target["exact_changed_parameter_gradients"]
        carrier_name = min(changed, key=lambda name: (parameters[name].numel(), name))
        carrier = parameters[carrier_name]; total = torch.zeros(carrier.shape, dtype=torch.float32)
        odd = torch.zeros_like(total); even = torch.zeros_like(total); energy = 0.0; rows = []
        region = campaign[target["region_id"]]
        for index, state in enumerate(states):
            values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
            seed = 20260821 + index
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True); candidate(values).backward(); torch.cuda.synchronize(device)
            baseline = carrier.grad.detach().float().cpu().clone()
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            observer = GeneratedFP32Observer(
                modules=modules, campaign_rows=[region],
                repair_targets={region["region_id"]: [target["endpoint"]]}, allow_unlisted_calls=True,
            )
            with observer: candidate(values).backward()
            torch.cuda.synchronize(device)
            delta = carrier.grad.detach().float().cpu() - baseline
            total.add_(delta); (even if index % 2 == 0 else odd).add_(delta)
            state_energy = square(delta); energy += state_energy
            rows.append({"state_id": state["state_id"], "delta_l2": state_energy**0.5})
            print(json.dumps({"event": "GEMMA4_RECALL_FORMATION", "region": region["region_id"], "step": index + 1}), flush=True)
        resultant_energy = square(total); odd_energy = square(odd); even_energy = square(even)
        inner = float(torch.sum(odd * even).item())
        outputs.append({
            "region_id": region["region_id"], "operation": target["operation"],
            "phase": target["phase"], "endpoint": target["endpoint"],
            "carrier": carrier_name, "carrier_coordinates": carrier.numel(),
            "complete_coordinate_statistics": {
                "path_energy": energy, "resultant_energy": resultant_energy,
                "coherence_amplification": (resultant_energy / max(energy, 1e-30))**0.5,
                "odd_even_resultant_cosine": inner / max((odd_energy * even_energy)**0.5, 1e-30),
            },
            "records": rows,
        })
    payload = {
        "schema": "kernel-analyzer-gemma4-recall-formation-v1",
        "status": "COMPLETE_COMPLETE_COORDINATES", "results": outputs,
        "claim_boundary": "Deterministic smallest reached parameter; formation only, no trajectory labels.",
    }
    payload["result_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
