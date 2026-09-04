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
from scripts.run_heldout_lmhead_consequence import adam_delta
from scripts.run_training_bias_profile_v2_empirical import (
    _append_contrast,
    _finish_stages,
)


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
    parser.add_argument(
        "--training-bias-profile-v2-output",
        type=Path,
        help=(
            "Also write the unified gradient/update profile. This requires a "
            "32-state input bank and is a method-bridge measurement, not a new "
            "prospective case selection."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    reach = json.loads(args.exact_reach.read_text())["results"]
    bank_payload = json.loads(args.input_bank.read_text())
    if args.training_bias_profile_v2_output is not None:
        states = list(bank_payload["states"])
        if len(states) != 32:
            raise RuntimeError("the unified Gemma profile requires exactly 32 states")
    else:
        states = [row for row in bank_payload["states"] if row["role"] == "CONFIRMATION"]
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
    profile_outputs = []
    for target in reach:
        changed = target["exact_changed_parameter_gradients"]
        carrier_name = min(changed, key=lambda name: (parameters[name].numel(), name))
        carrier = parameters[carrier_name]; total = torch.zeros(carrier.shape, dtype=torch.float32)
        odd = torch.zeros_like(total); even = torch.zeros_like(total); energy = 0.0; rows = []
        profile_store = {
            "PARAMETER_GRADIENT": {},
            "ADAMW_UPDATE": {},
        }
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
            repair_gradient = carrier.grad.detach().float().cpu().clone()
            delta = repair_gradient - baseline
            total.add_(delta); (even if index % 2 == 0 else odd).add_(delta)
            state_energy = square(delta); energy += state_energy
            rows.append({"state_id": state["state_id"], "delta_l2": state_energy**0.5})
            if args.training_bias_profile_v2_output is not None:
                # The unified convention is candidate minus repair.  AdamW is
                # evaluated from the same zero-initialized moments for both
                # implementations, matching the other cold-start profiles.
                effect_gradient = baseline - repair_gradient
                zeros = torch.zeros_like(repair_gradient)
                candidate_update, _, _ = adam_delta(
                    baseline, zeros, zeros, 1,
                    learning_rate=1.0e-4, beta1=0.9, beta2=0.95,
                )
                repair_update, _, _ = adam_delta(
                    repair_gradient, zeros, zeros, 1,
                    learning_rate=1.0e-4, beta1=0.9, beta2=0.95,
                )
                _append_contrast(
                    profile_store, "PARAMETER_GRADIENT",
                    effect_gradient, repair_gradient,
                )
                _append_contrast(
                    profile_store, "ADAMW_UPDATE",
                    candidate_update - repair_update, repair_update,
                )
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
        if args.training_bias_profile_v2_output is not None:
            profile_outputs.append({
                "case_id": (
                    "gemma4_e2b_" + target["region_id"].replace(":", "_")
                    + "_" + target["endpoint"]
                ),
                "model": "google/gemma-4-E2B",
                "implementation_relation": "NEW_IMPL_NEW_MODEL_METHOD_BRIDGE",
                "region_id": target["region_id"],
                "operation": target["operation"],
                "endpoint": target["endpoint"],
                "target_parameter": carrier_name,
                "optimizer": {
                    "name": "AdamW", "lr": 1.0e-4,
                    "betas": [0.9, 0.95], "epsilon": 1.0e-8,
                    "weight_decay": 0.0, "moments": "ZERO_AT_EVERY_INPUT_STATE",
                },
                "stages": _finish_stages(profile_store),
            })
    payload = {
        "schema": "kernel-analyzer-gemma4-recall-formation-v1",
        "status": "COMPLETE_COMPLETE_COORDINATES", "results": outputs,
        "claim_boundary": "Deterministic smallest reached parameter; formation only, no trajectory labels.",
    }
    payload["result_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.training_bias_profile_v2_output is not None:
        profile_payload = {
            "schema": "kernel-analyzer-gemma4-training-bias-profile-v2-method-bridge-v1",
            "status": "COMPLETE_METHOD_BRIDGE_NOT_PROSPECTIVE_CASE_SELECTION",
            "input_bank": str(args.input_bank),
            "cases": profile_outputs,
            "claim_boundary": (
                "The implementation positions were known from earlier Gemma work. "
                "This run tests whether the frozen unified statistics transfer to "
                "a new model and implementation family; it is not a fresh discovery set."
            ),
        }
        args.training_bias_profile_v2_output.parent.mkdir(parents=True, exist_ok=True)
        args.training_bias_profile_v2_output.write_text(
            json.dumps(profile_payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__": main()
