#!/usr/bin/env python3
"""Four-arm 32-step consequence for the frozen Gemma-4 PLE/RMSNorm target."""

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


CARRIER = "model.language_model.per_layer_model_projection.weight"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--runtime-release", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--carrier", default=CARRIER)
    parser.add_argument("--region-id")
    parser.add_argument("--endpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 32), default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--optimizer", choices=("adamw", "sgd", "adamw_reset_moments"),
        default="adamw",
        help="Feedback intervention; all modes use the same declared F+B arms.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    prediction = json.loads(args.prediction.read_text())
    if prediction["status"] != "PREDICTION_FROZEN_BEFORE_TRAJECTORY":
        raise RuntimeError("prediction is not frozen")
    bank = json.loads(args.input_bank.read_text())
    states = [row for row in bank["states"] if row["role"] == "TRAJECTORY"][:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("trajectory population incomplete")
    device = torch.device(args.device)
    # Match the freeze runner's seed so any initialized auxiliary buffers and
    # compiler constants produce the same generated F+B wrapper bytes.
    configure_candidate_runtime(24000)
    model = load_model("gemma4", args.model, device)
    carrier = dict(model.named_parameters())[args.carrier]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    # Warm with the same first trajectory state used by the frozen release.
    # Using the base bank's first engineering state can change Gemma-4's
    # generated graph node ordering even when tensor shapes match.
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.runtime_release / "capture.json").read_text())
    observed = [
        hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()
        for module, _ in wrapper_modules(modules)
    ]
    if observed != [row["sha256"] for row in capture["modules"]]:
        raise RuntimeError(
            "runtime wrapper bytes differ from frozen release: "
            f"observed={observed} expected={[row['sha256'] for row in capture['modules']]}"
        )
    with gzip.open(args.runtime_release / "campaign.json.gz", "rt") as handle:
        campaign = json.load(handle)
    if args.region_id:
        target = next(row for row in campaign["rows"] if row["region_id"] == args.region_id)
        repair_endpoints = [args.endpoint] if args.endpoint else target["output_names"]
    else:
        target = sorted(
            (row for row in campaign["rows"] if row["phase"] == "FORWARD" and "embedding_mean_mul_pow_view" in row["symbol"]),
            key=lambda row: (row["source_path"], row["source_line"], row["region_id"]),
        )[0]
        repair_endpoints = target["output_names"]

    initial = carrier.detach().float().clone()
    cmaster = initial.clone(); rmaster = initial.clone()
    cm = torch.zeros_like(initial); cv = torch.zeros_like(initial)
    rm = torch.zeros_like(initial); rv = torch.zeros_like(initial)
    sums = {name: torch.zeros_like(initial) for name in ("local", "feedback", "actual")}
    energies = {name: 0.0 for name in sums}
    records = []

    def update(gradient_value: torch.Tensor, first: torch.Tensor,
               second: torch.Tensor, step: int):
        if args.optimizer == "sgd":
            return -args.learning_rate * gradient_value, first, second
        if args.optimizer == "adamw_reset_moments":
            zero_first = torch.zeros_like(first)
            zero_second = torch.zeros_like(second)
            delta, _, _ = adam_delta(
                gradient_value, zero_first, zero_second, 1,
                learning_rate=args.learning_rate,
            )
            return delta, zero_first, zero_second
        return adam_delta(
            gradient_value, first, second, step,
            learning_rate=args.learning_rate,
        )

    def gradient(master: torch.Tensor, state: dict, repair: bool) -> torch.Tensor:
        with torch.no_grad(): carrier.copy_(master.to(carrier.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        if repair:
            observer = GeneratedFP32Observer(
                modules=modules, campaign_rows=[target],
                repair_targets={target["region_id"]: repair_endpoints},
                allow_unlisted_calls=True,
            )
            with observer: candidate(values).backward()
        else:
            candidate(values).backward()
        torch.cuda.synchronize(device)
        return carrier.grad.detach().float().clone()

    for index, state in enumerate(states):
        step = index + 1
        gc_c = gradient(cmaster, state, False); gr_c = gradient(cmaster, state, True)
        gc_r = gradient(rmaster, state, False); gr_r = gradient(rmaster, state, True)
        uc_c, ncm, ncv = update(gc_c, cm, cv, step)
        ur_c, _, _ = update(gr_c, cm, cv, step)
        uc_r, _, _ = update(gc_r, rm, rv, step)
        ur_r, nrm, nrv = update(gr_r, rm, rv, step)
        next_c = cmaster + uc_c; next_r = rmaster + ur_r
        # Recompute realized master writes before the symmetric decomposition.
        uc_c = next_c - cmaster; ur_c = (cmaster + ur_c) - cmaster
        uc_r = (rmaster + uc_r) - rmaster; ur_r = next_r - rmaster
        local = 0.5 * ((uc_c - ur_c) + (uc_r - ur_r))
        feedback = 0.5 * ((uc_c - uc_r) + (ur_c - ur_r))
        actual = (next_c - next_r) - (cmaster - rmaster)
        residual = actual - local - feedback
        for name, value in (("local", local), ("feedback", feedback), ("actual", actual)):
            sums[name].add_(value); energies[name] += float(torch.sum(value * value).item())
        records.append({
            "step": step, "state_id": state["state_id"],
            "local_l2": float(torch.linalg.vector_norm(local)),
            "feedback_l2": float(torch.linalg.vector_norm(feedback)),
            "actual_l2": float(torch.linalg.vector_norm(actual)),
            "drift_l2": float(torch.linalg.vector_norm(next_c - next_r)),
            "recurrence_residual_l2": float(torch.linalg.vector_norm(residual)),
        })
        cmaster, rmaster, cm, cv, rm, rv = next_c, next_r, ncm, ncv, nrm, nrv
        print(json.dumps({"event": "GEMMA4_NORM_CONSEQUENCE_STEP", "step": step}), flush=True)
        del gc_c, gr_c, gc_r, gr_r, local, feedback, actual, residual
        torch.cuda.empty_cache()
    stats = {}
    for name in sums:
        resultant = float(torch.linalg.vector_norm(sums[name]))
        stats[name] = {
            "path_energy": energies[name], "resultant_l2": resultant,
            "coherence_amplification": resultant / max(energies[name]**0.5, 1e-30),
        }
    payload = {
        "schema": "kernel-analyzer-gemma4-norm-consequence-v1",
        "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
        "prediction": prediction, "steps": args.steps, "carrier": args.carrier,
        "optimizer": {"name": args.optimizer, "learning_rate": args.learning_rate},
        "target": {"region_id": target["region_id"], "endpoint": args.endpoint,
                   "symbol": target["symbol"]},
        "carrier_coordinates": carrier.numel(), "statistics": stats,
        "final_drift_l2": float(torch.linalg.vector_norm(cmaster - rmaster)),
        "records": records,
        "claim_boundary": "Four-arm symmetric one-parameter trajectory; no feedback prediction was made.",
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
