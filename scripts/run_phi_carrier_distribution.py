#!/usr/bin/env python3
"""Measure 32-step Phi lm-head source persistence on frozen carriers.

Each carrier receives an independent candidate/repair trajectory.  Exactly one
declared parameter evolves in each trajectory and all other model parameters
remain at the checkpoint.  This is deliberately a carrier-scale distribution,
not full-parameter training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "archive/round1_code/src"), str(ROOT / "src")]

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


def summarize(increments: list[torch.Tensor]) -> dict[str, Any]:
    if len(increments) != 32:
        raise ValueError("strict carrier distribution requires 32 increments")
    rows = []
    for horizon in (2, 4, 8, 16, 32):
        selected = increments[:horizon]
        resultant = torch.stack(selected).double().sum(0)
        energy = math.fsum(float(torch.dot(row.double(), row.double())) for row in selected)
        scale = math.sqrt(max(energy, 0.0))
        distance = float(torch.linalg.vector_norm(resultant))
        rows.append({
            "horizon": horizon,
            "coherence_amplification": distance / max(scale, 1e-30),
            "resultant_l2": distance,
            "diffusive_step_scale": scale,
        })
    final = rows[-1]
    return {
        "coherence_amplification": final["coherence_amplification"],
        "resultant_l2": final["resultant_l2"],
        "diffusive_step_scale": final["diffusive_step_scale"],
        "prefix_curve": rows,
        "zero_effect": final["diffusive_step_scale"] == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=ROOT / "results/coverage/runtime_releases/phi4_seq64_r1",
        help="runtime release produced in the same environment as this run",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--steps", type=int, choices=(2, 32), default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=None,
        help="optional /data1 directory for final candidate/repair carrier masters",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    selected = [
        row for row in manifest["carriers"]
        if int(row["index"]) % args.shard_count == args.shard
    ]
    if not selected:
        raise ValueError("empty carrier shard")
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = list(bank.get("states", bank.get("records")))[:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("frozen input bank is incomplete")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    configure_candidate_runtime(24000)
    process_start = time.monotonic()
    model = load_model(
        "phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device
    )
    model.eval()
    start = len(PyCodeCache.modules)
    step = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    step(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.release_dir / "capture.json").read_text())
    validate_release(wrapper_modules(modules), capture)
    parameters = dict(model.named_parameters())
    missing = [row["carrier"] for row in selected if row["carrier"] not in parameters]
    if missing:
        raise RuntimeError(f"frozen carriers are absent: {missing}")

    records = []
    if args.final_dir is not None:
        args.final_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        name = item["carrier"]
        parameter = parameters[name]
        original = parameter.detach().float().clone()
        candidate_master = original.clone()
        repair_master = original.clone()
        increments: list[torch.Tensor] = []
        step_rows = []
        carrier_start = time.monotonic()

        def gradient(master: torch.Tensor, state: dict[str, Any], repair: bool) -> torch.Tensor:
            with torch.no_grad():
                parameter.copy_(master.to(parameter.dtype))
            model.zero_grad(set_to_none=True)
            values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
            observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
            if observer is None:
                step(values).backward()
            else:
                with observer:
                    step(values).backward()
            torch.cuda.synchronize(device)
            if parameter.grad is None:
                raise RuntimeError(f"declared carrier gradient is absent: {name}")
            return parameter.grad.detach().float().clone()

        for index, state in enumerate(states):
            gc = gradient(candidate_master, state, False)
            gr = gradient(repair_master, state, True)
            next_candidate = candidate_master - args.learning_rate * gc
            next_repair = repair_master - args.learning_rate * gr
            increment = (next_candidate - candidate_master) - (next_repair - repair_master)
            increments.append(increment.detach().cpu())
            candidate_master, repair_master = next_candidate, next_repair
            step_rows.append({
                "step": index + 1,
                "state_id": str(state.get("state_id", state.get("sequence_id", index))),
                "increment_l2": float(torch.linalg.vector_norm(increment)),
            })
        with torch.no_grad():
            parameter.copy_(original.to(parameter.dtype))
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - carrier_start
        final_masters = None
        if args.final_dir is not None and args.steps == 32:
            safe_name = name.replace(".", "__")
            candidate_path = args.final_dir / f"{safe_name}.candidate.pt"
            repair_path = args.final_dir / f"{safe_name}.repair.pt"
            torch.save(candidate_master.cpu(), candidate_path)
            torch.save(repair_master.cpu(), repair_path)
            final_masters = {
                "candidate_path": str(candidate_path),
                "repair_path": str(repair_path),
                "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                "repair_sha256": hashlib.sha256(repair_path.read_bytes()).hexdigest(),
            }
        result = {
            **item,
            "coordinates": parameter.numel(),
            "measurement": summarize(increments) if args.steps == 32 else None,
            "elapsed_seconds": elapsed,
            "f_and_b_calls": 2 * args.steps,
            "final_masters": final_masters,
            "records": step_rows,
        }
        records.append(result)
        print(json.dumps({
            "event": "PHI_CARRIER_COMPLETE",
            "carrier": name,
            "A": None if result["measurement"] is None else result["measurement"]["coherence_amplification"],
            "elapsed_seconds": elapsed,
        }), flush=True)

    payload = {
        "schema": "kernel-analyzer-phi-carrier-distribution-shard-v1",
        "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
        "manifest": str(args.manifest),
        "runtime_release": str(args.release_dir),
        "selection_sha256": manifest["selection_sha256"],
        "shard": args.shard,
        "shard_count": args.shard_count,
        "device": str(device),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "optimizer": "SGD_FP32_MASTER",
        "only_one_declared_carrier_evolves_per_trajectory": True,
        "rows": records,
        "elapsed_seconds": time.monotonic() - process_start,
        "claim_boundary": (
            "Each row is an independent one-parameter candidate/repair trajectory. "
            "The sample is outcome-blind and carrier-scale, not full-parameter training."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(records)}))


if __name__ == "__main__":
    main()
