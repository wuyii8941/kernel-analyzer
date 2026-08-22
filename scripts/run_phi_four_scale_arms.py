#!/usr/bin/env python3
"""Run four comparable 32-step perturbation arms on the Phi lm-head carrier.

The experiment deliberately updates only the already closed final-norm carrier.
Consequently the precision arm is a full-model F+B precision contrast observed
through one evolving parameter, not a claim about full-parameter training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "src"))

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


def metric(increments: list[torch.Tensor], initial: torch.Tensor,
           final_left: torch.Tensor, final_right: torch.Tensor) -> dict[str, Any]:
    if not increments:
        raise ValueError("empty arm")
    resultant = torch.stack(increments).double().sum(0)
    energy = sum(float(torch.dot(v.double(), v.double())) for v in increments)
    drift = (final_left - final_right).double()
    drift_l2 = float(torch.linalg.vector_norm(drift))
    step_scale = math.sqrt(max(energy, 0.0))
    return {
        "final_distance_l2": drift_l2,
        "coherence_amplification": drift_l2 / max(step_scale, 1e-30),
        "summed_increment_l2": float(torch.linalg.vector_norm(resultant)),
        "telescoping_residual_l2": float(torch.linalg.vector_norm(resultant - drift)),
        "diffusive_step_scale": step_scale,
        "distance_over_diffusive_step_scale": drift_l2 / max(step_scale, 1e-30),
        "distance_over_initial_parameter_l2": drift_l2 / max(
            float(torch.linalg.vector_norm(initial.double())), 1e-30
        ),
        "zero_perturbation": energy == 0.0,
    }


def prefix_curve(increments: list[torch.Tensor]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for horizon in (2, 4, 8, 16, 32):
        if horizon > len(increments):
            continue
        selected = increments[:horizon]
        resultant = torch.stack(selected).double().sum(0)
        energy = sum(float(torch.dot(v.double(), v.double())) for v in selected)
        scale = math.sqrt(max(energy, 0.0))
        distance = float(torch.linalg.vector_norm(resultant))
        rows.append({
            "horizon": horizon,
            "distance_l2": distance,
            "diffusive_step_scale": scale,
            "coherence_amplification": distance / max(scale, 1e-30),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=(2, 8, 16, 32), default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bf16-device", default="cuda:0")
    parser.add_argument("--fp32-device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=(
        ROOT / "results/property/joint_bias_formation_v1/four_scale_arms/phi_lmhead.json"
    ))
    args = parser.parse_args()
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = list(bank.get("states", bank.get("records")))[:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("frozen input bank is incomplete")
    # A deterministic derangement; the multiset is exactly unchanged.
    order = list(range(args.steps))
    alternate_order = order[args.steps // 2:] + order[:args.steps // 2]

    bf16_device = torch.device(args.bf16_device)
    fp32_device = torch.device(args.fp32_device)
    configure_candidate_runtime(24000)
    bf16_model = load_model(
        "phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), bf16_device
    )
    bf16_model.eval()
    start = len(PyCodeCache.modules)
    bf16_step = torch.compile(LossStep(bf16_model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=bf16_device)
    bf16_model.zero_grad(set_to_none=True); bf16_step(warm).backward()
    torch.cuda.synchronize(bf16_device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((
        ROOT / "results/coverage/runtime_releases/phi4_seq64_r1/capture.json"
    ).read_text())
    validate_release(wrapper_modules(modules), capture)
    bf16_parameter = bf16_model.model.norm.weight

    # The FP32 arm uses the same stored checkpoint and eager F+B.  Copy the
    # BF16-materialised carrier into it so both carrier masters start bitwise
    # equal; the rest of the checkpoint is fixed in both arms.
    fp32_model = AutoModelForCausalLM.from_pretrained(
        "/data1/tzh/models/microsoft/Phi-4-mini-instruct", dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).to(fp32_device).train()
    fp32_model.config.use_cache = False
    fp32_step = LossStep(fp32_model)
    fp32_parameter = fp32_model.model.norm.weight
    initial = bf16_parameter.detach().float().cpu().clone()
    with torch.no_grad():
        fp32_parameter.copy_(initial.to(fp32_device))

    masters = {
        "a_candidate": initial.clone(), "a_repair": initial.clone(),
        "b_seed0": initial.clone(), "b_seed1": initial.clone(),
        "c_order0": initial.clone(), "c_order1": initial.clone(),
        "d_bf16": initial.clone(), "d_fp32": initial.clone(),
    }
    increments: dict[str, list[torch.Tensor]] = {key: [] for key in ("A", "B", "C", "D")}
    rows = []

    def bf16_gradient(master: torch.Tensor, state: dict[str, Any], seed: int,
                      repair: bool = False) -> torch.Tensor:
        torch.cuda.set_device(bf16_device)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            bf16_parameter.copy_(master.to(bf16_device, dtype=bf16_parameter.dtype))
        bf16_model.zero_grad(set_to_none=True)
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=bf16_device)
        observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
        if observer is None:
            bf16_step(values).backward()
        else:
            with observer:
                bf16_step(values).backward()
        torch.cuda.synchronize(bf16_device)
        return bf16_parameter.grad.detach().float().cpu().clone()

    def fp32_gradient(master: torch.Tensor, state: dict[str, Any], seed: int) -> torch.Tensor:
        torch.cuda.set_device(fp32_device)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            fp32_parameter.copy_(master.to(fp32_device))
        fp32_model.zero_grad(set_to_none=True)
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=fp32_device)
        fp32_step(values).backward(); torch.cuda.synchronize(fp32_device)
        return fp32_parameter.grad.detach().cpu().clone()

    def advance(name: str, gradient: torch.Tensor) -> torch.Tensor:
        before = masters[name]
        after = before - args.learning_rate * gradient
        realized = after - before
        masters[name] = after
        return realized

    for index in range(args.steps):
        seed = 24000 + index
        state = states[index]
        alternate_state = states[alternate_order[index]]
        # A: exact candidate / matched endpoint repair.
        a0 = advance("a_candidate", bf16_gradient(masters["a_candidate"], state, seed, False))
        a1 = advance("a_repair", bf16_gradient(masters["a_repair"], state, seed, True))
        increments["A"].append(a0 - a1)
        # B: same checkpoint and data, only RNG differs.  Phi has dropout=0;
        # a zero result is reported as NOT_INFORMATIVE rather than a negative.
        b0 = advance("b_seed0", bf16_gradient(masters["b_seed0"], state, seed, False))
        b1 = advance("b_seed1", bf16_gradient(masters["b_seed1"], state, seed + 1_000_000, False))
        increments["B"].append(b0 - b1)
        # C: identical batch multiset, fixed different temporal order.
        c0 = advance("c_order0", bf16_gradient(masters["c_order0"], state, seed, False))
        c1 = advance("c_order1", bf16_gradient(masters["c_order1"], alternate_state, seed, False))
        increments["C"].append(c0 - c1)
        # D: full-model BF16 versus full-model FP32 F+B, same carrier master.
        d0 = advance("d_bf16", bf16_gradient(masters["d_bf16"], state, seed, False))
        d1 = advance("d_fp32", fp32_gradient(masters["d_fp32"], state, seed))
        increments["D"].append(d0 - d1)
        rows.append({
            "step": index + 1,
            "state_id": str(state.get("state_id", state.get("sequence_id", index))),
            "alternate_state_id": str(alternate_state.get(
                "state_id", alternate_state.get("sequence_id", alternate_order[index])
            )),
            "increment_l2": {
                key: float(torch.linalg.vector_norm(increments[key][-1])) for key in increments
            },
        })
        print(json.dumps({"event": "FOUR_SCALE_STEP", **rows[-1]}), flush=True)

    pairs = {
        "A": ("a_candidate", "a_repair"), "B": ("b_seed0", "b_seed1"),
        "C": ("c_order0", "c_order1"), "D": ("d_bf16", "d_fp32"),
    }
    metrics = {
        arm: metric(increments[arm], initial, masters[left], masters[right])
        for arm, (left, right) in pairs.items()
    }
    for arm in metrics:
        metrics[arm]["prefix_curve"] = prefix_curve(increments[arm])
    metrics["B"]["interpretation"] = (
        "NOT_INFORMATIVE_ZERO_RNG_SENSITIVITY" if metrics["B"]["zero_perturbation"]
        else "MEASURED_RNG_SENSITIVITY"
    )
    payload = {
        "schema": "kernel-analyzer-four-scale-arms-v1",
        "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
        "case": "phi4_seq64_lmhead_dx",
        "steps": args.steps,
        "updated_parameter": "model.norm.weight",
        "only_declared_parameter_updated": True,
        "arms": {
            "A_operator": metrics["A"], "B_rng": metrics["B"],
            "C_data_order": metrics["C"], "D_precision": metrics["D"],
        },
        "records": rows,
        "protocol": {
            "optimizer": "SGD_FP32_MASTER", "learning_rate": args.learning_rate,
            "data_order_permutation": alternate_order,
            "same_starting_carrier": True,
            "precision_arm": "full-model BF16-vs-FP32 F+B; selected carrier updated",
        },
        "claim_boundary": (
            "All arms use the same initial checkpoint and frozen batch multiset, and are "
            "compared on the same final-norm carrier. Only that carrier evolves. Therefore "
            "this is a controlled carrier-scale comparison, not full-parameter training."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "arms": payload["arms"]}, sort_keys=True))


if __name__ == "__main__":
    main()
