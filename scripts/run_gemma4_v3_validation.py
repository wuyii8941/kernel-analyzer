#!/usr/bin/env python3
"""Same-process Gemma held-out validation for the shared short screen.

The runtime release, formation measurement, frozen source prediction, and
short consequence all use the same compiled forward/backward wrappers.  This
avoids treating cross-process Inductor byte differences as a scientific result.
Only compact certificates are written; full vectors stay in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.short_persistence import SharedShortPersistenceScreen  # noqa: E402
from scripts.generated_fp32_observer import GeneratedFP32Observer  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import (  # noqa: E402
    freeze_or_validate_release,
    load_model,
    wrapper_modules,
)
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402


CARRIERS = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.per_layer_model_projection.weight",
)
TARGET_NEEDLE = "embedding_mean_mul_pow_view"


def l2(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def choose_target(campaign: dict) -> dict:
    choices = sorted(
        (
            row for row in campaign["rows"]
            if row["phase"] == "FORWARD" and TARGET_NEEDLE in row["symbol"]
        ),
        key=lambda row: (row["source_path"], row["source_line"], row["region_id"]),
    )
    if not choices:
        raise RuntimeError("frozen current release has no PLE/RMSNorm target")
    return choices[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument(
        "--consequence-bank", type=Path,
        help="Optional disjoint bank for the closed trajectory; if omitted, the formation states are reused only for engineering dry runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-seed", type=int, default=24000)
    parser.add_argument("--steps", type=int, choices=(2, 8, 16), default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--null-draws", type=int, default=2000)
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    all_states = bank["states"]
    warm_states = [row for row in all_states if row.get("role") == "ENGINEERING"]
    states = [row for row in all_states if row.get("role") == "CONFIRMATION"][:args.steps]
    if not warm_states or len(states) != args.steps:
        raise RuntimeError("Gemma bank lacks the required engineering/confirmation states")
    consequence_bank_path = args.consequence_bank or args.input_bank
    consequence_bank = json.loads(consequence_bank_path.read_text())
    consequence_states = [
        row for row in consequence_bank["states"]
        if row.get("role") == "TRAJECTORY"
    ] if args.consequence_bank else states
    if len(consequence_states) < args.steps:
        raise RuntimeError("consequence bank lacks the requested disjoint trajectory states")
    consequence_states = consequence_states[:args.steps]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    release_dir = output_dir / "runtime_release"
    device = torch.device(args.device)
    configure_candidate_runtime(args.runtime_seed)
    model = load_model("gemma4", args.model, device)
    parameters = dict(model.named_parameters())
    missing = sorted(set(CARRIERS) - set(parameters))
    if missing:
        raise RuntimeError(f"declared carrier absent: {missing}")

    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([warm_states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    wrappers = wrapper_modules(modules)
    inventory_path, campaign_path = freeze_or_validate_release(
        modules=wrappers,
        release=release_dir,
        architecture="gemma4",
        input_bank=args.input_bank,
        state=warm_states[0],
        allow_graph_breaks=True,
    )
    del inventory_path
    import gzip
    with gzip.open(campaign_path, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    target = choose_target(campaign)
    repair_targets = {target["region_id"]: target["output_names"]}

    def gradient(master: torch.Tensor, state: dict, repair: bool, carrier_name: str) -> torch.Tensor:
        parameter = parameters[carrier_name]
        with torch.no_grad():
            parameter.copy_(master.to(parameter.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        if repair:
            observer = GeneratedFP32Observer(
                modules=modules,
                campaign_rows=[target],
                repair_targets=repair_targets,
                allow_unlisted_calls=True,
            )
            with observer:
                candidate(values).backward()
        else:
            candidate(values).backward()
        torch.cuda.synchronize(device)
        result = parameter.grad.detach().float().clone()
        del values
        return result

    # Open-loop formation: this is completed before the prediction object is
    # written and before any paired trajectory is executed.
    formation_total = {name: torch.zeros_like(parameters[name], dtype=torch.float32) for name in CARRIERS}
    formation_odd = {name: torch.zeros_like(formation_total[name]) for name in CARRIERS}
    formation_even = {name: torch.zeros_like(formation_total[name]) for name in CARRIERS}
    formation_path_energy = 0.0
    formation_records = []
    for index, state in enumerate(states):
        state_energy = 0.0
        for name in CARRIERS:
            parameter = parameters[name]
            with torch.no_grad():
                base = parameter.detach().float().clone()
            candidate_grad = gradient(base, state, False, name)
            repair_grad = gradient(base, state, True, name)
            delta = candidate_grad - repair_grad
            formation_total[name].add_(delta)
            (formation_odd if index % 2 else formation_even)[name].add_(delta)
            energy = float(torch.sum(delta * delta).item())
            state_energy += energy
            del base, candidate_grad, repair_grad, delta
        formation_path_energy += state_energy
        formation_records.append({"state_id": state["state_id"], "delta_l2": state_energy ** 0.5})
        torch.cuda.empty_cache()

    resultant_energy = sum(float(torch.sum(value * value).item()) for value in formation_total.values())
    odd_energy = sum(float(torch.sum(value * value).item()) for value in formation_odd.values())
    even_energy = sum(float(torch.sum(value * value).item()) for value in formation_even.values())
    odd_even_inner = sum(
        float(torch.sum(formation_odd[name] * formation_even[name]).item()) for name in CARRIERS
    )
    formation_amplification = (resultant_energy / max(formation_path_energy, 1e-30)) ** 0.5
    odd_even_cosine = odd_even_inner / max((odd_energy * even_energy) ** 0.5, 1e-30)
    prediction = {
        "schema": "kernel-analyzer-gemma4-v3-source-prediction-v1",
        "status": "PREDICTION_FROZEN_BEFORE_CONSEQUENCE",
        "source_prediction": (
            "SOURCE_PERSISTENCE_RISK"
            if formation_amplification > 1.10 and odd_even_cosine > 0.10
            else "NO_SOURCE_PERSISTENCE_UNDER_PROTOCOL"
        ),
        "rule": {"amplification_gt": 1.10, "odd_even_cosine_gt": 0.10},
        "evidence": {
            "states": args.steps,
            "complete_coordinate_open_loop_amplification": formation_amplification,
            "odd_even_resultant_cosine": odd_even_cosine,
        },
        "target_region_id": target["region_id"],
        "target_symbol": target["symbol"],
        "runtime_release": str(release_dir),
        "claim_boundary": "Source prediction is frozen before the paired trajectory; feedback is outside this source branch.",
    }
    (output_dir / "prediction.json").write_text(json.dumps(prediction, indent=2, sort_keys=True) + "\n")
    (output_dir / "formation.json").write_text(json.dumps({
        "schema": "kernel-analyzer-gemma4-v3-formation-v1",
        "status": "COMPLETE",
        "prediction": prediction,
        "records": formation_records,
        "runtime_release": str(release_dir),
    }, indent=2, sort_keys=True) + "\n")

    # Closed-loop consequence and shared short screen.  The local/feedback/
    # actual decomposition is the same four-arm update used by the existing
    # Gemma consequence runner.
    carrier_name = CARRIERS[1]
    carrier = parameters[carrier_name]
    initial = carrier.detach().float().clone()
    cmaster = initial.clone(); rmaster = initial.clone()
    cm = torch.zeros_like(initial); cv = torch.zeros_like(initial)
    rm = torch.zeros_like(initial); rv = torch.zeros_like(initial)
    sums = {name: torch.zeros_like(initial) for name in ("local", "feedback", "actual")}
    energies = {name: 0.0 for name in sums}
    short = (
        SharedShortPersistenceScreen(
            projection_dim=args.projection_dim,
            projection_seed=20260822,
            expected_steps=args.steps,
            null_draws=args.null_draws,
            prefix_growth_mode="after_warmup",
        ) if args.steps >= 4 else None
    )
    records = []
    for index, state in enumerate(consequence_states):
        step = index + 1
        gc_c = gradient(cmaster, state, False, carrier_name)
        gr_c = gradient(cmaster, state, True, carrier_name)
        gc_r = gradient(rmaster, state, False, carrier_name)
        gr_r = gradient(rmaster, state, True, carrier_name)
        uc_c, ncm, ncv = adam_delta(gc_c, cm, cv, step, learning_rate=args.learning_rate)
        ur_c, _, _ = adam_delta(gr_c, cm, cv, step, learning_rate=args.learning_rate)
        uc_r, _, _ = adam_delta(gc_r, rm, rv, step, learning_rate=args.learning_rate)
        ur_r, nrm, nrv = adam_delta(gr_r, rm, rv, step, learning_rate=args.learning_rate)
        next_c = cmaster + uc_c; next_r = rmaster + ur_r
        uc_c = next_c - cmaster; ur_c = (cmaster + ur_c) - cmaster
        uc_r = (rmaster + uc_r) - rmaster; ur_r = next_r - rmaster
        local = 0.5 * ((uc_c - ur_c) + (uc_r - ur_r))
        feedback = 0.5 * ((uc_c - uc_r) + (ur_c - ur_r))
        actual = (next_c - next_r) - (cmaster - rmaster)
        residual = actual - local - feedback
        for name, value in (("local", local), ("feedback", feedback), ("actual", actual)):
            sums[name].add_(value)
            energies[name] += float(torch.sum(value * value).item())
            if short is not None:
                short.add_chunks(
                    f"gemma4_e2b_ple_rmsnorm::{name}",
                    (chunk for chunk in (value.detach().float().cpu().numpy(),)),
                )
        records.append({
            "step": step,
            "state_id": state["state_id"],
            "local_l2": l2(local),
            "feedback_l2": l2(feedback),
            "actual_l2": l2(actual),
            "recurrence_residual_l2": l2(residual),
            "drift_l2": l2(next_c - next_r),
        })
        cmaster, rmaster, cm, cv, rm, rv = next_c, next_r, ncm, ncv, nrm, nrv
        del gc_c, gr_c, gc_r, gr_r, local, feedback, actual, residual
        torch.cuda.empty_cache()

    stats = {}
    for name in sums:
        stats[name] = {
            "path_energy": energies[name],
            "resultant_l2": l2(sums[name]),
            "coherence_amplification": l2(sums[name]) / max(energies[name] ** 0.5, 1e-30),
        }
    consequence = {
        "schema": "kernel-analyzer-gemma4-v3-consequence-v1",
        "status": "COMPLETE",
        "prediction": prediction,
        "steps": args.steps,
        "carrier": carrier_name,
        "statistics": stats,
        "final_drift_l2": l2(cmaster - rmaster),
        "records": records,
        "runtime_release": str(release_dir),
        "formation_bank": str(args.input_bank),
        "consequence_bank": str(consequence_bank_path),
        "formation_state_role": "CONFIRMATION",
        "consequence_state_role": "TRAJECTORY" if args.consequence_bank else "CONFIRMATION_ENGINEERING_REUSE",
        "claim_boundary": "Same-process current wrapper release; source prediction and consequence are separated, and feedback remains out of source scope.",
    }
    (output_dir / "consequence.json").write_text(json.dumps(consequence, indent=2, sort_keys=True) + "\n")
    short_payload = {
        "schema": "kernel-analyzer-gemma4-v3-short-screen-dryrun-v1",
        "status": "NOT_RUN_INSUFFICIENT_STEPS" if short is None else "COMPLETE",
        "input": {
            "case_id": "gemma4_e2b_ple_rmsnorm",
            "state_role": "CONFIRMATION",
            "steps": args.steps,
            "projection_dimension": args.projection_dim,
            "runtime_release": str(release_dir),
            "prediction_revealed_before_trajectory": True,
        },
    } if short is None else short.finalize()
    if short is not None:
        short_payload["input"] = {
            "case_id": "gemma4_e2b_ple_rmsnorm",
            "state_role": "CONFIRMATION",
            "steps": args.steps,
            "projection_dimension": args.projection_dim,
            "runtime_release": str(release_dir),
            "prediction_revealed_before_trajectory": True,
        }
    (output_dir / "short_screen.json").write_text(json.dumps(short_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "GEMMA4_V3_VALIDATION_COMPLETE",
        "output_dir": str(output_dir),
        "source_prediction": prediction["source_prediction"],
        "source_amplification": formation_amplification,
        "short_actual_status": short_payload.get("paths", {}).get(
            "gemma4_e2b_ple_rmsnorm::actual", {}
        ).get("status", short_payload["status"]),
    }), flush=True)


if __name__ == "__main__":
    main()
