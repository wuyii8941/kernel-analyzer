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


GEMMA_CARRIERS = (
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
    parser.add_argument(
        "--architecture", choices=("gemma4", "generic"), default="gemma4",
        help="Model loader used for the same-process target replay. The generic "
             "loader is used for ordinary text-only causal LMs such as Llama.",
    )
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument(
        "--consequence-bank", type=Path,
        help="Optional disjoint bank for the closed trajectory; if omitted, the formation states are reused only for engineering dry runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-seed", type=int, default=24000)
    parser.add_argument("--steps", type=int, choices=(2, 8, 16), default=16,
                        help="open-loop formation states")
    parser.add_argument(
        "--consequence-steps", type=int, choices=(2, 8, 16, 32, 4096),
        help="closed-loop trajectory steps; may use a disjoint long trajectory bank",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--null-draws", type=int, default=2000)
    parser.add_argument("--target-region")
    parser.add_argument("--target-endpoint")
    parser.add_argument("--carrier")
    parser.add_argument("--case-id", default="gemma4_e2b_ple_rmsnorm")
    parser.add_argument(
        "--raw-capture-dir",
        type=Path,
        help="Optional external cache directory for exact endpoint, gradient and update values.",
    )
    parser.add_argument(
        "--raw-update-only",
        action="store_true",
        help="Capture only closed-loop effective-update pairs. Use when formation pairs already exist in an earlier exact replay.",
    )
    args = parser.parse_args()
    consequence_steps = args.consequence_steps or args.steps

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
    if len(consequence_states) < consequence_steps:
        raise RuntimeError("consequence bank lacks the requested disjoint trajectory states")
    consequence_states = consequence_states[:consequence_steps]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_capture_dir = args.raw_capture_dir
    if raw_capture_dir is not None:
        raw_capture_dir.mkdir(parents=True, exist_ok=True)
    release_dir = output_dir / "runtime_release"
    device = torch.device(args.device)
    configure_candidate_runtime(args.runtime_seed)
    model = load_model(args.architecture, args.model, device)
    parameters = dict(model.named_parameters())
    if args.architecture == "gemma4":
        missing = sorted(set(GEMMA_CARRIERS) - set(parameters))
        if missing:
            raise RuntimeError(f"declared carrier absent: {missing}")
    elif args.carrier is None:
        raise RuntimeError("--architecture generic requires an explicit --carrier")
    if args.carrier is not None and args.carrier not in parameters:
        raise RuntimeError(f"requested carrier absent: {args.carrier}")

    # A single-carrier consequence replay does not need gradients for the
    # other 1,900+ model parameters.  Leaving every parameter trainable makes
    # autograd retain large embedding and projection backward buffers even
    # though the protocol only observes the declared carrier; on Gemma this
    # can make a valid replay look like a compiler/runtime failure.  Freeze
    # the unobserved parameters only when the caller explicitly declares one
    # carrier.  The all-carrier historical protocol remains unchanged.
    requested_carrier = args.carrier
    if requested_carrier is not None:
        for name, parameter in parameters.items():
            parameter.requires_grad_(name == requested_carrier)

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
        architecture=args.architecture,
        input_bank=args.input_bank,
        state=warm_states[0],
        allow_graph_breaks=True,
    )
    del inventory_path
    import gzip
    with gzip.open(campaign_path, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    if args.target_region is None and args.architecture != "gemma4":
        raise RuntimeError("generic target replay requires --target-region")
    target = choose_target(campaign) if args.target_region is None else next(
        row for row in campaign["rows"] if row["region_id"] == args.target_region
    )
    repair_endpoints = [args.target_endpoint] if args.target_endpoint else target["output_names"]
    repair_targets = {target["region_id"]: repair_endpoints}
    carriers = (args.carrier,) if args.carrier else GEMMA_CARRIERS

    raw_gradient_records = []

    def gradient(
        master: torch.Tensor,
        state: dict,
        repair: bool,
        carrier_name: str,
        *,
        capture_raw_endpoint: bool = False,
        return_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
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
                # Endpoint output pairs are needed for the open-loop
                # tolerance rows.  Repeating those large tensors for every
                # closed-loop step wastes memory and can make a valid GPU
                # run look like an OOM, so capture them only in formation.
                raw_capture_dir=raw_capture_dir if capture_raw_endpoint else None,
            )
            with observer:
                step_loss = candidate(values)
                step_loss_value = float(step_loss.detach().float().cpu())
                step_loss.backward()
            if raw_capture_dir is not None and capture_raw_endpoint:
                raw_gradient_records.append({
                    "state_id": state["state_id"],
                    "carrier": carrier_name,
                    "observer_records": observer.records,
                })
        else:
            step_loss = candidate(values)
            step_loss_value = float(step_loss.detach().float().cpu())
            step_loss.backward()
        torch.cuda.synchronize(device)
        result = parameter.grad.detach().float().clone()
        del values
        return (result, step_loss_value) if return_loss else result

    # Open-loop formation: this is completed before the prediction object is
    # written and before any paired trajectory is executed.
    formation_total = {name: torch.zeros_like(parameters[name], dtype=torch.float32) for name in carriers}
    formation_odd = {name: torch.zeros_like(formation_total[name]) for name in carriers}
    formation_even = {name: torch.zeros_like(formation_total[name]) for name in carriers}
    formation_path_energy = 0.0
    formation_records = []
    raw_formation_records = []
    for index, state in enumerate(states):
        state_energy = 0.0
        for name in carriers:
            parameter = parameters[name]
            with torch.no_grad():
                base = parameter.detach().float().clone()
            candidate_grad = gradient(base, state, False, name)
            repair_grad = gradient(
                base, state, True, name,
                capture_raw_endpoint=(not args.raw_update_only),
            )
            delta = candidate_grad - repair_grad
            formation_total[name].add_(delta)
            (formation_odd if index % 2 else formation_even)[name].add_(delta)
            energy = float(torch.sum(delta * delta).item())
            state_energy += energy
            if raw_capture_dir is not None and not args.raw_update_only:
                state_dir = raw_capture_dir / "formation" / f"{state['state_id']}_{name}"
                state_dir.mkdir(parents=True, exist_ok=True)
                vector_path = state_dir / "vectors.pt"
                torch.save({
                    "state_id": state["state_id"],
                    "carrier": name,
                    "candidate_gradient": candidate_grad.detach().cpu(),
                    "repair_gradient": repair_grad.detach().cpu(),
                    "gradient_difference": delta.detach().cpu(),
                }, vector_path)
                raw_formation_records.append({
                    "state_id": state["state_id"],
                    "carrier": name,
                    "path": str(vector_path),
                })
            del base, candidate_grad, repair_grad, delta
        formation_path_energy += state_energy
        formation_records.append({"state_id": state["state_id"], "delta_l2": state_energy ** 0.5})
        torch.cuda.empty_cache()

    resultant_energy = sum(float(torch.sum(value * value).item()) for value in formation_total.values())
    odd_energy = sum(float(torch.sum(value * value).item()) for value in formation_odd.values())
    even_energy = sum(float(torch.sum(value * value).item()) for value in formation_even.values())
    odd_even_inner = sum(
        float(torch.sum(formation_odd[name] * formation_even[name]).item()) for name in carriers
    )
    formation_amplification = (resultant_energy / max(formation_path_energy, 1e-30)) ** 0.5
    odd_even_cosine = odd_even_inner / max((odd_energy * even_energy) ** 0.5, 1e-30)
    prediction = {
        "schema": "kernel-analyzer-target-v3-source-prediction-v1",
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
        "architecture": args.architecture,
        "runtime_release": str(release_dir),
        "claim_boundary": "Source prediction is frozen before the paired trajectory; feedback is outside this source branch.",
    }
    (output_dir / "prediction.json").write_text(json.dumps(prediction, indent=2, sort_keys=True) + "\n")
    (output_dir / "formation.json").write_text(json.dumps({
        "schema": "kernel-analyzer-target-v3-formation-v1",
        "status": "COMPLETE",
        "prediction": prediction,
        "records": formation_records,
        "runtime_release": str(release_dir),
    }, indent=2, sort_keys=True) + "\n")

    # Closed-loop consequence and shared short screen.  The local/feedback/
    # actual decomposition is the same four-arm update used by the existing
    # Gemma consequence runner.
    carrier_name = carriers[0]
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
            expected_steps=consequence_steps,
            null_draws=args.null_draws,
            prefix_growth_mode="after_warmup",
        ) if consequence_steps >= 4 else None
    )
    records = []
    raw_trajectory_records = []
    for index, state in enumerate(consequence_states):
        step = index + 1
        gc_c, candidate_loss = gradient(cmaster, state, False, carrier_name, return_loss=True)
        gr_c = gradient(cmaster, state, True, carrier_name)
        gc_r = gradient(rmaster, state, False, carrier_name)
        gr_r, repair_loss = gradient(rmaster, state, True, carrier_name, return_loss=True)
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
                    f"{args.case_id}::{name}",
                    (chunk for chunk in (value.detach().float().cpu().numpy(),)),
                )
        if raw_capture_dir is not None:
            step_dir = raw_capture_dir / "trajectory" / f"step_{step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            vector_path = step_dir / "vectors.pt"
            torch.save({
                "step": step,
                "state_id": state["state_id"],
                # Keep the two same-state effective-update pairs as well as
                # the local/feedback decomposition.  The former are needed
                # for exact update ULP and rtol/atol comparisons; a local
                # difference alone cannot support those claims.
                "candidate_update_at_candidate_state": uc_c.detach().cpu(),
                "repair_update_at_candidate_state": ur_c.detach().cpu(),
                "local_update": local.detach().cpu(),
            }, vector_path)
            raw_trajectory_records.append({
                "step": step,
                "state_id": state["state_id"],
                "path": str(vector_path),
            })
        records.append({
            "step": step,
            "state_id": state["state_id"],
            "local_l2": l2(local),
            "feedback_l2": l2(feedback),
            "actual_l2": l2(actual),
            "recurrence_residual_l2": l2(residual),
            "drift_l2": l2(next_c - next_r),
            "candidate_loss": candidate_loss,
            "repair_loss": repair_loss,
            "paired_loss_gap": candidate_loss - repair_loss,
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
        "schema": "kernel-analyzer-target-v3-consequence-v1",
        "status": "COMPLETE_LONG_HORIZON" if consequence_steps >= 4096 else "COMPLETE",
        "prediction": prediction,
        "steps": consequence_steps,
        "carrier": carrier_name,
        "architecture": args.architecture,
        "statistics": stats,
        "final_drift_l2": l2(cmaster - rmaster),
        "records": records,
        "runtime_release": str(release_dir),
        "formation_bank": str(args.input_bank),
        "consequence_bank": str(consequence_bank_path),
        "formation_state_role": "CONFIRMATION",
        "consequence_state_role": "TRAJECTORY" if args.consequence_bank else "CONFIRMATION_ENGINEERING_REUSE",
        "loss_audit": {
            "recorded": True,
            "any_period_split": any(abs(float(row.get("paired_loss_gap", 0.0))) > 1e-8 for row in records),
            "max_abs_gap": max((abs(float(row.get("paired_loss_gap", 0.0))) for row in records), default=0.0),
            "final_gap": records[-1].get("paired_loss_gap") if records else None,
            "last_512_mean": (
                sum(float(row.get("paired_loss_gap", 0.0)) for row in records[-min(512, len(records)):])
                / max(1, min(512, len(records)))
            ) if records else None,
            "last_512_max_abs": max((abs(float(row.get("paired_loss_gap", 0.0))) for row in records[-min(512, len(records)):]), default=0.0),
            "tolerance": 1e-8,
        },
        "claim_boundary": "Same-process current wrapper release; source prediction and consequence are separated, and feedback remains out of source scope.",
        "gradient_scope": {
            "declared_carrier_only": requested_carrier is not None,
            "carrier": requested_carrier,
            "other_parameters_frozen": requested_carrier is not None,
        },
    }
    if raw_capture_dir is not None:
        raw_manifest = {
            "schema": "kernel-analyzer-gemma4-v3-raw-capture-v1",
            "status": "COMPLETE_RAW_ENDPOINT_GRADIENT_UPDATE_CAPTURE",
            "case_id": args.case_id,
            "target_region": target["region_id"],
            "target_endpoint": repair_endpoints,
            "carrier": carrier_name,
            "formation_gradient_records": raw_formation_records,
            "trajectory_update_records": raw_trajectory_records,
            "observer_gradient_records": raw_gradient_records,
            "claim_boundary": "Formation stores endpoint and gradient pairs; consequence stores same-state effective-update pairs and the local magnitude. Closed-loop gradient/feedback vectors are summarized in consequence.json and are not duplicated in raw storage.",
        }
        raw_manifest_path = raw_capture_dir / "raw_manifest.json"
        raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n")
        consequence["raw_capture_manifest"] = str(raw_manifest_path)
    (output_dir / "consequence.json").write_text(json.dumps(consequence, indent=2, sort_keys=True) + "\n")
    short_payload = {
        "schema": "kernel-analyzer-target-v3-short-screen-dryrun-v1",
        "status": "NOT_RUN_INSUFFICIENT_STEPS" if short is None else "COMPLETE",
        "input": {
            "case_id": args.case_id,
            "state_role": "CONFIRMATION",
            "steps": consequence_steps,
            "projection_dimension": args.projection_dim,
            "runtime_release": str(release_dir),
            "prediction_revealed_before_trajectory": True,
        },
    } if short is None else short.finalize()
    if short is not None:
        short_payload["input"] = {
            "case_id": args.case_id,
            "state_role": "CONFIRMATION",
            "steps": consequence_steps,
            "projection_dimension": args.projection_dim,
            "runtime_release": str(release_dir),
            "prediction_revealed_before_trajectory": True,
        }
    (output_dir / "short_screen.json").write_text(json.dumps(short_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "TARGET_V3_VALIDATION_COMPLETE",
        "output_dir": str(output_dir),
        "case_id": args.case_id,
        "architecture": args.architecture,
        "source_prediction": prediction["source_prediction"],
        "source_amplification": formation_amplification,
        "short_actual_status": short_payload.get("paths", {}).get(
            f"{args.case_id}::actual", {}
        ).get("status", short_payload["status"]),
    }), flush=True)


if __name__ == "__main__":
    main()
