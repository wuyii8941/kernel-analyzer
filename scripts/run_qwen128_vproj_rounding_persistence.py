#!/usr/bin/env python3
"""Live ROUNDING_ONLY persistence test for Qwen seq128 v_proj.

The repair arm is the conditional expectation of coordinate-wise unbiased
stochastic BF16 materialization, estimated with repeated draws at each common
state.  The virtual repair trajectory advances by the mean AdamW update and
mean moment recurrences.  A split-repeat path measures Monte Carlo noise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_analyzer.trajectory_persistence import OrderedVectorPath, cosine  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import file_digest, load_model  # noqa: E402
from scripts.run_mm_source_aligned_repair import (  # noqa: E402
    SourceAlignedMMRepair,
    validate_target_call,
)


CID = "qwen_seq128_forward_8_output"
TARGET_SHA = "1847d6184bdf781a1b57531571a298c898337332aa694e47a96de633d30ed2af"
CARRIER = "model.layers.0.self_attn.v_proj.weight"


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def adam_arm(
    gradient: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    step: int,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_first = first * beta1 + gradient * (1.0 - beta1)
    next_second = second * beta2 + gradient.square() * (1.0 - beta2)
    update = -learning_rate * (next_first / (1.0 - beta1**step)) / (
        (next_second / (1.0 - beta2**step)).sqrt() + epsilon
    )
    return update, next_first, next_second


def materialized_update(master: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Return the increment actually representable in the FP32 master."""
    return (master + update) - master


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def average(values: list[torch.Tensor]) -> torch.Tensor:
    result = torch.zeros_like(values[0])
    for value in values:
        result.add_(value, alpha=1.0 / len(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, default=ROOT / "results/coverage/qwen_seq128_input_bank.json")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "results/coverage/runtime_releases/qwen_seq128_r1")
    parser.add_argument("--decomposition", type=Path, default=ROOT / "results/coverage/cases/qwen128_vproj_precision_decomposition.json")
    parser.add_argument("--candidate-id", default=CID)
    parser.add_argument("--target-sha", default=TARGET_SHA)
    parser.add_argument("--carrier", default=CARRIER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--calibration-steps", type=int, default=8)
    parser.add_argument("--rounding-repeats", type=int, default=8)
    parser.add_argument("--rounding-seed-base", type=int, default=7_300_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if not 2 <= args.steps <= 32:
        raise ValueError("steps must be in [2, 32]")
    if not 1 <= args.calibration_steps < args.steps:
        raise ValueError("calibration steps must precede evaluation")
    if args.rounding_repeats < 4 or args.rounding_repeats % 2:
        raise ValueError("rounding repeats must be an even integer >= 4")

    queue = json.loads((ROOT / "results/coverage/bias_candidate_queue.json").read_text())
    bound = next(row for row in queue["candidates"] if row["candidate_id"] == args.candidate_id)
    exact = bound["exact_generated_call"]
    if exact["function"] != "extern_kernels.mm" or exact["source_line_sha256"] != args.target_sha:
        raise RuntimeError("candidate does not bind the declared MM")
    decomposition = json.loads(args.decomposition.read_text())
    if decomposition["candidate_id"] != args.candidate_id:
        raise RuntimeError("precision decomposition belongs to another candidate")
    if "output_rounding" not in decomposition["coherent_sources"]:
        raise RuntimeError("ROUNDING_ONLY is not source-aligned for this candidate")

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))[:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("input bank is shorter than trajectory")
    capture = json.loads((args.release_dir / "capture.json").read_text())
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank differs from frozen runtime release")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("qwen", args.model, device)
    target = dict(model.named_parameters())[args.carrier]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not bool(capture.get("allow_graph_breaks", False)), dynamic=False,
    )
    first_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([first_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    runtime_target = validate_target_call(
        [module for module, _phase in wrapper_modules(modules)], args.target_sha
    )

    beta1, beta2, epsilon = 0.9, 0.95, 1e-8
    initial = target.detach().float().clone()
    candidate_master = initial.clone()
    repair_master = initial.clone()
    candidate_first = torch.zeros_like(initial)
    candidate_second = torch.zeros_like(initial)
    repair_first = torch.zeros_like(initial)
    repair_second = torch.zeros_like(initial)

    def gradient(
        master: torch.Tensor,
        tokens: list[int],
        state_index: int,
        mode: str | None,
        *,
        rounding_seed: int | None = None,
        rounding_stratum_index: int | None = None,
        rounding_stratum_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        with torch.no_grad():
            target.copy_(master.to(target.dtype))
        model_seed = 24_000 + state_index
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        observer = None if mode is None else SourceAlignedMMRepair(
            modules, args.target_sha, mode, rounding_seed=rounding_seed,
            rounding_stratum_index=rounding_stratum_index,
            rounding_stratum_count=rounding_stratum_count,
        )
        if observer is None:
            loss = candidate(values)
            loss.backward()
        else:
            with observer:
                loss = candidate(values)
                loss.backward()
        torch.cuda.synchronize(device)
        if target.grad is None:
            raise RuntimeError("declared carrier gradient is absent")
        result = target.grad.detach().float().clone()
        target.grad = None
        return loss.detach(), result, None if observer is None else observer.summary

    def repair_ensemble(
        master: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        tokens: list[int],
        state_index: int,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        updates: list[torch.Tensor] = []
        next_firsts: list[torch.Tensor] = []
        next_seconds: list[torch.Tensor] = []
        local_l2 = []
        half = args.rounding_repeats // 2
        for repeat in range(args.rounding_repeats):
            # Two independent stratified half-ensembles provide both a low-
            # variance conditional mean and an honest split-noise estimate.
            group = repeat // half
            stratum = repeat % half
            seed = args.rounding_seed_base + state_index * 1000 + group
            _, repair_gradient, summary = gradient(
                master, tokens, state_index, "ROUNDING_ONLY", rounding_seed=seed,
                rounding_stratum_index=stratum, rounding_stratum_count=half,
            )
            update, next_first, next_second = adam_arm(
                repair_gradient, first, second, step,
                learning_rate=args.learning_rate, beta1=beta1, beta2=beta2,
                epsilon=epsilon,
            )
            updates.append(update)
            next_firsts.append(next_first)
            next_seconds.append(next_second)
            assert summary is not None
            local_l2.append(float(summary["intervention"]["l2"]))
            del repair_gradient
        split_difference = average(updates[:half]) - average(updates[half:])
        return (
            average(updates), average(next_firsts), average(next_seconds),
            split_difference,
            {
                "repeat_count": args.rounding_repeats,
                "mean_local_intervention_l2": sum(local_l2) / len(local_l2),
                "update_split_difference_l2": norm(split_difference),
            },
        )

    # One exact sham checks that installing the observer without changing the
    # target endpoint leaves the complete declared F+B carrier unchanged.
    natural_loss, natural_grad, _ = gradient(initial, first_tokens, 0, None)
    sham_loss, sham_grad, _ = gradient(initial, first_tokens, 0, "SHAM")
    sham_exact = bool(torch.equal(natural_loss, sham_loss) and torch.equal(natural_grad, sham_grad))
    del natural_grad, sham_grad
    if not sham_exact:
        raise RuntimeError("matched source observer sham changed the F+B carrier")

    local_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=args.calibration_steps)
    feedback_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=args.calibration_steps)
    actual_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=args.calibration_steps)
    mc_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=args.calibration_steps)
    max_recurrence_relative = 0.0
    records = []

    for offset, state in enumerate(states):
        step = offset + 1
        tokens = state.get("token_ids", state.get("input_ids"))
        state_id = str(state.get("sequence_id", state.get("state_id", offset)))
        drift_before = candidate_master - repair_master

        candidate_loss_c, candidate_grad_c, _ = gradient(
            candidate_master, tokens, offset, None
        )
        ucc_raw, next_candidate_first, next_candidate_second = adam_arm(
            candidate_grad_c, candidate_first, candidate_second, step,
            learning_rate=args.learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon,
        )
        urc_raw, _, _, split_c_raw, repair_summary_c = repair_ensemble(
            candidate_master, candidate_first, candidate_second, tokens,
            offset, step,
        )
        candidate_loss_r, candidate_grad_r, _ = gradient(
            repair_master, tokens, offset, None
        )
        ucr_raw, _, _ = adam_arm(
            candidate_grad_r, repair_first, repair_second, step,
            learning_rate=args.learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon,
        )
        urr_raw, next_repair_first, next_repair_second, split_r_raw, repair_summary_r = repair_ensemble(
            repair_master, repair_first, repair_second, tokens,
            offset, step,
        )

        ucc = materialized_update(candidate_master, ucc_raw)
        urc = materialized_update(candidate_master, urc_raw)
        ucr = materialized_update(repair_master, ucr_raw)
        urr = materialized_update(repair_master, urr_raw)
        split_c = split_c_raw
        split_r = split_r_raw
        local = ((ucc - urc) + (ucr - urr)) * 0.5
        feedback = ((ucc - ucr) + (urc - urr)) * 0.5
        mc_split = (split_c + split_r) * 0.5
        candidate_master.add_(ucc_raw)
        repair_master.add_(urr_raw)
        candidate_first, candidate_second = next_candidate_first, next_candidate_second
        repair_first, repair_second = next_repair_first, next_repair_second
        drift_after = candidate_master - repair_master
        actual_increment = ucc - urr
        master_difference_residual = (drift_after - drift_before) - actual_increment
        recurrence = actual_increment - local - feedback
        recurrence_relative = norm(recurrence) / max(
            norm(actual_increment), norm(local), norm(feedback), 1e-30
        )
        max_recurrence_relative = max(max_recurrence_relative, recurrence_relative)

        local_row = local_path.add(local)
        feedback_row = feedback_path.add(feedback)
        actual_row = actual_path.add(actual_increment)
        mc_row = mc_path.add(mc_split)
        row = {
            "step": step,
            "state_id": state_id,
            "candidate_loss_at_candidate_state": float(candidate_loss_c.item()),
            "candidate_loss_at_repair_state": float(candidate_loss_r.item()),
            "local_effect_l2": local_row["l2"],
            "feedback_effect_l2": feedback_row["l2"],
            "actual_drift_increment_l2": actual_row["l2"],
            "monte_carlo_split_difference_l2": mc_row["l2"],
            "local_frozen_carrier_projection": local_row["frozen_carrier_projection"],
            "recurrence_residual_l2": norm(recurrence),
            "recurrence_relative": recurrence_relative,
            "master_difference_residual_l2": norm(master_difference_residual),
            "drift_l2": norm(drift_after),
            "candidate_state_repair": repair_summary_c,
            "repair_state_repair": repair_summary_r,
        }
        records.append(row)
        write(args.output, {
            "schema": "kernel-analyzer-qwen-vproj-rounding-persistence-v1",
            "status": "RUNNING", "steps_complete": step, "records": records,
        })
        print(json.dumps({"event": "QWEN_ROUNDING_STEP", **{
            key: value for key, value in row.items()
            if key not in {"candidate_state_repair", "repair_state_repair"}
        }}), flush=True)
        del candidate_grad_c, candidate_grad_r, ucc, urc, ucr, urr
        del ucc_raw, urc_raw, ucr_raw, urr_raw, split_c_raw, split_r_raw
        del local, feedback, mc_split, actual_increment, recurrence, split_c, split_r
        del master_difference_residual
        torch.cuda.empty_cache()

    local_summary = local_path.finalize()
    feedback_summary = feedback_path.finalize()
    actual_summary = actual_path.finalize()
    mc_summary = mc_path.finalize()
    final_drift = candidate_master - repair_master
    assert local_path.total is not None and feedback_path.total is not None
    local_final_cosine = cosine(local_path.total, final_drift)
    feedback_final_cosine = cosine(feedback_path.total, final_drift)
    local_to_mc = local_summary["resultant_l2"] / max(mc_summary["resultant_l2"], 1e-30)
    recurrence_closed = max_recurrence_relative <= 1e-5
    mc_resolved = local_to_mc >= 3.0
    persistent_local = bool(
        recurrence_closed and mc_resolved
        and local_summary["coherence_amplification"] >= 2.0
        and actual_summary["coherence_amplification"] >= 2.0
        and local_final_cosine is not None and local_final_cosine >= 0.5
    )
    feedback_sustained = bool(
        not persistent_local and recurrence_closed and mc_resolved
        and feedback_summary["coherence_amplification"] >= 2.0
        and actual_summary["coherence_amplification"] >= 2.0
        and feedback_final_cosine is not None and feedback_final_cosine >= 0.5
    )
    verdict = (
        "UNRESOLVED_MONTE_CARLO" if not mc_resolved else
        "PERSISTENT_LOCAL_BIAS" if persistent_local else
        "FEEDBACK_SUSTAINED_SEPARATION" if feedback_sustained else
        "DIFFUSIVE_OR_CANCELING_SEPARATION"
    )
    payload = {
        "schema": "kernel-analyzer-qwen-vproj-rounding-persistence-v1",
        "status": "COMPLETE",
        "verdict": verdict,
        "case": {
            "candidate_id": args.candidate_id,
            "target_parameter": args.carrier,
            "formation_contrast": "ROUNDING_ONLY_UNBIASED_BF16",
            "repair_trajectory": "virtual conditional-mean stochastic-rounding AdamW arm",
        },
        "protocol": {
            "steps": args.steps,
            "calibration_steps": args.calibration_steps,
            "rounding_repeats_per_state_arm": args.rounding_repeats,
            "same_weight_and_moments_within_each_counterfactual_state": True,
            "symmetric_four_counterfactual_recurrence": True,
            "optimizer": {
                "name": "AdamW", "learning_rate": args.learning_rate,
                "betas": [beta1, beta2], "epsilon": epsilon, "weight_decay": 0.0,
            },
            "predeclared_classification": {
                "coherence_amplification_min": 2.0,
                "final_alignment_cosine_min": 0.5,
                "local_resultant_over_mc_split_min": 3.0,
                "recurrence_relative_max": 1e-5,
            },
        },
        "controls": {"matched_sham_exact": sham_exact},
        "summaries": {
            "local": local_summary,
            "feedback": feedback_summary,
            "actual_drift_increment": actual_summary,
            "monte_carlo_split": mc_summary,
            "local_resultant_over_mc_split": local_to_mc,
            "final_drift_l2": norm(final_drift),
            "local_final_drift_cosine": local_final_cosine,
            "feedback_final_drift_cosine": feedback_final_cosine,
            "max_recurrence_relative": max_recurrence_relative,
            "global_master_difference_closure_l2": norm(
                actual_path.total - final_drift.reshape(-1)
            ),
        },
        "records": records,
        "bindings": {
            "runtime_target": runtime_target,
            "input_bank": str(args.input_bank),
            "decomposition": str(args.decomposition),
        },
        "claim_boundary": (
            "The repair trajectory is a finite-repeat estimate of the conditional-mean "
            "source-debiased optimizer arm, not one stochastic rounding realization. "
            "The split-repeat path is retained as the Monte Carlo resolution control."
        ),
    }
    write(args.output, payload)
    print(json.dumps({
        "event": "QWEN_ROUNDING_COMPLETE", "verdict": verdict,
        "local_amplification": local_summary["coherence_amplification"],
        "feedback_amplification": feedback_summary["coherence_amplification"],
        "actual_amplification": actual_summary["coherence_amplification"],
        "local_to_mc": local_to_mc,
    }), flush=True)


if __name__ == "__main__":
    main()
