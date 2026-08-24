#!/usr/bin/env python3
"""Run a resumable four-arm consequence trajectory for one bound endpoint.

This runner reuses a frozen runtime release and exact endpoint repair.  It
advances only the declared carrier parameter and computes the symmetric
candidate/repair recurrence from four real counterfactual optimizer updates.
Formation labels are neither read nor emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
                str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.bias_consequence_v21 import BiasConsequenceTrace  # noqa: E402
from kernel_analyzer.persistence_property import aligned_level_statistics_from_gram  # noqa: E402
from scripts.aot_capture import AOTForwardBackwardCapture  # noqa: E402
from scripts.capture_bound_endpoint_bias_formation_v21 import (  # noqa: E402
    align_reference_to_candidate,
    partial_reduction_reference,
    reference_cut_gates,
    validate_runtime_structure,
)
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_same_dtype_semantic_oracle import load  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver  # noqa: E402


def coherence_curve(vectors: list[torch.Tensor]) -> list[dict[str, float | int]]:
    """Summarize an ordered vector path without fitting a direction."""
    total = torch.zeros_like(vectors[0], dtype=torch.float64)
    energy = 0.0
    rows: list[dict[str, float | int]] = []
    for index, vector in enumerate(vectors, 1):
        value = vector.detach().double().reshape(-1)
        total.add_(value)
        energy += float(torch.dot(value, value))
        if index in (2, 4, 8, 16, 32):
            scale = math.sqrt(max(energy, 0.0))
            rows.append({
                "horizon": index,
                "resultant_l2": float(torch.linalg.vector_norm(total)),
                "diffusive_scale_l2": scale,
                "coherence_amplification": float(torch.linalg.vector_norm(total)) / max(scale, 1e-30),
            })
    return rows


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def deterministic_projection(case_id: str, coordinates: int, device: torch.device) -> torch.Tensor:
    seed = int(hashlib.sha256(case_id.encode()).hexdigest()[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (coordinates,), generator=generator, dtype=torch.int8)
    return signs.mul_(2).sub_(1).float().div_(math.sqrt(coordinates)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("qwen", "phi", "mamba", "deepseek8b"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    # Permit the same runner to perform the predeclared long horizon.  The
    # frozen bank and state-role filter still enforce the actual availability.
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--state-role")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument("--recurrence-tolerance", type=float, default=1e-6)
    parser.add_argument("--recurrence-relative-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--three-stage-output", type=Path,
        help="Optional same-run operator-output, gradient, and AdamW-update summary.",
    )
    parser.add_argument(
        "--raw-stage-output", type=Path,
        help=(
            "Optional raw replay capture for optimizer-state ablations. This "
            "stores per-step endpoint/gradient/update vectors and pre-step "
            "AdamW moments; it is intentionally opt-in."
        ),
    )
    args = parser.parse_args()
    if args.resume and args.three_stage_output is not None:
        raise ValueError(
            "--resume cannot be combined with --three-stage-output because the "
            "checkpoint does not store the full stage vectors"
        )
    if args.resume and args.raw_stage_output is not None:
        raise ValueError(
            "--resume cannot be combined with --raw-stage-output because the "
            "checkpoint does not store the complete raw optimizer capture"
        )

    architecture = "deepseek8" if args.architecture == "deepseek8b" else args.architecture
    cases = json.loads(args.case_plan.read_text(encoding="utf-8"))["cases"]
    selected = [case for case in cases if str(case.get("case_id")) == args.case_id]
    if len(selected) != 1:
        raise ValueError("case-id must select exactly one frozen case")
    case = selected[0]
    task_id = str(case["task_id"])
    carrier_name = str(case["carrier"])

    capture = json.loads((args.release_dir / "capture.json").read_text(encoding="utf-8"))
    campaign = load(args.release_dir / "campaign.json.gz")
    inventory = load(args.release_dir / "inventory.json.gz")
    plan = load(args.release_dir / "same_dtype_tasks.json.gz")
    by_task = {str(row["task_id"]): row for row in plan["rows"]}
    if task_id not in by_task:
        raise ValueError("case task is absent from the frozen release")
    task = by_task[task_id]

    references_by_cut_task: dict[str, str] = {}
    cuts: list[dict[str, Any]] = []
    frozen_cuts = {
        str(row["task_id"]).removeprefix("same-dtype:"): row
        for row in plan["reference_cut_tasks"]
    }
    if case.get("reference_method") not in {
        "PARTIAL_REDUCTION_FROM_BOUND_INPUT", "EXTERNAL_FP32_RECOMPUTE"
    }:
        custom = case.get("reference_cut_task")
        if custom is not None:
            cut = dict(custom)
        else:
            endpoint = str(task.get("exact_aot_endpoint_id"))
            if endpoint not in frozen_cuts:
                raise RuntimeError("reference cut is absent for the selected endpoint")
            cut = frozen_cuts[endpoint]
        cuts.append(cut)
        references_by_cut_task[str(cut["task_id"])] = task_id

    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank.get("states", bank.get("records", []))
    if args.state_role:
        states = [row for row in states if str(row.get("role")) == args.state_role]
    states = states[:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("frozen trajectory bank does not contain the requested steps")
    state_ids = [str(row.get("state_id", row.get("sequence_id", index)))
                 for index, row in enumerate(states)]
    if len(set(state_ids)) != len(state_ids):
        raise RuntimeError("trajectory state IDs are not unique")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    configure_candidate_runtime(32_000)
    model = load_model(architecture, args.model, device)
    parameters = dict(model.named_parameters())
    if carrier_name not in parameters:
        raise RuntimeError("declared carrier parameter is absent")
    carrier = parameters[carrier_name]

    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not args.allow_graph_breaks, dynamic=False,
    )
    warm_tokens = states[0].get("input_ids", states[0].get("token_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_runtime_structure(wrapper_modules(modules), capture)

    active: dict[str, Any] = {"sink": None}

    def dispatch(cut: Any, outputs: tuple[Any, ...]) -> None:
        if active["sink"] is not None:
            active["sink"](cut, outputs)

    reference_capture = None
    reference = None
    if cuts:
        reference_capture = AOTForwardBackwardCapture(
            reference_cut_tasks=cuts, reference_value_sink=dispatch,
        )
        with inductor_config.patch({"force_disable_caches": True}):
            reference = torch.compile(
                LossStep(model), backend=reference_capture.inductor_partition_backend(),
                fullgraph=not args.allow_graph_breaks, dynamic=False,
            )
            model.zero_grad(set_to_none=True)
            warm_loss = reference(warm)
            reference_capture.bind_user_outputs(warm_loss)
            warm_loss.register_hook(reference_capture.bind_user_cotangent)
            warm_loss.backward()
            torch.cuda.synchronize(device)
        if not all(reference_cut_gates(reference_capture).values()):
            raise RuntimeError("reference cut failed during consequence warm-up")

    expected_length = int(capture["input"]["sequence_length"])

    def gradient(
        master: torch.Tensor,
        state: dict[str, Any],
        repair_arm: bool,
        seed: int,
    ) -> tuple[
        torch.Tensor,
        int,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
        float,
    ]:
        with torch.no_grad():
            carrier.copy_(master.to(carrier.dtype))
        tokens = state.get("input_ids", state.get("token_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if not repair_arm:
            candidate_endpoint: dict[str, torch.Tensor] = {}

            def candidate_sink(observed_id: str, tensor: torch.Tensor, metadata: Any) -> None:
                if observed_id != task_id or candidate_endpoint:
                    raise RuntimeError("candidate endpoint cardinality changed")
                candidate_endpoint[observed_id] = tensor.detach().clone()

            if args.raw_stage_output is not None:
                candidate_observer = SameDtypeSemanticCandidateObserver(
                    modules=modules,
                    campaign_rows=campaign["rows"],
                    inventory_rows=inventory["runtime_call_audit"]["rows"],
                    task_rows=[task],
                    sink=candidate_sink,
                    include_unresolved_tasks=True,
                )
                with candidate_observer:
                    model.zero_grad(set_to_none=True)
                    candidate(values).backward()
                candidate_observer.validate()
            else:
                model.zero_grad(set_to_none=True)
                candidate(values).backward()
            torch.cuda.synchronize(device)
            if carrier.grad is None:
                raise RuntimeError("candidate carrier gradient is absent")
            loss_value = float(candidate(values).detach().float().cpu())
            # The preceding call is intentionally a fresh loss-only forward;
            # it does not change the captured gradient or optimizer state.
            return (
                carrier.grad.detach().float().clone(),
                0,
                None,
                candidate_endpoint.get(task_id),
                loss_value,
            )

        references: dict[str, torch.Tensor] = {}
        if reference is not None and reference_capture is not None:
            def reference_sink(cut: Any, outputs: tuple[Any, ...]) -> None:
                endpoint = references_by_cut_task[str(cut.task_id)]
                tensors = [value for value in outputs if isinstance(value, torch.Tensor)]
                if len(tensors) != 1 or endpoint in references:
                    raise RuntimeError("reference endpoint cardinality changed")
                references[endpoint] = tensors[0].detach().clone()

            active["sink"] = reference_sink
            model.zero_grad(set_to_none=True)
            reference_loss = reference(values)
            reference_capture.bind_user_outputs(reference_loss)
            reference_loss.register_hook(reference_capture.bind_user_cotangent)
            reference_loss.backward()
            torch.cuda.synchronize(device)
            active["sink"] = None

        delivered: dict[str, Any] = {}

        def repair_sink(observed_id: str, tensor: torch.Tensor, metadata: Any) -> None:
            if observed_id != task_id or delivered:
                raise RuntimeError("repair endpoint identity drifted")
            if case.get("reference_method") == "PARTIAL_REDUCTION_FROM_BOUND_INPUT":
                references[observed_id] = partial_reduction_reference(
                    metadata, tensor, sequence_length=expected_length,
                    feature_count=int(carrier.numel()),
                ).detach().clone()
            elif case.get("reference_method") == "EXTERNAL_FP32_RECOMPUTE":
                external = fp32_external_reference(
                    str(metadata.get("external_symbol")),
                    metadata.get("runtime_args", ()),
                    metadata.get("runtime_kwargs", {}),
                ).to(tensor.dtype)
                references[observed_id] = align_reference_to_candidate(
                    external, tensor
                ).detach().clone()
            target = align_reference_to_candidate(references[observed_id], tensor)
            before = tensor.detach().clone()
            tensor.copy_(target)
            delivered["changed"] = int(torch.count_nonzero(before != target))
            delivered["endpoint_pair"] = (
                before.detach().cpu().clone(),
                target.detach().cpu().clone(),
            )
            if args.three_stage_output is not None:
                delivered["endpoint_delta"] = (
                    before.detach().float() - target.detach().float()
                ).cpu().reshape(-1).clone()

        model.zero_grad(set_to_none=True)
        observer = SameDtypeSemanticCandidateObserver(
            modules=modules,
            campaign_rows=campaign["rows"],
            inventory_rows=inventory["runtime_call_audit"]["rows"],
            task_rows=[task], sink=repair_sink, include_unresolved_tasks=True,
        )
        with observer:
            candidate(values).backward()
        torch.cuda.synchronize(device)
        observer.validate()
        if carrier.grad is None or "changed" not in delivered:
            raise RuntimeError("repair did not reach the endpoint/carrier")
        repair_loss_value = float(reference_loss.detach().float().cpu()) if reference is not None else float(candidate(values).detach().float().cpu())
        return (
            carrier.grad.detach().float().clone(), delivered["changed"],
            delivered.get("endpoint_delta"), delivered.get("endpoint_pair"),
            repair_loss_value,
        )

    initial = carrier.detach().float().clone()
    candidate_master = initial.clone()
    repair_master = initial.clone()
    candidate_m = torch.zeros_like(initial)
    candidate_v = torch.zeros_like(initial)
    repair_m = torch.zeros_like(initial)
    repair_v = torch.zeros_like(initial)
    start_step = 0
    saved_rows: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("case_id") != args.case_id or checkpoint.get("state_ids") != state_ids:
            raise RuntimeError("checkpoint does not match the frozen case/trajectory")
        candidate_master = checkpoint["candidate_master"].to(device)
        repair_master = checkpoint["repair_master"].to(device)
        candidate_m = checkpoint["candidate_m"].to(device)
        candidate_v = checkpoint["candidate_v"].to(device)
        repair_m = checkpoint["repair_m"].to(device)
        repair_v = checkpoint["repair_v"].to(device)
        start_step = int(checkpoint["next_step"])
        saved_rows = list(checkpoint["rows"])

    projection = deterministic_projection(args.case_id, initial.numel(), device).reshape(initial.shape)
    trace = BiasConsequenceTrace(args.case_id, state_ids, args.recurrence_tolerance)
    local_vectors: list[torch.Tensor] = []
    feedback_vectors: list[torch.Tensor] = []
    actual_vectors: list[torch.Tensor] = []
    endpoint_vectors: list[torch.Tensor] = []
    candidate_state_gradient_vectors: list[torch.Tensor] = []
    candidate_state_update_vectors: list[torch.Tensor] = []
    raw_candidate_gradient_vectors: list[torch.Tensor] = []
    raw_repair_gradient_vectors: list[torch.Tensor] = []
    raw_candidate_update_vectors: list[torch.Tensor] = []
    raw_repair_update_vectors: list[torch.Tensor] = []
    raw_endpoint_vectors: list[torch.Tensor] = []
    raw_endpoint_candidate_vectors: list[torch.Tensor] = []
    raw_endpoint_reference_vectors: list[torch.Tensor] = []
    raw_endpoint_candidate_dtype: str | None = None
    raw_endpoint_reference_dtype: str | None = None
    raw_candidate_moments: list[torch.Tensor] = []
    raw_candidate_second_moments: list[torch.Tensor] = []
    raw_repair_moments: list[torch.Tensor] = []
    raw_repair_second_moments: list[torch.Tensor] = []

    def arm(value: torch.Tensor) -> dict[str, Any]:
        flat = value.detach().double().reshape(-1)
        signed = float(torch.dot(flat, projection.double().reshape(-1)))
        return {"vector": flat.cpu().tolist(), "signed_value": signed}

    for row in saved_rows:
        trace.add(
            row["step_id"],
            candidate_at_candidate_state=row["candidate_at_candidate_state"],
            repair_at_candidate_state=row["repair_at_candidate_state"],
            candidate_at_repair_state=row["candidate_at_repair_state"],
            repair_at_repair_state=row["repair_at_repair_state"],
            drift_before=row["drift_before"], drift_after=row["drift_after"],
            metadata=row.get("metadata", {}),
        )
        local_vectors.append(torch.tensor(row["derived_local_vector"], dtype=torch.float64))
        feedback_vectors.append(torch.tensor(row["derived_feedback_vector"], dtype=torch.float64))
        actual_vectors.append(torch.tensor(row["derived_actual_vector"], dtype=torch.float64))

    for index in range(start_step, args.steps):
        state = states[index]
        step = index + 1
        seed = 51000 + index
        drift_before = candidate_master - repair_master
        gc_c, changed_cc, _, candidate_endpoint_c, loss_cc = gradient(candidate_master, state, False, seed)
        gr_c, changed_rc, endpoint_c, endpoint_pair_c, _loss_rc = gradient(candidate_master, state, True, seed)
        gc_r, changed_cr, _, _, _loss_cr = gradient(repair_master, state, False, seed)
        gr_r, changed_rr, endpoint_r, _, loss_rr = gradient(repair_master, state, True, seed)
        raw_uc_c, next_cm, next_cv = adam_delta(
            gc_c, candidate_m, candidate_v, step,
            learning_rate=args.learning_rate,
        )
        raw_ur_c, _, _ = adam_delta(
            gr_c, candidate_m, candidate_v, step,
            learning_rate=args.learning_rate,
        )
        raw_uc_r, _, _ = adam_delta(
            gc_r, repair_m, repair_v, step,
            learning_rate=args.learning_rate,
        )
        raw_ur_r, next_rm, next_rv = adam_delta(
            gr_r, repair_m, repair_v, step,
            learning_rate=args.learning_rate,
        )
        next_candidate_master = candidate_master + raw_uc_c
        next_repair_master = repair_master + raw_ur_r
        uc_c = next_candidate_master - candidate_master
        ur_c = (candidate_master + raw_ur_c) - candidate_master
        uc_r = (repair_master + raw_uc_r) - repair_master
        ur_r = next_repair_master - repair_master
        drift_after = next_candidate_master - next_repair_master
        local = 0.5 * ((uc_c - ur_c) + (uc_r - ur_r))
        feedback = 0.5 * ((uc_c - uc_r) + (ur_c - ur_r))
        actual = drift_after - drift_before
        residual = actual - local - feedback
        relative = float(torch.linalg.vector_norm(residual)) / max(
            float(torch.linalg.vector_norm(actual)), 1e-30
        )
        raw_row = {
            "step_id": state_ids[index],
            "candidate_at_candidate_state": arm(uc_c),
            "repair_at_candidate_state": arm(ur_c),
            "candidate_at_repair_state": arm(uc_r),
            "repair_at_repair_state": arm(ur_r),
            "drift_before": arm(drift_before),
            "drift_after": arm(drift_after),
            "derived_local_vector": local.detach().double().cpu().reshape(-1).tolist(),
            "derived_feedback_vector": feedback.detach().double().cpu().reshape(-1).tolist(),
            "derived_actual_vector": actual.detach().double().cpu().reshape(-1).tolist(),
            "metadata": {
                "step": step,
                "state_id": state_ids[index],
                "repair_changed_coordinates": [changed_rc, changed_rr],
                "candidate_observation_changed_coordinates": [changed_cc, changed_cr],
                "recurrence_relative": relative,
                "candidate_loss": loss_cc,
                "repair_loss": loss_rr,
                "paired_loss_gap": loss_cc - loss_rr,
            },
        }
        if args.raw_stage_output is not None:
            raw_candidate_gradient_vectors.append(gc_c.detach().float().cpu().reshape(-1))
            raw_repair_gradient_vectors.append(gr_c.detach().float().cpu().reshape(-1))
            raw_candidate_update_vectors.append(raw_uc_c.detach().float().cpu().reshape(-1))
            raw_repair_update_vectors.append(raw_ur_c.detach().float().cpu().reshape(-1))
            if endpoint_c is None:
                raise RuntimeError("raw-stage capture requested but endpoint residual is absent")
            if endpoint_pair_c is None or candidate_endpoint_c is None:
                raise RuntimeError("raw-stage capture requested but endpoint candidate/reference pair is absent")
            if not torch.equal(candidate_endpoint_c, endpoint_pair_c[0].to(candidate_endpoint_c.device)):
                raise RuntimeError("raw-stage candidate endpoint changed between candidate and repair arms")
            raw_endpoint_vectors.append(endpoint_c.detach().float().cpu().reshape(-1))
            raw_endpoint_candidate_vectors.append(candidate_endpoint_c.detach().cpu().reshape(-1))
            raw_endpoint_reference_vectors.append(endpoint_pair_c[1].detach().cpu().reshape(-1))
            raw_endpoint_candidate_dtype = str(candidate_endpoint_c.dtype)
            raw_endpoint_reference_dtype = str(endpoint_pair_c[1].dtype)
            raw_candidate_moments.append(candidate_m.detach().float().cpu().reshape(-1))
            raw_candidate_second_moments.append(candidate_v.detach().float().cpu().reshape(-1))
            raw_repair_moments.append(repair_m.detach().float().cpu().reshape(-1))
            raw_repair_second_moments.append(repair_v.detach().float().cpu().reshape(-1))
        trace.add(
            raw_row["step_id"],
            candidate_at_candidate_state=raw_row["candidate_at_candidate_state"],
            repair_at_candidate_state=raw_row["repair_at_candidate_state"],
            candidate_at_repair_state=raw_row["candidate_at_repair_state"],
            repair_at_repair_state=raw_row["repair_at_repair_state"],
            drift_before=raw_row["drift_before"], drift_after=raw_row["drift_after"],
            metadata=raw_row["metadata"],
        )
        saved_rows.append(raw_row)
        local_vectors.append(local.detach().double().cpu().reshape(-1))
        feedback_vectors.append(feedback.detach().double().cpu().reshape(-1))
        actual_vectors.append(actual.detach().double().cpu().reshape(-1))
        if args.three_stage_output is not None:
            if endpoint_c is None or endpoint_r is None:
                raise RuntimeError("three-stage capture requested but endpoint residual is absent")
            endpoint_vectors.append(endpoint_c)
            candidate_state_gradient_vectors.append((gc_c - gr_c).detach().cpu().reshape(-1))
            candidate_state_update_vectors.append((raw_uc_c - raw_ur_c).detach().cpu().reshape(-1))
        candidate_master = next_candidate_master
        repair_master = next_repair_master
        candidate_m, candidate_v = next_cm, next_cv
        repair_m, repair_v = next_rm, next_rv
        atomic_torch_save({
            "case_id": args.case_id, "state_ids": state_ids,
            "candidate_master": candidate_master.cpu(),
            "repair_master": repair_master.cpu(),
            "candidate_m": candidate_m.cpu(), "candidate_v": candidate_v.cpu(),
            "repair_m": repair_m.cpu(), "repair_v": repair_v.cpu(),
            "next_step": step, "rows": saved_rows,
        }, args.checkpoint)
        print(json.dumps({
            "event": "BOUND_CONSEQUENCE_STEP", "case_id": args.case_id,
            "step": step, "state_id": state_ids[index],
            "recurrence_relative": relative,
        }, sort_keys=True), flush=True)
        del gc_c, gr_c, gc_r, gr_r, local, feedback, actual, residual
        torch.cuda.empty_cache()

    result = trace.finalize()
    loss_records = [row.get("metadata", {}) for row in saved_rows]
    loss_gaps = [float(row.get("paired_loss_gap", 0.0)) for row in loss_records]
    matrix = torch.stack([
        vector for triple in zip(local_vectors, feedback_vectors, actual_vectors)
        for vector in triple
    ])
    statistics = aligned_level_statistics_from_gram(
        (matrix @ matrix.T).numpy(), state_ids=state_ids,
        level_ids=("local", "feedback", "actual"), sign_flip_draws=4000,
        seed=20260820,
    )
    max_relative = max(
        float(row["metadata"]["recurrence_relative"]) for row in saved_rows
    )
    if max_relative > args.recurrence_relative_tolerance:
        result["status"] = "INVALID_RECURRENCE_RELATIVE"
    result.update({
        "runner": "scripts/run_bound_endpoint_consequence_v21.py",
        "architecture": args.architecture,
        "model": str(args.model.resolve()),
        "release": str(args.release_dir),
        "case_plan": str(args.case_plan),
        "carrier": carrier_name,
        "carrier_coordinates": int(carrier.numel()),
        "steps": args.steps,
        "trajectory_status": (
            "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN"
        ),
        "optimizer": {
            "name": "AdamW", "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0,
        },
        "projection": {
            "kind": "candidate_blind_fixed_rademacher",
            "seed_rule": "sha256(case_id)",
            "primary_verdict_uses_projection": False,
        },
        "statistics": statistics,
        "max_recurrence_relative": max_relative,
        "recurrence_relative_tolerance": args.recurrence_relative_tolerance,
        "loss_audit": {
            "recorded": bool(loss_records),
            "any_period_split": any(abs(value) > 1e-8 for value in loss_gaps),
            "max_abs_gap": max((abs(value) for value in loss_gaps), default=0.0),
            "final_gap": loss_gaps[-1] if loss_gaps else None,
            "last_512_mean": (sum(loss_gaps[-min(512, len(loss_gaps)):]) /
                              max(1, min(512, len(loss_gaps)))) if loss_gaps else None,
            "last_512_max_abs": max((abs(value) for value in loss_gaps[-min(512, len(loss_gaps)):]), default=0.0),
            "tolerance": 1e-8,
        },
        "claim_boundary": (
            "Four real counterfactual arms on one declared parameter carrier. "
            "Formation labels are not read or emitted. A 32-step result is a "
            "bounded one-parameter consequence, not full-model training safety."
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    if args.three_stage_output is not None:
        three_stage = {
            "schema": "kernel-analyzer-bound-endpoint-three-stage-v1",
            "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
            "case_id": args.case_id,
            "state_ids": state_ids,
            "sequence_length": expected_length,
            "optimizer": {
                "name": "AdamW", "learning_rate": args.learning_rate,
                "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0,
            },
            "state_path": "live candidate state; candidate and repair evaluated at the identical pre-step state and moments",
            "stages": {
                "operator_output_error": {"coherence_curve": coherence_curve(endpoint_vectors)},
                "parameter_gradient_error": {"coherence_curve": coherence_curve(candidate_state_gradient_vectors)},
                "adamw_effective_update_error": {"coherence_curve": coherence_curve(candidate_state_update_vectors)},
            },
            "claim_boundary": "All three stages come from the same exact endpoint and live candidate-state path. This remains a one-carrier, not full-parameter, result.",
        }
        args.three_stage_output.parent.mkdir(parents=True, exist_ok=True)
        staged_temporary = args.three_stage_output.with_name("." + args.three_stage_output.name + ".tmp")
        staged_temporary.write_text(json.dumps(three_stage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged_temporary.replace(args.three_stage_output)
    if args.raw_stage_output is not None:
        raw_stage = {
            "schema": "kernel-analyzer-bound-endpoint-raw-stage-v1",
            "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
            "case_id": args.case_id,
            "state_ids": state_ids,
            "sequence_length": expected_length,
            "optimizer": {
                "name": "AdamW", "learning_rate": args.learning_rate,
                "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0,
            },
            "state_path": "live candidate state; all vectors are same-state counterfactuals",
            "vectors": {
                "operator_output_error": [v.tolist() for v in raw_endpoint_vectors],
                "operator_output_candidate": [v.tolist() for v in raw_endpoint_candidate_vectors],
                "operator_output_reference": [v.tolist() for v in raw_endpoint_reference_vectors],
                "operator_output_candidate_dtype": raw_endpoint_candidate_dtype,
                "operator_output_reference_dtype": raw_endpoint_reference_dtype,
                "candidate_gradient": [v.tolist() for v in raw_candidate_gradient_vectors],
                "repair_gradient": [v.tolist() for v in raw_repair_gradient_vectors],
                "candidate_update": [v.tolist() for v in raw_candidate_update_vectors],
                "repair_update": [v.tolist() for v in raw_repair_update_vectors],
                "candidate_first_moment_before_step": [v.tolist() for v in raw_candidate_moments],
                "candidate_second_moment_before_step": [v.tolist() for v in raw_candidate_second_moments],
                "repair_first_moment_before_step": [v.tolist() for v in raw_repair_moments],
                "repair_second_moment_before_step": [v.tolist() for v in raw_repair_second_moments],
            },
            "claim_boundary": (
                "Raw same-state replay data for optimizer ablations. It does not "
                "by itself establish a natural warm-phase result."
            ),
        }
        args.raw_stage_output.parent.mkdir(parents=True, exist_ok=True)
        raw_temporary = args.raw_stage_output.with_name("." + args.raw_stage_output.name + ".tmp")
        raw_temporary.write_text(json.dumps(raw_stage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raw_temporary.replace(args.raw_stage_output)
    args.checkpoint.unlink(missing_ok=True)
    print(json.dumps({
        "event": "BOUND_CONSEQUENCE_COMPLETE", "case_id": args.case_id,
        "output": str(args.output), "status": result["status"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
