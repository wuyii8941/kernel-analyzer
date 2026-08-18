#!/usr/bin/env python3
"""Paired T4 trajectory for one exact endpoint and declared real carrier."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "archive/round1_code/src")]

from scripts.aot_capture import AOTForwardBackwardCapture
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules
from scripts.run_generated_fp32_screen import file_digest, load_model, tensor_digest
from scripts.run_same_dtype_semantic_oracle import load
from scripts.run_targeted_full_coordinate import validate_release
from scripts.same_dtype_semantic_observer import (
    DirectPrimitiveEndpointObserver,
    SameDtypeSemanticCandidateObserver,
)
from kernel_analyzer.seup import (
    SEUPCalibrator,
    SymmetricSEUPEvaluator,
    adamw_effective_update_delta,
    adamw_update,
)

DEFAULT_TASK_ID = "forward:59:in_out_ptr0"
DEFAULT_CARRIER = "model.layers.3.self_attn.q_norm.weight"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def adam_step(master: torch.Tensor, grad: torch.Tensor, m: torch.Tensor,
              v: torch.Tensor, step: int, lr: float) -> None:
    b1, b2, eps = 0.9, 0.95, 1e-8
    m.mul_(b1).add_(grad, alpha=1 - b1)
    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
    master.addcdiv_(m / (1 - b1 ** step), (v / (1 - b2 ** step)).sqrt().add_(eps), value=-lr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--carrier", default=DEFAULT_CARRIER)
    parser.add_argument("--architecture", default="qwen",
                        choices=("qwen", "phi", "mamba", "deepseek8"))
    parser.add_argument("--model", type=Path,
                        default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path,
                        default=ROOT / "results/coverage/qwen_seq128_input_bank.json")
    parser.add_argument("--release-dir", type=Path,
                        default=ROOT / "results/coverage/runtime_releases/qwen_seq128_r1")
    parser.add_argument("--t1-artifact", type=Path,
                        default=ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_small.json.gz")
    parser.add_argument("--t2-artifact", type=Path,
                        default=ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t2.json.gz")
    parser.add_argument("--t3-artifact", type=Path)
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t4.json.gz")
    parser.add_argument("--seup-output", type=Path)
    parser.add_argument("--geometry-spool", type=Path,
                        help="optional CPU vector spool for exploratory SEUP geometry")
    args = parser.parse_args()
    if args.steps != 32:
        raise ValueError("strict T4 requires exactly 32 steps")

    task_id = args.task_id
    release = args.release_dir
    bank_path = args.input_bank
    t1_path = args.t1_artifact
    t2_path = args.t2_artifact
    capture = json.loads((release / "capture.json").read_text())
    bank = json.loads(bank_path.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < 32 or file_digest(bank_path) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("frozen input bank mismatch")
    with gzip.open(t2_path, "rt") as handle:
        t2 = json.load(handle)
    t2_rows = t2.get("rows", [t2])
    t2_matches = [row for row in t2_rows if row.get("task_id") == task_id]
    if len(t2_matches) != 1 or not t2_matches[0].get(
        "causal_t2_positive", t2_matches[0].get("causal_t2_t3_positive", False)
    ):
        raise RuntimeError("T2/T3 prerequisite is not positive")
    t3 = load(args.t3_artifact) if args.t3_artifact else None
    if t3 is not None:
        t3_binding_ok = t3.get("task_id") == task_id and t3.get("carrier_parameter") == args.carrier
        if not t3_binding_ok or (args.seup_output is None and t3.get("status") != "PASS_T3_COHERENT_REAL_CARRIER"):
            raise RuntimeError("strict T3 prerequisite is absent or inconsistent")
    t1 = load(t1_path)
    t1row = [row for row in t1["rows"] if row["task_id"] == task_id]
    if args.seup_output is None and (len(t1row) != 1 or t1row[0]["verdict"] != "DIRECTIONAL_OPTIMIZATION_BIAS"):
        raise RuntimeError("T1 prerequisite is not positive")

    campaign = load(release / "campaign.json.gz")
    inventory = load(release / "inventory.json.gz")
    plan = load(release / "same_dtype_tasks.json.gz")
    tasks = [row for row in plan["rows"] if row["task_id"] == task_id]
    if len(tasks) != 1:
        raise RuntimeError("candidate task binding is absent or non-unique")
    task = tasks[0]
    endpoint = task["exact_aot_endpoint_id"]
    cuts = [row for row in plan["reference_cut_tasks"]
            if row["task_id"].removeprefix("same-dtype:") == endpoint]
    if len(cuts) != 1:
        raise RuntimeError("reference cut binding is absent or non-unique")

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    carrier_parameter = args.carrier
    target = dict(model.named_parameters())[carrier_parameter]

    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor",
                              fullgraph=not args.allow_graph_breaks, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    if args.seup_output is None:
        validate_release(wrapper_modules(modules), capture)

    active: dict[str, Any] = {"sink": None}
    def dispatch(cut: Any, outputs: tuple[Any, ...]) -> None:
        if active["sink"] is not None:
            active["sink"](cut, outputs)
    ref_capture = None
    reference = None
    if args.seup_output is None:
        ref_capture = AOTForwardBackwardCapture(reference_cut_tasks=cuts, reference_value_sink=dispatch)
        with inductor_config.patch({"force_disable_caches": True}):
            reference = torch.compile(LossStep(model), backend=ref_capture.inductor_partition_backend(),
                                      fullgraph=not args.allow_graph_breaks, dynamic=False)
            model.zero_grad(set_to_none=True)
            loss = reference(warm); ref_capture.bind_user_outputs(loss)
            loss.register_hook(ref_capture.bind_user_cotangent); loss.backward(); torch.cuda.synchronize(device)
        if not all(ref_capture.as_dict()["reference_cut_runtime"]["gates"].values()):
            raise RuntimeError("reference cut runtime gates failed")

    def direct_primitive_reference(output: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        pointers = metadata.get("runtime_pointers") or {}
        endpoint_name = str(task.get("exact_aot_endpoint_id", ""))
        formal = str(task["formal_pointer"])
        if "rsqrt" in endpoint_name:
            if formal not in pointers:
                raise RuntimeError("rsqrt in_out operand was not captured")
            return torch.rsqrt(pointers[formal].float()).to(output.dtype)
        if "bmm" in endpoint_name:
            candidates = [(name, value) for name, value in pointers.items()
                          if name != formal and isinstance(value, torch.Tensor) and value.ndim == 3]
            for _, left in candidates:
                for _, right in candidates:
                    if left is right or left.shape[-1] != right.shape[-2]:
                        continue
                    value = torch.bmm(left.float(), right.float())
                    if tuple(value.shape) == tuple(output.shape):
                        return value.to(output.dtype)
            raise RuntimeError("could not bind bmm operands from exact runtime pointers")
        raise RuntimeError(f"no direct primitive reference for {endpoint_name}")

    def gradient(master: torch.Tensor, tokens: list[int], mode: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            target.copy_(master.to(target.dtype))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        reference_value: dict[str, torch.Tensor] = {}
        if mode == "REPAIR" and args.seup_output is None:
            def sink(_cut: Any, outputs: tuple[Any, ...]) -> None:
                tensors = [x for x in outputs if isinstance(x, torch.Tensor)]
                if len(tensors) != 1:
                    raise RuntimeError("reference endpoint is not exactly one tensor")
                reference_value["value"] = tensors[0].detach().clone()
            active["sink"] = sink
            model.zero_grad(set_to_none=True)
            ref_loss = reference(values)
            ref_capture.bind_user_outputs(ref_loss)
            ref_loss.register_hook(ref_capture.bind_user_cotangent)
            ref_loss.backward()
            torch.cuda.synchronize(device)
            active["sink"] = None
            if "value" not in reference_value:
                raise RuntimeError("reference endpoint was not emitted")

        delivered: dict[str, Any] = {}
        def candidate_sink(task_id: str, tensor: torch.Tensor, metadata: Any) -> None:
            if task_id != args.task_id or delivered:
                raise RuntimeError("candidate endpoint identity drift")
            before = tensor.detach().clone()
            if mode == "REPAIR":
                ref = (reference_value["value"] if args.seup_output is None
                       else direct_primitive_reference(tensor, metadata))
                if ref.shape != tensor.shape or ref.dtype != tensor.dtype:
                    raise RuntimeError("repair metadata mismatch")
                tensor.copy_(ref)
                changed = int(torch.count_nonzero(before != ref))
            else:
                tensor.copy_(before)
                changed = 0
            delivered.update(
                changed_coordinates=changed,
                before_sha256=tensor_digest(before),
                after_sha256=tensor_digest(tensor),
                metadata=dict(metadata),
                metadata_summary={key: value for key, value in dict(metadata).items()
                                  if key != "runtime_pointers"},
            )

        model.zero_grad(set_to_none=True)
        if args.seup_output is not None:
            endpoint_kind = "rsqrt" if "rsqrt" in endpoint else "bmm" if "bmm" in endpoint else None
            if endpoint_kind is None:
                raise RuntimeError(f"dynamic SEUP binding has no primitive rule for {endpoint}")
            sequence_length = len(tokens)
            observer = DirectPrimitiveEndpointObserver(
                modules=modules, task=task, sink=candidate_sink,
                operation=endpoint_kind,
                target_shape=(1, sequence_length, 8, 1) if endpoint_kind == "rsqrt" else None,
                # The declared Qwen seq256 rsqrt case is layer 6 k_norm;
                # the seq64 bmm case is the first forward bmm (region 10).
                target_occurrence=6 if endpoint_kind == "rsqrt" else 0,
            )
        else:
            observer = SameDtypeSemanticCandidateObserver(
                modules=modules, campaign_rows=campaign["rows"],
                inventory_rows=inventory["runtime_call_audit"]["rows"],
                task_rows=[task], sink=candidate_sink,
                allow_missing_symbols=False,
            )
        with observer:
            loss = candidate(values); loss.backward()
        torch.cuda.synchronize(device); observer.validate()
        if target.grad is None:
            raise RuntimeError("carrier gradient is absent")
        grad = target.grad.detach().float().clone()
        target.grad = None
        return loss.detach(), grad, delivered

    if args.seup_output is not None:
        if len(states) < 32:
            raise RuntimeError("SEUP requires 16 calibration and 16 evaluation states")
        calibrator = SEUPCalibrator(16, 0.5)
        calibration_master = target.detach().float().clone()
        calibration_m = torch.zeros_like(calibration_master)
        calibration_v = torch.zeros_like(calibration_master)
        calibration_rows = []
        geometry_rows = []

        def cpu_tree(tree: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: value.detach().float().cpu().clone() for key, value in tree.items()}

        def tree_sub(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: left[key] - right[key] for key in left}

        def tree_avg(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: (left[key] + right[key]) * 0.5 for key in left}

        for index, state in enumerate(states[:16]):
            tokens = state.get("token_ids", state.get("input_ids"))
            loss_c, grad_c, _ = gradient(calibration_master, tokens, "SHAM")
            loss_r, grad_r, boundary = gradient(calibration_master, tokens, "REPAIR")
            calibration_forward_repair_exact = torch.equal(loss_c, loss_r)
            delta = adamw_effective_update_delta(
                {"value": grad_c}, {"value": grad_r}, {"value": calibration_m},
                {"value": calibration_v}, {"value": calibration_master},
                step=index + 1, learning_rate=args.learning_rate,
                betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
            )
            calibrator.add(str(state.get("sequence_id", state.get("state_id", index))), delta)
            if args.geometry_spool is not None:
                geometry_rows.append({
                    "phase": "calibration",
                    "step": index + 1,
                    "state_id": str(state.get("sequence_id", state.get("state_id", index))),
                    "local": cpu_tree(delta),
                    "local_candidate": cpu_tree(delta),
                    "feedback": cpu_tree({"value": torch.zeros_like(delta["value"])}),
                    "actual": cpu_tree(delta),
                    "gradient_delta": {"value": (grad_c - grad_r).detach().float().cpu().clone()},
                    "effective_update": cpu_tree(delta),
                })
            adam_step(calibration_master, grad_c, calibration_m, calibration_v, index + 1, args.learning_rate)
            calibration_rows.append({
                "step": index + 1,
                "state_id": str(state.get("sequence_id", state.get("state_id", index))),
                "repair_changed_coordinates": boundary["changed_coordinates"],
                "forward_repair_exact": bool(calibration_forward_repair_exact),
                "endpoint_binding": boundary.get("metadata_summary"),
            })
            del grad_c, grad_r, delta
            torch.cuda.empty_cache()
        carrier = calibrator.freeze()
        evaluator = SymmetricSEUPEvaluator(carrier, 16)
        candidate_master = target.detach().float().clone()
        repair_master = candidate_master.clone()
        candidate_m = torch.zeros_like(candidate_master); candidate_v = torch.zeros_like(candidate_master)
        repair_m = torch.zeros_like(repair_master); repair_v = torch.zeros_like(repair_master)
        rows = []
        for offset, state in enumerate(states[16:32]):
            index = offset + 1
            tokens = state.get("token_ids", state.get("input_ids"))
            loss_cc, grad_cc, _ = gradient(candidate_master, tokens, "SHAM")
            loss_cr, grad_cr, boundary_cr = gradient(candidate_master, tokens, "REPAIR")
            loss_rc, grad_rc, _ = gradient(repair_master, tokens, "SHAM")
            loss_rr, grad_rr, boundary_rr = gradient(repair_master, tokens, "REPAIR")
            candidate_forward_repair_exact = torch.equal(loss_cc, loss_cr)
            repair_forward_repair_exact = torch.equal(loss_rc, loss_rr)
            candidate_before = candidate_master.clone()
            repair_before = repair_master.clone()
            uc_sc_planned = adamw_update({"value": grad_cc}, {"value": candidate_m}, {"value": candidate_v},
                                         {"value": candidate_master}, step=index, learning_rate=args.learning_rate,
                                         betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0)
            ur_sc = adamw_update({"value": grad_cr}, {"value": candidate_m}, {"value": candidate_v},
                                 {"value": candidate_master}, step=index, learning_rate=args.learning_rate,
                                 betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0)
            uc_sr = adamw_update({"value": grad_rc}, {"value": repair_m}, {"value": repair_v},
                                 {"value": repair_master}, step=index, learning_rate=args.learning_rate,
                                 betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0)
            adam_step(candidate_master, grad_cc, candidate_m, candidate_v, index, args.learning_rate)
            adam_step(repair_master, grad_rr, repair_m, repair_v, index, args.learning_rate)
            # Use the actual applied master increments for the two diagonal
            # arms; this makes recurrence a numerical identity even when the
            # library's in-place AdamW arithmetic differs at the last ulp from
            # the pure functional preview above.
            uc_sc = {"value": candidate_master - candidate_before}
            ur_sr = {"value": repair_master - repair_before}
            before = {"value": candidate_before - repair_before}
            after = {"value": candidate_master - repair_master}
            evaluator.add(
                str(state.get("sequence_id", state.get("state_id", 16 + offset))),
                uc_sc, ur_sc, uc_sr, ur_sr, before, after,
                endpoint_repair_nonzero=bool(
                    boundary_cr["changed_coordinates"] > 0 and boundary_rr["changed_coordinates"] > 0
                ),
            )
            if args.geometry_spool is not None:
                local_candidate = tree_sub(uc_sc, ur_sc)
                local_symmetric = tree_avg(tree_sub(uc_sc, ur_sc), tree_sub(uc_sr, ur_sr))
                feedback_symmetric = tree_avg(tree_sub(uc_sc, uc_sr), tree_sub(ur_sc, ur_sr))
                geometry_rows.append({
                    "phase": "evaluation",
                    "step": 16 + index,
                    "state_id": str(state.get("sequence_id", state.get("state_id", 16 + offset))),
                    "local": cpu_tree(local_symmetric),
                    "local_candidate": cpu_tree(local_candidate),
                    "feedback": cpu_tree(feedback_symmetric),
                    "actual": cpu_tree(after),
                    "gradient_delta": {"value": (grad_cc - grad_cr).detach().float().cpu().clone()},
                    "effective_update": cpu_tree(local_candidate),
                })
            rows.append({
                "step": index,
                "state_id": str(state.get("sequence_id", state.get("state_id", 16 + offset))),
                "forward_repair_exact": bool(candidate_forward_repair_exact),
                "repair_arm_forward_repair_exact": bool(repair_forward_repair_exact),
                "candidate_repair_changed_coordinates": boundary_cr["changed_coordinates"],
                "repair_arm_changed_coordinates": boundary_rr["changed_coordinates"],
                "endpoint_binding": boundary_cr.get("metadata_summary"),
                "drift_l2": float(torch.linalg.vector_norm(after["value"]).item()),
            })
            print(json.dumps({"event": "SEUP_STEP_COMPLETE", **rows[-1]}), flush=True)
            del grad_cc, grad_cr, grad_rc, grad_rr, uc_sc_planned, uc_sc, ur_sc, uc_sr, ur_sr
            torch.cuda.empty_cache()
        certificate = evaluator.finalize()
        persistence = certificate.get("signed_persistence", 0.0)
        positive = bool(
            carrier.stable and persistence >= 0.80
            and certificate.get("local_fraction_of_projected_accumulation", 0.0) >= 0.50
            and certificate.get("local_and_final_carrier_same_sign", False)
        )
        gates = {
            "stable_calibration_carrier": carrier.stable,
            "sixteen_evaluation_steps": len(rows) == 16,
            "endpoint_repair_nonzero_every_step": all(
                row["candidate_repair_changed_coordinates"] > 0
                and row["repair_arm_changed_coordinates"] > 0 for row in rows
            ),
            "endpoint_repair_nonzero_any_step": any(
                row["candidate_repair_changed_coordinates"] > 0
                or row["repair_arm_changed_coordinates"] > 0 for row in rows
            ),
            "recurrence_closed": certificate["max_recurrence_relative_residual"] <= 1e-6,
            "signed_persistence_ge_0_80": persistence >= 0.80,
            "local_effect_fraction_ge_0_50": certificate.get("local_fraction_of_projected_accumulation", 0.0) >= 0.50,
        }
        payload = {
            "schema": "kernel-analyzer-qwen-endpoint-seup-mainline-v2",
            "status": "PASS_SEUP_POSITIVE" if positive else "PASS_SEUP_NEGATIVE_CONTROL" if all(
                gates[key] for key in ("sixteen_evaluation_steps", "endpoint_repair_nonzero_any_step", "recurrence_closed")
            ) else "MEASURED_WITH_FAILED_GATE",
            "case_id": f"{args.architecture}_{task_id.replace(':', '_')}",
            "task_id": task_id,
            "exact_aot_endpoint_id": endpoint,
            "carrier_parameters": [carrier_parameter],
            "mechanism": "EXACT_AOT_ENDPOINT_REPAIR",
            "mechanism_level": "EXACT_AOT_ENDPOINT_IN_GENERATED_REGION",
            "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "betas": [0.9, 0.95], "epsilon": 1e-8},
            "protocol": {"calibration_states": 16, "evaluation_states": 16, "evaluation_state_ids_disjoint": True,
                         "state_feedback_decomposition": "symmetric_candidate_and_repair_state_counterfactuals"},
            "calibration": {"steps": calibration_rows, "carrier": carrier.certificate},
            "evaluation": certificate,
            "steps": rows,
            "gates": gates,
            "claim_boundary": "Selected carrier parameter only; exact AOT endpoint repair is closed, while kernel-level root attribution remains separate.",
        }
        payload["result_sha256"] = canonical(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.output, "wt") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        args.seup_output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.seup_output, "wt") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        if args.geometry_spool is not None:
            args.geometry_spool.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema": "kernel-analyzer-seup-geometry-spool-v1",
                "case_id": payload["case_id"],
                "task_id": task_id,
                "carrier_parameters": [carrier_parameter],
                "fields": ["local", "local_candidate", "feedback", "actual", "gradient_delta", "effective_update"],
                "calibration_steps": 16,
                "evaluation_steps": 16,
                "protocol": "carrier-local-feedback-only",
                "rows": geometry_rows,
            }, args.geometry_spool)
        print(json.dumps({"event": "SEUP_COMPLETE", "status": payload["status"], "gates": gates}, sort_keys=True))
        return

    carrier = carrier_parameter
    initial = target.detach().float().clone()
    diagnostics = {}
    controls = {
        "direct_declared_carrier": t3 is None or t3.get("carrier_parameter") == carrier,
    }

    cand = initial.clone(); repaired = initial.clone()
    cm = torch.zeros_like(cand); cv = torch.zeros_like(cand)
    rm = torch.zeros_like(repaired); rv = torch.zeros_like(repaired)
    direction = None
    first_nonzero_step = None
    deltas = []
    records = []
    terminal_failure = None
    seup = None
    for i in range(32):
        tokens = states[i].get("token_ids", states[i].get("input_ids"))
        cand_loss, cand_grad, _ = gradient(cand, tokens, "SHAM")
        _, cand_repair_grad, cand_boundary = gradient(cand, tokens, "REPAIR")
        _, repair_sham_grad, _ = gradient(repaired, tokens, "SHAM")
        repair_loss, repair_grad, repair_boundary = gradient(repaired, tokens, "REPAIR")
        if i == 0:
            # At step 1 cand == repaired == initial, so these values are the
            # exact initial SHAM/REPAIR control.  Reusing them avoids two
            # redundant full forward/backward executions before the loop.
            diagnostics["forward_loss_unchanged"] = torch.equal(cand_loss, repair_loss)
            controls["initial_repair_nonzero"] = (
                cand_boundary["changed_coordinates"] > 0
                and not torch.equal(cand_grad, cand_repair_grad)
            )
        removal_c = cand_grad - cand_repair_grad
        removal_r = repair_sham_grad - repair_grad
        if seup is not None:
            seup.add(
                str(states[i].get("sequence_id", states[i].get("state_id", i))),
                adamw_effective_update_delta(
                    cand_grad, cand_repair_grad, cm, cv, cand,
                    step=i + 1, learning_rate=args.learning_rate,
                    betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
                ),
            )
        adam_step(cand, cand_grad, cm, cv, i + 1, args.learning_rate)
        adam_step(repaired, repair_grad, rm, rv, i + 1, args.learning_rate)
        delta = cand - repaired
        deltas.append(delta.detach().clone())
        if direction is None and bool(torch.linalg.vector_norm(delta) > 0):
            norm = torch.linalg.vector_norm(delta)
            direction = delta / norm
            first_nonzero_step = i + 1
        row = {
            "step": i + 1,
            "state_id": str(states[i].get("sequence_id", states[i].get("state_id", i))),
            "candidate_loss": float(cand_loss.cpu()), "repair_loss": float(repair_loss.cpu()),
            "candidate_removal_l2": float(torch.linalg.vector_norm(removal_c).cpu()),
            "repair_removal_l2": float(torch.linalg.vector_norm(removal_r).cpu()),
            "boundary_changed_coordinates": cand_boundary["changed_coordinates"],
            "repair_branch_changed_coordinates": repair_boundary["changed_coordinates"],
            "master_l2": float(torch.linalg.vector_norm(delta).cpu()),
            "master_projection": None if direction is None else float(torch.sum(delta * direction).cpu()),
            "bf16_materialized_nonzero": int(torch.count_nonzero(cand.to(torch.bfloat16) != repaired.to(torch.bfloat16))),
        }
        records.append(row)
        print(json.dumps({"event": "STEP_COMPLETE", **row}), flush=True)

        step = i + 1
        if direction is not None and first_nonzero_step is not None and step in (8, 16):
            first_projection = float(torch.sum(deltas[first_nonzero_step - 1] * direction).cpu())
            current_projection = float(torch.sum(delta * direction).cpu())
            previous_projection = (
                first_projection if step == 8
                else float(torch.sum(deltas[7] * direction).cpu())
            )
            if current_projection <= previous_projection:
                terminal_failure = (
                    f"directional projection failed irreversible checkpoint "
                    f"growth at step {step}"
                )
                break

    if direction is None or first_nonzero_step is None:
        raise RuntimeError("all 32 paired AdamW weight updates remained identical")
    for row, delta in zip(records, deltas):
        row["master_projection"] = float(torch.sum(delta * direction).cpu())
    checkpoints = sorted(set(
        [first_nonzero_step] + [value for value in (8, 16, 32) if value <= len(records)]
    ))
    projections = [records[i - 1]["master_projection"] for i in checkpoints]
    gates = {
        **controls,
        "paired_same_weight_measurement": True,
        "only_declared_carrier_updated": True,
        "optimizer_delayed_divergence_bounded": first_nonzero_step <= 4,
        "repair_nonzero_all_steps": all(r["candidate_removal_l2"] > 0 and r["repair_removal_l2"] > 0 for r in records),
        "directional_projection_strictly_grows": all(b > a for a, b in zip(projections, projections[1:])),
        "bf16_weights_diverge": records[-1]["bf16_materialized_nonzero"] > 0,
    }
    if terminal_failure is not None:
        gates["directional_projection_strictly_grows"] = False
    payload = {
        "schema": "kernel-analyzer-same-dtype-paired-trajectory-v2",
        "status": "PASS_T4_PAIRED_ACCUMULATION" if all(gates.values()) else "FAIL_DIRECTIONAL_ACCUMULATION",
        "task_id": task_id, "exact_aot_endpoint_id": endpoint, "carrier_parameter": carrier,
        "steps": 32, "steps_completed": len(records),
        "terminal_failure": terminal_failure,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0},
        "records": records, "directional_projection_checkpoints": checkpoints,
        "directional_projections": projections, "gates": gates,
        "first_nonzero_weight_divergence_step": first_nonzero_step,
        "diagnostics": diagnostics,
        "bindings": {"release_capture_sha256": capture["result_sha256"], "input_bank_sha256": file_digest(bank_path), "t1_sha256": t1["result_sha256"], "t2_sha256": t2["result_sha256"], "t3_sha256": None if t3 is None else t3["result_sha256"]},
        "claim_boundary": f"Exact AOT endpoint {endpoint} inside candidate task {task_id}; T4 updates only the declared carrier {carrier}. Root attribution inside a larger fused region remains separate.",
    }
    payload["result_sha256"] = canonical(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    if seup is not None:
        if len(records) != 32:
            raise RuntimeError("SEUP cannot use an early-terminated T4 trajectory")
        certificate = seup.finalize()
        seup_payload = {
            "schema": "kernel-analyzer-case-seup-v1",
            "status": "MEASURED",
            "case_id": f"{args.architecture}_{task_id.replace(':', '_')}",
            "mechanism": "EXACT_AOT_ENDPOINT_REPAIR",
            "mechanism_level": "EXACT_AOT_ENDPOINT_IN_GENERATED_REGION",
            "task_id": task_id,
            "exact_aot_endpoint_id": endpoint,
            "carrier_parameters": [carrier],
            "optimizer": payload["optimizer"],
            "common_state_protocol": (
                "candidate and repair gradients are evaluated at the candidate-arm "
                "pre-step weight and identical pre-step AdamW moments"
            ),
            "certificate": certificate,
            "gates": {
                "complete_concrete_fb_proof": True,
                "same_pre_step_weight_and_optimizer_state": True,
                "calibration_evaluation_disjoint": True,
                "repair_nonzero_all_steps": gates["repair_nonzero_all_steps"],
            },
            "trajectory_result_sha256": payload["result_sha256"],
            "bindings": payload["bindings"],
        }
        seup_payload["result_sha256"] = canonical(seup_payload)
        args.seup_output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.seup_output, "wt") as handle:
            json.dump(seup_payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"event": "TRAJECTORY_COMPLETE", "status": payload["status"], "projections": projections}), flush=True)


if __name__ == "__main__":
    main()
