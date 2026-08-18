#!/usr/bin/env python3
"""Batch strict T3 carrier certificates while sharing model/reference execution."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
                str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.streaming import StreamingGramAccumulator
from scripts.aot_capture import AOTForwardBackwardCapture
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules
from scripts.run_generated_fp32_screen import file_digest, load_model, tensor_digest
from scripts.run_same_dtype_semantic_oracle import load
from scripts.run_targeted_full_coordinate import validate_release
from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("qwen", "phi", "mamba", "deepseek8"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--repeat-check-states", type=int, default=1)
    parser.add_argument("--allow-graph-breaks", action="store_true")
    args = parser.parse_args()
    if args.states != 32 or not 1 <= args.repeat_check_states <= args.states:
        raise ValueError("strict T3 requires 32 states and a valid repeat pilot")

    cases = json.loads(args.case_plan.read_text())["cases"]
    if not cases:
        raise RuntimeError("T3 batch plan is empty")
    task_ids = [str(row["task_id"]) for row in cases]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("T3 batch task IDs are not unique")

    release = args.release_dir
    capture = json.loads((release / "capture.json").read_text())
    campaign = load(release / "campaign.json.gz")
    inventory = load(release / "inventory.json.gz")
    plan = load(release / "same_dtype_tasks.json.gz")
    by_task = {str(row["task_id"]): row for row in plan["rows"]}
    if not set(task_ids) <= set(by_task):
        raise RuntimeError("T3 batch contains an absent exact task")
    tasks = {task_id: by_task[task_id] for task_id in task_ids}
    endpoints = {str(row["exact_aot_endpoint_id"]) for row in tasks.values()}
    cuts = [row for row in plan["reference_cut_tasks"]
            if str(row["task_id"]).removeprefix("same-dtype:") in endpoints]
    if {str(row["task_id"]).removeprefix("same-dtype:") for row in cuts} != endpoints:
        raise RuntimeError("T3 reference-cut denominator is incomplete")

    bank = json.loads(args.input_bank.read_text()); states = bank["states"]
    if len(states) < 32 or file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("frozen T3 input bank mismatch")

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    named_parameters = dict(model.named_parameters())
    carriers = {str(row["carrier"]) for row in cases}
    if not carriers <= set(named_parameters):
        raise RuntimeError("T3 batch carrier parameter is absent")

    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor",
                              fullgraph=not args.allow_graph_breaks, dynamic=False)
    warm_tokens = states[0].get("input_ids", states[0].get("token_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:]); validate_release(wrapper_modules(modules), capture)

    active: dict[str, Any] = {"sink": None}
    def dispatch(cut: Any, outputs: tuple[Any, ...]) -> None:
        if active["sink"] is not None:
            active["sink"](cut, outputs)
    ref_capture = AOTForwardBackwardCapture(reference_cut_tasks=cuts, reference_value_sink=dispatch)
    with inductor_config.patch({"force_disable_caches": True}):
        reference = torch.compile(LossStep(model), backend=ref_capture.inductor_partition_backend(),
                                  fullgraph=not args.allow_graph_breaks, dynamic=False)
        model.zero_grad(set_to_none=True)
        loss = reference(warm); ref_capture.bind_user_outputs(loss)
        loss.register_hook(ref_capture.bind_user_cotangent); loss.backward(); torch.cuda.synchronize(device)
    if not all(ref_capture.as_dict()["reference_cut_runtime"]["gates"].values()):
        raise RuntimeError("T3 batch reference cut failed")

    accumulators = {}
    records = {task_id: [] for task_id in task_ids}
    for row in cases:
        task_id = str(row["task_id"]); carrier = str(row["carrier"])
        accumulators[task_id] = StreamingGramAccumulator(
            Path(row["spool_dir"]), canonical({"task_id": task_id, "carrier": carrier})[:24]
        )

    for index, state in enumerate(states[:32]):
        state_id = str(state.get("sequence_id", state.get("state_id", index)))
        tokens = state.get("input_ids", state.get("token_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 31000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); base_loss = candidate(values); base_loss.backward()
        torch.cuda.synchronize(device)
        baselines = {}
        for carrier in carriers:
            gradient = named_parameters[carrier].grad
            if gradient is None: raise RuntimeError(f"baseline carrier gradient absent: {carrier}")
            baselines[carrier] = gradient.detach().float().cpu().clone()

        refs: dict[str, torch.Tensor] = {}
        def ref_sink(cut: Any, outputs: tuple[Any, ...]) -> None:
            endpoint = str(cut.task_id).removeprefix("same-dtype:")
            tensors = [value for value in outputs if isinstance(value, torch.Tensor)]
            if len(tensors) != 1 or endpoint in refs:
                raise RuntimeError("T3 batch reference endpoint changed")
            refs[endpoint] = tensors[0].detach().clone()
        active["sink"] = ref_sink
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); ref_loss = reference(values)
        ref_capture.bind_user_outputs(ref_loss); ref_loss.register_hook(ref_capture.bind_user_cotangent)
        ref_loss.backward(); torch.cuda.synchronize(device); active["sink"] = None
        if set(refs) != endpoints: raise RuntimeError("T3 batch reference values incomplete")

        for case in cases:
            task_id = str(case["task_id"]); carrier = str(case["carrier"])
            task = tasks[task_id]; endpoint = str(task["exact_aot_endpoint_id"])
            repaired = []; changed = []
            repair_repeats = 2 if index < args.repeat_check_states else 1
            for _repeat in range(repair_repeats):
                delivered: dict[str, int] = {}
                def sink(observed_id: str, tensor: torch.Tensor, _metadata: Any) -> None:
                    if observed_id != task_id or delivered:
                        raise RuntimeError("T3 batch candidate endpoint drift")
                    ref = refs[endpoint]
                    if tensor.shape != ref.shape or tensor.dtype != ref.dtype:
                        raise RuntimeError("T3 batch repair metadata mismatch")
                    before = tensor.detach().clone(); tensor.copy_(ref)
                    delivered["coordinates"] = int(torch.count_nonzero(before != ref))
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                model.zero_grad(set_to_none=True)
                observer = SameDtypeSemanticCandidateObserver(
                    modules=modules, campaign_rows=campaign["rows"],
                    inventory_rows=inventory["runtime_call_audit"]["rows"],
                    task_rows=[task], sink=sink)
                with observer:
                    repair_loss = candidate(values); repair_loss.backward()
                torch.cuda.synchronize(device); observer.validate()
                gradient = named_parameters[carrier].grad
                if gradient is None: raise RuntimeError("repaired carrier gradient absent")
                repaired.append(gradient.detach().float().cpu().clone())
                changed.append(delivered["coordinates"])
            if len(repaired) == 2 and not torch.equal(repaired[0], repaired[1]):
                raise RuntimeError("T3 batch repaired carrier is runtime-unstable")
            correction = baselines[carrier] - repaired[0]
            vector = correction.numpy().astype(np.float64, copy=False).reshape(-1)
            spool = accumulators[task_id].add_array(state_id, vector)
            records[task_id].append({
                "state_id": state_id, "baseline_loss_sha256": tensor_digest(base_loss),
                "repair_loss_sha256": tensor_digest(repair_loss),
                "endpoint_changed_coordinates": changed[0],
                "carrier_nonzero_coordinates": int(torch.count_nonzero(correction)),
                "carrier_l2": float(torch.linalg.vector_norm(correction)),
                "carrier_signed_sum": float(correction.double().sum()),
                "repair_repeats": repair_repeats, "vector_sha256": spool["sha256"],
            })
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id,
                          "tasks": len(cases)}), flush=True)
        del baselines, refs, values
        torch.cuda.empty_cache()

    for case in cases:
        task_id = str(case["task_id"]); carrier = str(case["carrier"])
        target = named_parameters[carrier]
        certificate = accumulators[task_id].finalize(
            bootstrap_draws=4000, seed=23091, cleanup=True)
        task_records = records[task_id]
        gates = {
            "all_32_frozen_states": len(task_records) == 32,
            "same_fixed_model_weights": True,
            "same_dtype_exact_endpoint_repair": True,
            "pilot_two_repeats_bitwise_stable": all(
                row["repair_repeats"] == 2 for row in task_records[:args.repeat_check_states]),
            "remaining_states_single_repair_after_frozen_repeat_gate": all(
                row["repair_repeats"] == 1 for row in task_records[args.repeat_check_states:]),
            "repair_nonzero_every_state": all(row["endpoint_changed_coordinates"] > 0 for row in task_records),
            "real_carrier_nonzero_every_state": all(row["carrier_nonzero_coordinates"] > 0 for row in task_records),
            "complete_carrier_coordinates": certificate["coordinates"] == target.numel(),
            "cross_state_coherent_carrier": certificate["cluster_bootstrap_95"]["lower_95"] > 0,
        }
        task = tasks[task_id]
        payload = {
            "schema": "kernel-analyzer-complete-carrier-gram-v1",
            "status": "PASS_T3_COHERENT_REAL_CARRIER" if all(gates.values()) else "FAIL_T3_CARRIER_COHERENCE",
            "task_id": task_id, "exact_aot_endpoint_id": task["exact_aot_endpoint_id"],
            "carrier_parameter": carrier, "records": task_records,
            "certificate": certificate, "gates": gates,
            "execution_protocol": "BATCH_SHARED_BASELINE_REFERENCE_V1",
            "bindings": {"release_capture_sha256": capture["result_sha256"],
                         "input_bank_sha256": file_digest(args.input_bank),
                         "task_plan_sha256": plan["result_sha256"]},
        }
        payload["result_sha256"] = canonical(payload)
        write(Path(case["output"]), payload)
    print(json.dumps({"event": "T3_BATCH_COMPLETE", "tasks": len(cases)}), flush=True)


if __name__ == "__main__":
    main()
