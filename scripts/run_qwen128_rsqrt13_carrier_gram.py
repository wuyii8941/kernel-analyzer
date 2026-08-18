#!/usr/bin/env python3
"""Complete-coordinate T3 certificate for one exact endpoint/carrier pair."""

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

DEFAULT_TASK = "forward:59:in_out_ptr0"
DEFAULT_CARRIER = "model.layers.3.self_attn.q_proj.weight"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--architecture", default="qwen",
                        choices=("qwen", "phi", "mamba", "deepseek8"))
    parser.add_argument("--model", type=Path,
                        default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path,
                        default=ROOT / "results/coverage/qwen_seq128_input_bank.json")
    parser.add_argument("--release-dir", type=Path,
                        default=ROOT / "results/coverage/runtime_releases/qwen_seq128_r1")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--carrier", default=DEFAULT_CARRIER)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--repeat-check-states", type=int, default=1)
    parser.add_argument("--spool-dir", type=Path,
                        default=Path("/data1/tzh/cache/kernel_analyzer_spool/qwen128_rsqrt13_t3"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t3_gram.json.gz")
    args = parser.parse_args()
    if args.states != 32:
        raise ValueError("strict T3 requires all 32 frozen states")
    if not 1 <= args.repeat_check_states <= args.states:
        raise ValueError("repeat-check states must be within the frozen state count")

    release = args.release_dir
    bank_path = args.input_bank
    capture = json.loads((release / "capture.json").read_text())
    campaign = load(release / "campaign.json.gz")
    inventory = load(release / "inventory.json.gz")
    plan = load(release / "same_dtype_tasks.json.gz")
    task_id = args.task_id
    carrier_name = args.carrier
    task_rows = [row for row in plan["rows"] if row["task_id"] == task_id]
    if len(task_rows) != 1:
        raise RuntimeError("exact task is absent or non-unique")
    task = task_rows[0]; endpoint = str(task["exact_aot_endpoint_id"])
    cuts = [row for row in plan["reference_cut_tasks"]
            if str(row["task_id"]).removeprefix("same-dtype:") == endpoint]
    if len(cuts) != 1:
        raise RuntimeError("exact reference cut is absent or non-unique")
    bank = json.loads(bank_path.read_text()); states = bank["states"]
    if len(states) < 32 or file_digest(bank_path) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("frozen bank mismatch")

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    target = dict(model.named_parameters())[carrier_name]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor",
                              fullgraph=not args.allow_graph_breaks, dynamic=False)
    warm_tokens = states[0].get("input_ids", states[0].get("token_ids"))
    if warm_tokens is None:
        raise RuntimeError("input bank state has no token IDs")
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:]); validate_release(wrapper_modules(modules), capture)

    active: dict[str, Any] = {"sink": None}
    def dispatch(cut: Any, outputs: tuple[Any, ...]) -> None:
        if active["sink"] is not None: active["sink"](cut, outputs)
    ref_capture = AOTForwardBackwardCapture(reference_cut_tasks=cuts, reference_value_sink=dispatch)
    with inductor_config.patch({"force_disable_caches": True}):
        reference = torch.compile(LossStep(model), backend=ref_capture.inductor_partition_backend(),
                                  fullgraph=not args.allow_graph_breaks, dynamic=False)
        model.zero_grad(set_to_none=True)
        loss = reference(warm); ref_capture.bind_user_outputs(loss)
        loss.register_hook(ref_capture.bind_user_cotangent); loss.backward(); torch.cuda.synchronize(device)
    if not all(ref_capture.as_dict()["reference_cut_runtime"]["gates"].values()):
        raise RuntimeError("reference cut failed")

    accumulator = StreamingGramAccumulator(
        args.spool_dir, canonical({"task_id": task_id, "carrier": carrier_name})[:24]
    )
    records = []
    for index, state in enumerate(states[:32]):
        state_id = str(state.get("sequence_id", state.get("state_id", index)))
        tokens = state.get("input_ids", state.get("token_ids"))
        if tokens is None:
            raise RuntimeError("input bank state has no token IDs")
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 31000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); base_loss = candidate(values); base_loss.backward()
        torch.cuda.synchronize(device)
        if target.grad is None: raise RuntimeError("baseline carrier gradient absent")
        baseline = target.grad.detach().float().clone()

        ref_value: dict[str, torch.Tensor] = {}
        def ref_sink(_cut: Any, outputs: tuple[Any, ...]) -> None:
            tensors = [x for x in outputs if isinstance(x, torch.Tensor)]
            if len(tensors) != 1: raise RuntimeError("reference endpoint is not one tensor")
            ref_value["value"] = tensors[0].detach().clone()
        active["sink"] = ref_sink
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); ref_loss = reference(values)
        ref_capture.bind_user_outputs(ref_loss); ref_loss.register_hook(ref_capture.bind_user_cotangent)
        ref_loss.backward(); torch.cuda.synchronize(device); active["sink"] = None
        if "value" not in ref_value: raise RuntimeError("reference endpoint absent")

        repaired = []
        changed = []
        repair_repeats = 2 if index < args.repeat_check_states else 1
        for repeat in range(repair_repeats):
            delivered: dict[str, int] = {}
            def sink(task_id: str, tensor: torch.Tensor, _metadata: Any) -> None:
                if task_id != args.task_id or delivered: raise RuntimeError("candidate endpoint drift")
                before = tensor.detach().clone(); ref = ref_value["value"]
                tensor.copy_(ref); delivered["coordinates"] = int(torch.count_nonzero(before != ref))
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            observer = SameDtypeSemanticCandidateObserver(
                modules=modules, campaign_rows=campaign["rows"],
                inventory_rows=inventory["runtime_call_audit"]["rows"],
                task_rows=[task], sink=sink)
            with observer:
                repair_loss = candidate(values); repair_loss.backward()
            torch.cuda.synchronize(device); observer.validate()
            if target.grad is None: raise RuntimeError("repaired carrier gradient absent")
            repaired.append(target.grad.detach().float().clone()); changed.append(delivered["coordinates"])
        if len(repaired) == 2 and not torch.equal(repaired[0], repaired[1]):
            raise RuntimeError("repaired carrier gradient is runtime-unstable")
        correction = baseline - repaired[0]
        flat = correction.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        spool = accumulator.add_array(state_id, flat)
        records.append({"state_id": state_id, "baseline_loss_sha256": tensor_digest(base_loss),
                        "repair_loss_sha256": tensor_digest(repair_loss),
                        "endpoint_changed_coordinates": changed[0],
                        "carrier_nonzero_coordinates": int(torch.count_nonzero(correction)),
                        "carrier_l2": float(torch.linalg.vector_norm(correction).cpu()),
                        "carrier_signed_sum": float(correction.double().sum().cpu()),
                        "repair_repeats": repair_repeats,
                        "vector_sha256": spool["sha256"]})
        print(json.dumps({"event": "STATE_COMPLETE", **records[-1]}), flush=True)
        del baseline, repaired, correction, ref_value, values
        torch.cuda.empty_cache()

    certificate = accumulator.finalize(bootstrap_draws=4000, seed=23091, cleanup=True)
    gates = {
        "all_32_frozen_states": len(records) == 32,
        "same_fixed_model_weights": True,
        "same_dtype_exact_endpoint_repair": True,
        "pilot_two_repeats_bitwise_stable": all(
            row["repair_repeats"] == 2 for row in records[:args.repeat_check_states]
        ),
        "remaining_states_single_repair_after_frozen_repeat_gate": all(
            row["repair_repeats"] == 1 for row in records[args.repeat_check_states:]
        ),
        "repair_nonzero_every_state": all(row["endpoint_changed_coordinates"] > 0 for row in records),
        "real_carrier_nonzero_every_state": all(row["carrier_nonzero_coordinates"] > 0 for row in records),
        "complete_carrier_coordinates": certificate["coordinates"] == target.numel(),
        "cross_state_coherent_carrier": certificate["cluster_bootstrap_95"]["lower_95"] > 0,
    }
    payload = {"schema": "kernel-analyzer-complete-carrier-gram-v1",
               "status": "PASS_T3_COHERENT_REAL_CARRIER" if all(gates.values()) else "FAIL_T3_CARRIER_COHERENCE",
               "task_id": task_id, "exact_aot_endpoint_id": endpoint,
               "carrier_parameter": carrier_name,
               "records": records, "certificate": certificate, "gates": gates,
               "bindings": {"release_capture_sha256": capture["result_sha256"],
                            "input_bank_sha256": file_digest(bank_path),
                            "task_plan_sha256": plan["result_sha256"]}}
    payload["result_sha256"] = canonical(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"status": payload["status"], "lower_95": certificate["cluster_bootstrap_95"]["lower_95"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
