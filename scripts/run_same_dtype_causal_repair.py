#!/usr/bin/env python3
"""Repair one exact candidate endpoint with its same-dtype AOT reference value."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "archive/round1_code/src")]

from scripts.aot_capture import AOTForwardBackwardCapture  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.run_same_dtype_semantic_oracle import load, write  # noqa: E402
from scripts.run_targeted_causal_repair import (  # noqa: E402
    clone_gradients, gradient_delta_summary,
)
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402
from scripts.same_dtype_semantic_observer import (  # noqa: E402
    SameDtypeSemanticCandidateObserver,
)


ARCHITECTURES = {
    "qwen": "qwen3_1p7b", "mamba": "mamba_130m",
    "phi": "phi4_mini_3p8b", "deepseek8": "deepseek_r1_0528_qwen3_8b",
}


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().reshape(-1)
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": tensor_digest(value), "coordinates": int(flat.numel()),
        "max_abs": float(flat.float().abs().max()) if flat.numel() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--full-coordinate-t1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    args = parser.parse_args()
    if args.repeat != 2 or args.states < 1:
        raise ValueError("same-dtype repair requires two repeats and at least one state")

    t1 = load(args.full_coordinate_t1)
    matches = [row for row in t1["rows"] if str(row["task_id"]) == args.task_id]
    if len(matches) != 1 or matches[0]["verdict"] != "DIRECTIONAL_OPTIMIZATION_BIAS":
        raise RuntimeError("repair target lacks a passed full-coordinate T1")

    release = args.release_dir
    capture = json.loads((release / "capture.json").read_text())
    campaign = load(release / "campaign.json.gz")
    inventory = load(release / "inventory.json.gz")
    plan = load(release / "same_dtype_tasks.json.gz")
    task_matches = [row for row in plan["rows"] if str(row["task_id"]) == args.task_id]
    if len(task_matches) != 1:
        raise RuntimeError("task ID is absent or non-unique in the frozen plan")
    task = task_matches[0]
    endpoint = str(task["exact_aot_endpoint_id"])
    cuts = [
        row for row in plan["reference_cut_tasks"]
        if str(row["task_id"]).removeprefix("same-dtype:") == endpoint
    ]
    if len(cuts) != 1:
        raise RuntimeError("exact AOT repair cut is absent or non-unique")

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.states or file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match the frozen release")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not args.allow_graph_breaks, dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    args.reference_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.reference_cache_dir.resolve())
    active_sink: dict[str, Any] = {"fn": None}

    def dispatch(cut: Any, outputs: tuple[Any, ...]) -> None:
        if active_sink["fn"] is not None:
            active_sink["fn"](cut, outputs)

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
        warm_loss.backward(); torch.cuda.synchronize(device)
    structure = reference_capture.as_dict()
    if not all(structure["reference_cut_runtime"]["gates"].values()):
        raise RuntimeError("same-dtype repair reference cut failed")

    rows = []
    for state_index, state in enumerate(states[:args.states]):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 28000 + state_index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values); baseline_loss.backward(); torch.cuda.synchronize(device)
        baseline_identity = {"loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model)}
        baseline_gradients = clone_gradients(model)

        reference_value: dict[str, torch.Tensor] = {}
        def reference_sink(_cut: Any, outputs: tuple[Any, ...]) -> None:
            tensors = [value for value in outputs if isinstance(value, torch.Tensor)]
            if len(tensors) != 1:
                raise RuntimeError("repair reference endpoint is not one tensor")
            reference_value["value"] = tensors[0].detach().clone()

        active_sink["fn"] = reference_sink
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        ref_loss = reference(values)
        reference_capture.bind_user_outputs(ref_loss)
        ref_loss.register_hook(reference_capture.bind_user_cotangent)
        ref_loss.backward(); torch.cuda.synchronize(device)
        active_sink["fn"] = None
        if "value" not in reference_value:
            raise RuntimeError("same-dtype reference value was not emitted")

        repeat_rows = []
        frozen = None
        for repeat in range(args.repeat):
            arm_rows = {}
            for mode in ("SHAM", "REPAIR"):
                delivered: dict[str, Any] = {}
                def candidate_sink(task_id: str, tensor: torch.Tensor, metadata: Any) -> None:
                    if task_id != args.task_id or delivered:
                        raise RuntimeError("candidate repair endpoint identity changed")
                    before = tensor.detach().clone()
                    reference_tensor = reference_value["value"]
                    if before.shape != reference_tensor.shape or before.dtype != reference_tensor.dtype:
                        raise RuntimeError("same-dtype repair tensor metadata mismatch")
                    if mode == "REPAIR":
                        tensor.copy_(reference_tensor)
                    else:
                        tensor.copy_(before)
                    delivered.update({
                        "before": tensor_summary(before),
                        "reference": tensor_summary(reference_tensor),
                        "delivered": tensor_summary(tensor),
                        "metadata": dict(metadata),
                        "changed_coordinates": int(torch.count_nonzero(before != reference_tensor)),
                    })

                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                model.zero_grad(set_to_none=True)
                observer = SameDtypeSemanticCandidateObserver(
                    modules=modules, campaign_rows=campaign["rows"],
                    inventory_rows=inventory["runtime_call_audit"]["rows"],
                    task_rows=[task], sink=candidate_sink,
                )
                with observer:
                    arm_loss = candidate(values); arm_loss.backward()
                torch.cuda.synchronize(device); observer.validate()
                identity = {"loss": tensor_digest(arm_loss), "gradients": gradient_digest(model)}
                delta = gradient_delta_summary(model, baseline_gradients)
                arm_rows[mode] = {
                    "identity": identity, "endpoint": delivered,
                    "gradient_delta": delta,
                }
            if arm_rows["SHAM"]["identity"] != baseline_identity:
                raise RuntimeError("matched same-dtype sham perturbed the full step")
            if arm_rows["REPAIR"]["endpoint"]["delivered"]["sha256"] != arm_rows["REPAIR"]["endpoint"]["reference"]["sha256"]:
                raise RuntimeError("same-dtype repair was not delivered")
            signature = canonical(arm_rows)
            if frozen is None:
                frozen = signature
            elif signature != frozen:
                raise RuntimeError("same-dtype causal repair changed across repeats")
            repeat_rows.append({"repeat": repeat, "arms": arm_rows})

        repair = repeat_rows[0]["arms"]["REPAIR"]
        rows.append({
            "state_id": state_id, "baseline_identity": baseline_identity,
            "repeats": repeat_rows,
            "repair_changed_endpoint": repair["endpoint"]["changed_coordinates"] > 0,
            "repair_reached_parameter_gradients": repair["gradient_delta"]["changed_parameter_count"] > 0,
        })
        write(args.output, {
            "schema": "kernel-analyzer-same-dtype-causal-repair-v1",
            "status": "RUNNING", "task_id": args.task_id,
            "states_complete": len(rows), "rows": rows,
        })
        del baseline_gradients, reference_value, values
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)

    output = {
        "schema": "kernel-analyzer-same-dtype-causal-repair-v1",
        "status": "COMPLETE_SAME_DTYPE_CAUSAL_REPAIR",
        "architecture": ARCHITECTURES[args.architecture], "task_id": args.task_id,
        "full_coordinate_t1_sha256": t1["result_sha256"],
        "release_capture_sha256": capture["result_sha256"],
        "states": args.states, "repeats": args.repeat, "rows": rows,
        "gates": {
            "exact_wrapper_identity": True, "same_dtype_reference": True,
            "exact_aot_semantic_endpoint": True, "matched_sham_exact": True,
            "repair_nonnull_every_state": all(row["repair_changed_endpoint"] for row in rows),
            "repair_reaches_parameter_gradients_every_state": all(
                row["repair_reached_parameter_gradients"] for row in rows
            ),
        },
        "claim_boundary": "T2/T3 causal carrier only; paired multi-step accumulation remains required.",
    }
    output["causal_t2_t3_positive"] = all(output["gates"].values())
    output["result_sha256"] = canonical(output)
    write(args.output, output)
    print(json.dumps({"status": output["status"], "positive": output["causal_t2_t3_positive"]}))


if __name__ == "__main__":
    main()
