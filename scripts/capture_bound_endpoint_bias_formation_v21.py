#!/usr/bin/env python3
"""Capture v2.1 formation maps for already-bound exact endpoint repairs.

This runner deliberately reuses the frozen same-dtype release, exact AOT cut,
candidate observer, and a declared downstream parameter carrier.  It adds the
missing scientific measurement: independent open-loop populations for the
local endpoint residual, parameter-gradient residual, and stateless effective
update residual.  Large vectors are spooled under /data1 and deleted after a
single population Gram has been produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
                str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.bias_formation_v21 import (  # noqa: E402
    FormationPolicy,
    summarize_streamed_state_vector_files,
)
from kernel_analyzer.reference_relative_oracle import (  # noqa: E402
    ReferenceRelativeObservation,
    certify_reference_relative,
)
from scripts.aot_capture import AOTForwardBackwardCapture  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import file_digest, load_model  # noqa: E402
from scripts.run_same_dtype_semantic_oracle import load  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver  # noqa: E402


LR = 1.0e-4


def validate_runtime_structure(
    modules: list[tuple[Any, str]], capture: dict[str, Any]
) -> None:
    """Bind the runtime graph without requiring unstable codegen byte identity.

    Exact endpoint identity and call cardinality are checked later by the
    candidate observer for every state.  Here we only reject a changed graph
    partition/order; cache-path and generated-source byte changes are not a
    scientific gate.
    """

    observed = [phase.upper() for _, phase in modules]
    expected = [str(row["phase"]).upper() for row in capture["modules"]]
    if observed != expected:
        raise RuntimeError(
            f"runtime F+B wrapper structure changed: {observed} != {expected}"
        )


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def spool_vector(root: Path, case_id: str, layer: str, partition: str,
                 state_id: str, value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().float().cpu().contiguous().numpy().reshape(-1)
    target = root / case_id / layer / partition / f"{state_id}.f32"
    target.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(target)
    return {
        "state_id": state_id,
        "path": str(target),
        "storage_dtype": "float32",
        "coordinate_count": int(array.size),
    }


def common_state(state: dict[str, Any], seed: int, weights_digest: str) -> dict[str, str]:
    tokens = state.get("input_ids", state.get("token_ids"))
    input_digest = canonical(tokens)
    rng_digest = canonical({"cpu_seed": seed, "cuda_seed": seed})
    optimizer = canonical({"name": "STATELESS_SGD_FP32_MASTER", "lr": LR})
    none = canonical({"name": "none"})
    return {
        "candidate_weights_digest": weights_digest,
        "repair_weights_digest": weights_digest,
        "candidate_optimizer_digest": optimizer,
        "repair_optimizer_digest": optimizer,
        "candidate_input_digest": input_digest,
        "repair_input_digest": input_digest,
        "candidate_rng_digest": rng_digest,
        "repair_rng_digest": rng_digest,
        "candidate_scheduler_digest": none,
        "repair_scheduler_digest": none,
        "candidate_loss_scaler_digest": none,
        "repair_loss_scaler_digest": none,
    }


def reference_cut_gates(capture: AOTForwardBackwardCapture) -> dict[str, bool]:
    """Validate runtime cut binding without serializing the full AOT graph."""

    return {
        "all_tasks_extracted": (
            len(capture.reference_cut_extractions) == len(capture.reference_cut_tasks)
        ),
        "all_extracted_ports_exact": all(
            row["graph_binding"]["port_routes_exact"]
            for row in capture.reference_cut_extractions
        ),
        "all_replays_bitwise_equal": all(
            observation["bitwise_equal"]
            for run in capture.reference_cut_replay_runs
            for observation in run["observations"]
        ),
        "all_replays_finite": all(
            observation["all_finite"]
            for run in capture.reference_cut_replay_runs
            for observation in run["observations"]
        ),
        "all_tasks_replayed_at_least_once": (
            capture._verified_reference_cut_task_ids
            == {str(task["task_id"]) for task in capture.reference_cut_tasks}
        ),
    }


def align_reference_to_candidate(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Apply only the compiler-bound contiguous storage view at a region port."""
    if reference.dtype != candidate.dtype or reference.numel() != candidate.numel():
        raise RuntimeError(
            "endpoint metadata differs from region reference: "
            f"candidate={tuple(candidate.shape)}/{candidate.dtype}/{candidate.numel()} "
            f"reference={tuple(reference.shape)}/{reference.dtype}/{reference.numel()}"
        )
    if reference.shape == candidate.shape:
        return reference
    if not reference.is_contiguous() or not candidate.is_contiguous():
        raise RuntimeError("semantic-region storage view is not contiguous")
    return reference.reshape(candidate.shape)


def partial_reduction_reference(
    metadata: Mapping[str, Any], candidate: torch.Tensor,
    *, sequence_length: int, feature_count: int,
) -> torch.Tensor:
    """Reconstruct an Inductor split-reduction buffer from its bound input.

    This is not the final AOT reduction output.  It is the exact real-valued
    partial-reduction contract implemented by the generated buffer: contiguous
    [token, head, feature] input, partitioned into equally sized token groups,
    with token and head axes reduced in FP32.
    """
    pointers = metadata.get("runtime_pointers") or {}
    source = pointers.get("in_ptr0")
    if not isinstance(source, torch.Tensor):
        raise RuntimeError("partial reduction has no bound in_ptr0")
    if feature_count <= 0 or candidate.numel() % feature_count:
        raise RuntimeError("partial reduction feature partition is invalid")
    groups = candidate.numel() // feature_count
    if groups <= 0 or sequence_length % groups:
        raise RuntimeError("partial reduction token groups are invalid")
    if source.numel() % (sequence_length * feature_count):
        raise RuntimeError("partial reduction input coordinates are invalid")
    heads = source.numel() // (sequence_length * feature_count)
    tokens_per_group = sequence_length // groups
    logical = source.reshape(sequence_length, heads, feature_count)
    reference = logical.reshape(
        groups, tokens_per_group, heads, feature_count
    ).float().sum(dim=(1, 2))
    if candidate.dtype != torch.float32:
        raise RuntimeError("partial reduction buffer is not FP32")
    if candidate.ndim >= 2 and tuple(candidate.shape[-2:]) == (feature_count, groups):
        return reference.transpose(0, 1).reshape(candidate.shape)
    if candidate.is_contiguous():
        return reference.reshape(candidate.shape)
    raise RuntimeError(
        "partial reduction compiler view is unsupported: "
        f"shape={tuple(candidate.shape)} stride={tuple(candidate.stride())} "
        f"groups={groups} feature_count={feature_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("qwen", "phi", "mamba", "deepseek8", "deepseek8b"),
        required=True,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--engineering-reach-only", action="store_true")
    parser.add_argument("--screening-gram", action="store_true")
    parser.add_argument("--extended-confirmation", action="store_true")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    args = parser.parse_args()
    # Inventory/build scripts use the descriptive model key ``deepseek8b``;
    # the shared model loader historically used ``deepseek8``.  Normalize the
    # alias once so a valid frozen plan cannot fail before measurement.
    if args.architecture == "deepseek8b":
        args.architecture = "deepseek8"
    if args.engineering_reach_only and args.screening_gram:
        raise ValueError("engineering reach and screening Gram are distinct modes")
    if (not args.engineering_reach_only and not args.screening_gram
            and not args.extended_confirmation and args.states != 32):
        raise ValueError("v2.1 formation requires frozen 16+16 states")
    if args.extended_confirmation and (
            args.engineering_reach_only or args.screening_gram or args.states < 64
            or args.states % 2):
        raise ValueError("extended confirmation requires an even 32+32 or larger formal population")
    if args.engineering_reach_only and not 1 <= args.states <= 2:
        raise ValueError("engineering reach screen accepts one or two states")
    if args.screening_gram and not 4 <= args.states <= 8:
        raise ValueError("screening Gram accepts four to eight states")
    cases = json.loads(args.case_plan.read_text())["cases"]
    if not cases:
        raise ValueError("case plan is empty")
    task_ids = [str(case["task_id"]) for case in cases]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("case task IDs must be unique")

    capture = json.loads((args.release_dir / "capture.json").read_text())
    campaign = load(args.release_dir / "campaign.json.gz")
    inventory = load(args.release_dir / "inventory.json.gz")
    plan = load(args.release_dir / "same_dtype_tasks.json.gz")
    by_task = {str(row["task_id"]): row for row in plan["rows"]}
    if not set(task_ids) <= set(by_task):
        raise ValueError("case plan references an absent frozen task")
    tasks = {task_id: by_task[task_id] for task_id in task_ids}
    cases_by_task = {str(case["task_id"]): case for case in cases}
    references_by_cut_task: dict[str, str] = {}
    cuts = []
    frozen_cuts = {
        str(row["task_id"]).removeprefix("same-dtype:"): row
        for row in plan["reference_cut_tasks"]
    }
    for case in cases:
        task_id = str(case["task_id"])
        custom = case.get("reference_cut_task")
        if case.get("reference_method") in {
            "PARTIAL_REDUCTION_FROM_BOUND_INPUT", "EXTERNAL_FP32_RECOMPUTE"
        }:
            continue
        if custom is not None:
            cut = dict(custom)
            cuts.append(cut)
            references_by_cut_task[str(cut["task_id"])] = task_id
            continue
        endpoint = tasks[task_id].get("exact_aot_endpoint_id")
        if endpoint is None or str(endpoint) not in frozen_cuts:
            raise RuntimeError(f"reference cut is absent for {task_id}")
        cut = frozen_cuts[str(endpoint)]
        cuts.append(cut)
        references_by_cut_task[str(cut["task_id"])] = task_id
    frozen_bank = json.loads(args.input_bank.read_text())
    bank = json.loads((args.state_bank or args.input_bank).read_text())
    states = bank.get("states", bank.get("records"))
    frozen_states = frozen_bank.get("states", frozen_bank.get("records"))
    if len(states) < args.states or not frozen_states:
        raise RuntimeError("input bank does not contain the requested states")
    frozen_tokens = frozen_states[0].get(
        "input_ids", frozen_states[0].get("token_ids")
    )
    if canonical(frozen_tokens) != capture["input"]["token_ids_sha256"]:
        raise RuntimeError("the release anchor token sequence changed")
    expected_length = int(capture["input"]["sequence_length"])
    if any(len(state.get("input_ids", state.get("token_ids"))) != expected_length
           for state in states[:args.states]):
        raise RuntimeError("state bank changes the frozen sequence shape")
    state_ids = [str(state.get("state_id", state.get("sequence_id", index)))
                 for index, state in enumerate(states[:args.states])]
    if len(state_ids) != len(set(state_ids)):
        raise RuntimeError("state bank contains duplicate state IDs")

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    parameters = dict(model.named_parameters())
    carriers = {str(case["carrier"]) for case in cases}
    if not carriers <= set(parameters):
        raise RuntimeError("a declared carrier parameter is absent")

    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor",
                              fullgraph=not args.allow_graph_breaks, dynamic=False)
    warm_tokens = states[0].get("input_ids", states[0].get("token_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
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
            reference_cut_tasks=cuts, reference_value_sink=dispatch)
        with inductor_config.patch({"force_disable_caches": True}):
            reference = torch.compile(
                LossStep(model), backend=reference_capture.inductor_partition_backend(),
                fullgraph=not args.allow_graph_breaks, dynamic=False)
            model.zero_grad(set_to_none=True)
            warm_loss = reference(warm); reference_capture.bind_user_outputs(warm_loss)
            warm_loss.register_hook(reference_capture.bind_user_cotangent)
            warm_loss.backward(); torch.cuda.synchronize(device)
        if not all(reference_cut_gates(reference_capture).values()):
            raise RuntimeError("reference cut failed")

    # The frozen checkpoint manifest binds both counterfactual arms.  No model
    # or optimizer state is mutated anywhere in this open-loop runner.
    index = args.model / "model.safetensors.index.json"
    weights_digest = file_digest(index if index.exists() else args.model / "config.json")
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        task_id: {layer: [] for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")}
        for task_id in task_ids
    }
    metadata: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    reference_relative: dict[str, list[dict[str, Any]]] = {
        task_id: [] for task_id in task_ids
    }
    reach_rows: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    args.spool_dir.mkdir(parents=True, exist_ok=True)

    for state_index, state in enumerate(states[:args.states]):
        state_id = str(state.get("state_id", state.get("sequence_id", state_index)))
        split_index = args.states // 2 if args.extended_confirmation else 16
        partition = "calibration" if state_index < split_index else "confirmation"
        tokens = state.get("input_ids", state.get("token_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 41000 + state_index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); candidate_loss = candidate(values); candidate_loss.backward()
        torch.cuda.synchronize(device)
        baseline_gradients = {
            carrier: parameters[carrier].grad.detach().float().cpu().clone()
            for carrier in carriers
        }

        references: dict[str, torch.Tensor] = {}
        def reference_sink(cut: Any, outputs: tuple[Any, ...]) -> None:
            endpoint = references_by_cut_task[str(cut.task_id)]
            tensors = [value for value in outputs if isinstance(value, torch.Tensor)]
            if len(tensors) != 1 or endpoint in references:
                raise RuntimeError("reference endpoint cardinality changed")
            references[endpoint] = tensors[0].detach().clone()
        if reference is not None and reference_capture is not None:
            active["sink"] = reference_sink
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True); reference_loss = reference(values)
            reference_capture.bind_user_outputs(reference_loss)
            reference_loss.register_hook(reference_capture.bind_user_cotangent)
            reference_loss.backward(); torch.cuda.synchronize(device); active["sink"] = None

        # One observation-only sham captures every selected endpoint in the
        # frozen hotspot matrix.  Running one sham per case would add no
        # evidence and would multiply full F+B work by the candidate count.
        observed_locals: dict[str, torch.Tensor] = {}
        def sham_sink(observed_id: str, tensor: torch.Tensor, _metadata: Any) -> None:
            if observed_id in observed_locals or observed_id not in tasks:
                raise RuntimeError("candidate endpoint identity drifted")
            case = cases_by_task[observed_id]
            if case.get("reference_method") == "PARTIAL_REDUCTION_FROM_BOUND_INPUT":
                feature_count = int(parameters[str(case["carrier"])].numel())
                references[observed_id] = partial_reduction_reference(
                    _metadata, tensor, sequence_length=expected_length,
                    feature_count=feature_count,
                ).detach().clone()
            elif case.get("reference_method") == "EXTERNAL_FP32_RECOMPUTE":
                symbol = str(_metadata.get("external_symbol"))
                reference = fp32_external_reference(
                    symbol,
                    _metadata.get("runtime_args", ()),
                    _metadata.get("runtime_kwargs", {}),
                ).to(tensor.dtype)
                references[observed_id] = align_reference_to_candidate(
                    reference, tensor
                ).detach().clone()
            reference_value = align_reference_to_candidate(references[observed_id], tensor)
            observed_locals[observed_id] = (
                tensor.detach().float().cpu() - reference_value.detach().float().cpu()
            )
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        sham = SameDtypeSemanticCandidateObserver(
            modules=modules, campaign_rows=campaign["rows"],
            inventory_rows=inventory["runtime_call_audit"]["rows"],
            task_rows=list(tasks.values()), sink=sham_sink,
            include_unresolved_tasks=True)
        with sham:
            sham_loss = candidate(values); sham_loss.backward()
        torch.cuda.synchronize(device); sham.validate()
        if set(observed_locals) != set(task_ids):
            raise RuntimeError("shared sham did not observe the complete hotspot matrix")
        for carrier in carriers:
            sham_gradient = parameters[carrier].grad.detach().float().cpu()
            if not torch.equal(sham_gradient, baseline_gradients[carrier]):
                raise RuntimeError("observation-only matched sham changed a carrier gradient")

        for case in cases:
            task_id = str(case["task_id"]); carrier = str(case["carrier"])
            task = tasks[task_id]
            delivered: dict[str, int] = {}
            def repair_sink(observed_id: str, tensor: torch.Tensor, _metadata: Any) -> None:
                if observed_id != task_id or delivered:
                    raise RuntimeError("repair endpoint identity drifted")
                reference_value = align_reference_to_candidate(references[observed_id], tensor)
                before = tensor.detach().clone(); tensor.copy_(reference_value)
                delivered["changed"] = int(torch.count_nonzero(before != reference_value))
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            repair = SameDtypeSemanticCandidateObserver(
                modules=modules, campaign_rows=campaign["rows"],
                inventory_rows=inventory["runtime_call_audit"]["rows"],
                task_rows=[task], sink=repair_sink,
                include_unresolved_tasks=True)
            with repair:
                repair_loss = candidate(values); repair_loss.backward()
            torch.cuda.synchronize(device); repair.validate()
            repair_gradient = parameters[carrier].grad.detach().float().cpu().clone()
            gradient_delta = baseline_gradients[carrier] - repair_gradient
            update_delta = -LR * gradient_delta
            gradient_delta_flat = gradient_delta.double().reshape(-1)
            repair_gradient_flat = repair_gradient.double().reshape(-1)
            reference_relative[task_id].append({
                "condition_id": state_id,
                "partition": partition,
                "error_reference_dot": float(torch.dot(
                    gradient_delta_flat, repair_gradient_flat
                )),
                "error_energy": float(torch.dot(
                    gradient_delta_flat, gradient_delta_flat
                )),
                "reference_energy": float(torch.dot(
                    repair_gradient_flat, repair_gradient_flat
                )),
            })
            case_id = str(case.get("case_id", task_id.replace(":", "_")))
            if args.engineering_reach_only:
                reach_rows[task_id].append({
                    "state_id": state_id,
                    "endpoint_changed_coordinates": delivered["changed"],
                    "local_error_energy": float(torch.sum(observed_locals[task_id].double() ** 2)),
                    "carrier_gradient_error_energy": float(torch.sum(gradient_delta.double() ** 2)),
                    "carrier_reached": bool(torch.count_nonzero(gradient_delta)),
                })
            else:
                local_delta = observed_locals[task_id]
                direct_gradient_output = (
                    case.get("reference_method") == "EXTERNAL_FP32_RECOMPUTE"
                    and local_delta.shape == gradient_delta.shape
                    and torch.equal(local_delta, gradient_delta)
                )
                local_row = spool_vector(
                    args.spool_dir, case_id, "local", partition, state_id, local_delta)
                rows[task_id]["LOCAL_ENDPOINT"].append(local_row)
                if direct_gradient_output:
                    rows[task_id]["PARAMETER_GRADIENT"].append({**local_row, "scale": 1.0})
                    rows[task_id]["EFFECTIVE_UPDATE"].append({**local_row, "scale": -LR})
                else:
                    rows[task_id]["PARAMETER_GRADIENT"].append(spool_vector(
                        args.spool_dir, case_id, "gradient", partition, state_id, gradient_delta))
                    rows[task_id]["EFFECTIVE_UPDATE"].append(spool_vector(
                        args.spool_dir, case_id, "update", partition, state_id, update_delta))
            metadata[task_id].append({
                "state_id": state_id, "partition": partition,
                "common_state": common_state(state, seed, weights_digest),
                "endpoint_changed_coordinates": delivered["changed"],
            })
        print(json.dumps({"event": "FORMATION_STATE_COMPLETE", "state": state_id,
                          "cases": len(cases)}), flush=True)
        del values, baseline_gradients, references
        torch.cuda.empty_cache()

    if args.engineering_reach_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        reach_payload = {
            "schema": "kernel-analyzer-backward-equivalence-dynamic-reach-v1",
            "status": "ENGINEERING_ONLY_NO_BIAS_VERDICT",
            "architecture": args.architecture,
            "states": args.states,
            "cases": [{
                "case_id": str(case.get("case_id", str(case["task_id"]).replace(":", "_"))),
                "task_id": str(case["task_id"]), "carrier": str(case["carrier"]),
                "records": reach_rows[str(case["task_id"])],
                "carrier_reached_any_state": any(
                    row["carrier_reached"] for row in reach_rows[str(case["task_id"])]),
            } for case in cases],
        }
        target = args.output_dir / "engineering_reach.json"
        target.write_text(json.dumps(reach_payload, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(args.spool_dir)
        print(json.dumps({
            "event": "ENGINEERING_REACH_COMPLETE", "output": str(target),
            "cases": len(cases),
            "reached": sum(row["carrier_reached_any_state"] for row in reach_payload["cases"]),
        }), flush=True)
        return

    if args.screening_gram:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        policy = FormationPolicy(min_states=args.states, bootstrap_samples=2000)
        screening_cases = []
        for case in cases:
            task_id = str(case["task_id"])
            layers = {}
            for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"):
                certificate = summarize_streamed_state_vector_files(
                    rows[task_id][layer], layer=layer, partition="screening",
                    policy=policy)
                layers[layer] = certificate.as_dict()
            screening_cases.append({
                "case_id": str(case.get("case_id", task_id.replace(":", "_"))),
                "task_id": task_id, "carrier": str(case["carrier"]),
                "layers": layers,
            })
        payload = {
            "schema": "kernel-analyzer-backward-equivalence-screening-gram-v1",
            "status": "SCREENING_ONLY_NO_SCIENTIFIC_VERDICT",
            "architecture": args.architecture, "state_count": args.states,
            "selection_rule": (
                "Shortlist by gradient cross-state geometry without requiring local bias; "
                "all promoted cases require independent 16+16 confirmation."
            ),
            "cases": screening_cases,
        }
        target = args.output_dir / "screening_gram.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(args.spool_dir)
        print(json.dumps({"event": "SCREENING_GRAM_COMPLETE", "output": str(target),
                          "cases": len(screening_cases)}), flush=True)
        return

    population_size = args.states // 2 if args.extended_confirmation else 16
    policy = FormationPolicy(min_states=population_size, bootstrap_samples=2000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        task_id = str(case["task_id"]); case_id = str(case.get("case_id", task_id.replace(":", "_")))
        populations: dict[str, dict[str, Any]] = {"calibration": {}, "confirmation": {}}
        reference_relative_certificates: dict[str, dict[str, Any]] = {}
        for partition in populations:
            for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"):
                selected = [row for row in rows[task_id][layer] if partition in Path(row["path"]).parts]
                certificate = summarize_streamed_state_vector_files(
                    selected, layer=layer, partition=partition, policy=policy)
                certificate_row = certificate.as_dict()
                status = certificate.status
                # A declared parameter carrier that is not reached by this
                # endpoint repair is a binding miss, not evidence of centered
                # training error.  Keep the complete Gram for diagnosis but
                # fail closed before formation-stage inference.
                if (layer != "LOCAL_ENDPOINT" and
                        certificate.average_state_energy <= policy.energy_floor):
                    status = "UNRESOLVED_ZERO_CARRIER_EFFECT"
                    certificate_row["status"] = status
                    certificate_row["unresolved_reason"] = (
                        "declared carrier received no measurable repair-induced effect")
                populations[partition][layer] = certificate_row
                populations[partition][layer + "_status"] = status
            relative_rows = [
                row for row in reference_relative[task_id]
                if row["partition"] == partition
            ]
            try:
                relative_certificate = certify_reference_relative([
                    ReferenceRelativeObservation(
                        condition_id=str(row["condition_id"]),
                        error_reference_dot=float(row["error_reference_dot"]),
                        error_energy=float(row["error_energy"]),
                        reference_energy=float(row["reference_energy"]),
                    )
                    for row in relative_rows
                ])
                reference_relative_certificates[partition] = {
                    **relative_certificate.as_dict(),
                    "rows": relative_rows,
                }
            except (ValueError, ArithmeticError) as exc:
                reference_relative_certificates[partition] = {
                    "status": "REFERENCE_RELATIVE_UNRESOLVED",
                    "reason": str(exc),
                    "rows": relative_rows,
                }
        confirmation = [populations["confirmation"][layer + "_status"]
                        for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")]
        first_observed = next((layer for layer, status in zip(
            ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"), confirmation)
            if status == "BIASED"), None)
        first_confirmed = None
        prior_centered = True
        for layer, status in zip(("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"), confirmation):
            if first_confirmed is None and prior_centered and status == "BIASED":
                first_confirmed = layer
            if status != "CENTERED":
                prior_centered = False
        payload = {
            "schema": "kernel-analyzer-bias-formation-certificate-v2_1",
            "case_id": case_id, "status": "COMPLETE",
            "measurement_kind": "candidate_repair_ground_truth",
            "uses_candidate_measurements": True, "uses_historical_verdicts": False,
            "verdict_blind": True, "trajectory_drift_in_formation": False,
            "state_split": {"calibration_count": population_size,
                            "confirmation_count": population_size,
                            "both_open_loop_common_state": True, "disjoint": True},
            "policy": policy.as_dict(), "populations": populations,
            "reference_relative_parameter_gradient": (
                reference_relative_certificates
            ),
            "first_observed_biased_stage": first_observed,
            "first_confirmed_bias_stage": first_confirmed,
            "formation_point": "CONFIRMED" if first_confirmed else "UNRESOLVED",
            "rows": metadata[task_id],
            "binding": {
                "task_id": task_id, "exact_aot_endpoint_id": tasks[task_id]["exact_aot_endpoint_id"],
                "carrier_parameter": case["carrier"],
                "frozen_release_id": args.release_dir.name,
                "frozen_input_bank": str(args.input_bank),
                "measurement_state_bank": str(args.state_bank or args.input_bank),
            },
            "capture_provenance": {"runner": "scripts/capture_bound_endpoint_bias_formation_v21.py",
                                   "device": args.device, "raw_vectors_retained": False,
                                   "optimizer": "STATELESS_SGD_FP32_MASTER", "learning_rate": LR},
        }
        target = args.output_dir / f"{case_id}.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "FORMATION_CASE_COMPLETE", "case": case_id,
                          "first_confirmed_bias_stage": first_confirmed,
                          "output": str(target)}), flush=True)
    shutil.rmtree(args.spool_dir)


if __name__ == "__main__":
    main()
