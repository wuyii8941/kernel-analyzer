#!/usr/bin/env python3
"""Audit several full-coordinate T1 survivors with independent T2/T3 repairs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

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
from scripts.run_same_dtype_semantic_oracle import load, write
from scripts.run_targeted_causal_repair import clone_gradients, gradient_delta_summary
from scripts.run_targeted_full_coordinate import validate_release
from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver

ARCHITECTURES = {"qwen": "qwen3_1p7b", "mamba": "mamba_130m",
                 "phi": "phi4_mini_3p8b", "deepseek8": "deepseek_r1_0528_qwen3_8b"}


def canonical(value: object) -> str:
    """Hash nested result metadata, including observer tensor fields.

    The observer's ``metadata`` is allowed to retain typed tensor witnesses.
    Those are useful for the in-memory intervention but are not JSON-native;
    the old canonicalizer crashed at the first repair-repeat comparison.  Hash
    tensor bytes (plus shape/dtype/device) instead of silently dropping them.
    """
    def encode(item: object) -> object:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {
                "__tensor_sha256__": tensor_digest(tensor),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        if isinstance(item, dict):
            return {str(key): encode(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(value) for value in item]
        return item
    return hashlib.sha256(
        json.dumps(encode(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def json_safe(value: object) -> object:
    """Materialize observer metadata without serializing raw tensors."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor_sha256__": tensor_digest(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def summary(value: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "coordinates": value.numel(), "sha256": tensor_digest(value)}


def clone_selected_gradients(
    model: torch.nn.Module, names: set[str],
) -> dict[str, torch.Tensor | None]:
    parameters = dict(model.named_parameters())
    return {
        name: None if parameters[name].grad is None else parameters[name].grad.detach().cpu().clone()
        for name in sorted(names)
    }


def selected_gradient_delta_summary(
    model: torch.nn.Module, baseline: dict[str, torch.Tensor | None], names: set[str],
) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    rows = []; squared_l2 = 0.0; changed_elements = 0
    for name in sorted(names):
        parameter = parameters[name]; left = baseline[name]
        right = None if parameter.grad is None else parameter.grad.detach().cpu()
        if left is None or right is None:
            if left is not None or right is not None:
                rows.append({"parameter": name, "status": "PRESENCE_CHANGED",
                             "parameter_numel": int(parameter.numel())})
            continue
        if torch.equal(left, right):
            continue
        delta = right.float() - left.float(); count = int(torch.count_nonzero(delta))
        norm2 = float(torch.sum(delta.double().square()))
        squared_l2 += norm2; changed_elements += count
        rows.append({"parameter": name, "parameter_numel": int(parameter.numel()),
                     "parameter_shape": list(parameter.shape), "parameter_dtype": str(parameter.dtype),
                     "changed_elements": count, "l2": norm2 ** 0.5,
                     "max_abs": float(delta.abs().max()), "signed_sum": float(delta.double().sum())})
    return {"changed_parameter_count": len(rows), "changed_elements": changed_elements,
            "global_l2": squared_l2 ** 0.5, "parameters": rows}


def qwen_local_carrier_candidates(
    task_id: str, campaign_by_region: dict[str, dict], parameter_names: set[str],
    exact_aot_endpoint_id: str | None = None,
) -> list[str]:
    region = ":".join(task_id.split(":")[:2])
    # External-library regions (MM/BMM) can be exact semantic endpoints while
    # not having a Triton campaign row.  T2 still proves causal reach from its
    # exhaustive gradient scan; topology-aware carrier selection for those
    # endpoints is deferred to T3 instead of crashing or dropping the case.
    campaign_row = campaign_by_region.get(region)
    nodes = [] if campaign_row is None else [str(value) for value in campaign_row["source_nodes"]]
    candidates = []
    # Attention score BMMs are external-library regions and consequently have
    # no Triton campaign row.  Their exact AOT names advance by two per layer:
    # bmm, bmm_2, bmm_4, ... .  Use that proved origin only to propose o_proj;
    # the repaired gradient below remains the deciding numerical evidence.
    external_attention = re.fullmatch(
        r"forward:graph\d+:bmm(?:_(\d+))?", exact_aot_endpoint_id or ""
    )
    if external_attention:
        ordinal = int(external_attention.group(1) or 0)
        if ordinal % 2 == 0:
            candidates.append(
                f"model.layers.{ordinal // 2}.self_attn.o_proj.weight"
            )
    for prefix, parameter in (("q_embed", "q_proj"), ("k_embed", "k_proj")):
        layers = [int(match.group(1) or 0) for value in nodes
                  if (match := re.fullmatch(prefix + r"(?:_(\d+))?", value))]
        if layers:
            layer = max(layers)
            candidates.append(f"model.layers.{layer}.self_attn.{parameter}.weight")
            norm = "q_norm" if parameter == "q_proj" else "k_norm"
            candidates.append(f"model.layers.{layer}.self_attn.{norm}.weight")
    rsqrts = [int(match.group(1)) for value in nodes
              if (match := re.fullmatch(r"rsqrt_(\d+)", value))]
    if rsqrts:
        number = max(rsqrts)
        if number % 4 == 3:
            candidates.append(f"model.layers.{(number - 3) // 4}.self_attn.q_proj.weight")
        elif number % 4 == 0:
            candidates.append(f"model.layers.{number // 4 - 1}.mlp.gate_proj.weight")
    attention = [int(match.group(1)) for value in nodes
                 if (match := re.fullmatch(r"attn_output_(\d+)", value))]
    if attention and max(attention) % 4 == 0:
        layer = max(attention) // 4
        candidates.extend([
            f"model.layers.{layer}.self_attn.o_proj.weight",
            f"model.layers.{layer}.post_attention_layernorm.weight",
            f"model.layers.{layer + 1}.input_layernorm.weight",
        ])
    # DeepSeek's eager graph names the three score/probability values four
    # apart per layer (0/1/2, 4/5/6, ...).  A repaired score/probability can
    # therefore be tested directly at that layer's output projection before
    # falling back to an all-parameter scan.
    attention_weights = [int(match.group(1) or 0) for value in nodes
                         if (match := re.fullmatch(r"attn_weights(?:_(\d+))?", value))]
    if attention_weights:
        layer = max(attention_weights) // 4
        candidates.append(f"model.layers.{layer}.self_attn.o_proj.weight")
    # Preserve topology preference while removing aliases/duplicates.  Every
    # returned carrier is still checked against the actual repaired gradient;
    # topology is only a cheap search order, never evidence by itself.
    return list(dict.fromkeys(name for name in candidates if name in parameter_names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--full-coordinate-t1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--repeat-check-states", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument(
        "--reuse-reference-cache", action="store_true",
        help="reuse a validated TorchInductor reference cache; cache reuse changes compilation time only",
    )
    args = parser.parse_args()
    if args.repeat != 2 or args.states < 1:
        raise ValueError("T2/T3 requires two repeats and at least one state")
    if not 1 <= args.repeat_check_states <= args.states:
        raise ValueError("repeat-check states must be within the T2 state count")

    t1 = load(args.full_coordinate_t1)
    passed = [str(row["task_id"]) for row in t1["rows"]
              if row["verdict"] == "DIRECTIONAL_OPTIMIZATION_BIAS"]
    selected = args.task_id or passed
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(passed):
        raise RuntimeError("batch contains absent, duplicate, or nonpassing T1 task IDs")

    release = args.release_dir
    capture = json.loads((release / "capture.json").read_text())
    campaign = load(release / "campaign.json.gz")
    campaign_by_region = {str(row["region_id"]): row for row in campaign["rows"]}
    inventory = load(release / "inventory.json.gz")
    plan = load(release / "same_dtype_tasks.json.gz")
    by_task = {str(row["task_id"]): row for row in plan["rows"]}
    tasks = [by_task[value] for value in selected]
    endpoints = {str(row["exact_aot_endpoint_id"]) for row in tasks}
    cuts = [row for row in plan["reference_cut_tasks"]
            if str(row["task_id"]).removeprefix("same-dtype:") in endpoints]
    if len(cuts) != len(endpoints):
        raise RuntimeError("batch reference-cut denominator is incomplete")

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.states or file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match frozen release")
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor",
                              fullgraph=not args.allow_graph_breaks, dynamic=False)
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    args.reference_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.reference_cache_dir.resolve())
    active: dict[str, Any] = {"sink": None}
    def dispatch(cut: Any, outputs: tuple[Any, ...]) -> None:
        if active["sink"] is not None:
            active["sink"](cut, outputs)
    ref_capture = AOTForwardBackwardCapture(reference_cut_tasks=cuts, reference_value_sink=dispatch)
    # The reference graph is still captured with the exact endpoint cuts.  A
    # cache hit only reuses generated code after TorchInductor validates its
    # graph key; it does not reuse tensor values or candidate observations.
    # The original fail-closed mode remains the default for releases that do
    # not opt into an existing, cell-specific cache.
    cache_patch = {} if args.reuse_reference_cache else {"force_disable_caches": True}
    with inductor_config.patch(cache_patch):
        reference = torch.compile(LossStep(model), backend=ref_capture.inductor_partition_backend(),
                                  fullgraph=not args.allow_graph_breaks, dynamic=False)
        model.zero_grad(set_to_none=True)
        warm_loss = reference(warm); ref_capture.bind_user_outputs(warm_loss)
        warm_loss.register_hook(ref_capture.bind_user_cotangent); warm_loss.backward()
        torch.cuda.synchronize(device)
    if not all(ref_capture.as_dict()["reference_cut_runtime"]["gates"].values()):
        raise RuntimeError("batch reference cut failed")

    task_states: dict[str, list[dict[str, Any]]] = {value: [] for value in selected}
    selected_carriers: dict[str, str] = {}
    carrier_discovery: dict[str, list[dict[str, Any]]] = {}
    carrier_discovery_modes: dict[str, str] = {}
    terminal_failures: dict[str, dict[str, Any]] = {}
    parameter_names = {name for name, _parameter in model.named_parameters()}
    for state_index, state in enumerate(states[:args.states]):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 29000 + state_index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values); baseline_loss.backward(); torch.cuda.synchronize(device)
        baseline_identity = {
            "loss": tensor_digest(baseline_loss),
            "gradient_identity_basis": "zero_delta_against_cloned_baseline",
        }
        if state_index == 0:
            baseline_gradients = clone_gradients(model)
        else:
            active_tasks = set(selected) - set(terminal_failures)
            missing_carriers = active_tasks - set(selected_carriers)
            if missing_carriers:
                # A task that reached a terminal T2 failure during state 0
                # may not have a carrier to reuse.  Preserve the endpoint
                # failure and keep the rest of the batch running instead of
                # aborting the queue on a bookkeeping assertion.
                for task_id in sorted(missing_carriers):
                    terminal_failures[task_id] = {
                        "reason": "T2_CARRIER_NOT_FROZEN_AFTER_STATE_ZERO",
                        "state_id": state_id,
                    }
                active_tasks -= missing_carriers
            baseline_gradients = clone_selected_gradients(model, set(selected_carriers.values()))

        refs: dict[str, torch.Tensor] = {}
        def ref_sink(cut: Any, outputs: tuple[Any, ...]) -> None:
            endpoint = str(cut.task_id).removeprefix("same-dtype:")
            tensors = [value for value in outputs if isinstance(value, torch.Tensor)]
            if len(tensors) != 1 or endpoint in refs:
                raise RuntimeError("batch reference endpoint changed")
            refs[endpoint] = tensors[0].detach().clone()
        active["sink"] = ref_sink
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        ref_loss = reference(values); ref_capture.bind_user_outputs(ref_loss)
        ref_loss.register_hook(ref_capture.bind_user_cotangent); ref_loss.backward()
        torch.cuda.synchronize(device); active["sink"] = None
        if set(refs) != endpoints:
            raise RuntimeError("batch reference values are incomplete")

        for task in tasks:
            task_id = str(task["task_id"]); endpoint = str(task["exact_aot_endpoint_id"])
            if task_id in terminal_failures:
                continue
            repeats = []; frozen = None
            terminal = False
            repair_repeats = args.repeat if state_index < args.repeat_check_states else 1
            for repeat in range(repair_repeats):
                arms = {}
                # One exact sham is sufficient to prove that the intervention
                # wrapper is inert.  Repeat stability applies to the repaired
                # numerical arm, so rerunning the identical sham adds a full
                # model step without strengthening any Flash-style gate.
                modes = (
                    ("SHAM", "REPAIR")
                    if state_index < args.repeat_check_states and repeat == 0
                    else ("REPAIR",)
                )
                for mode in modes:
                    delivered: dict[str, Any] = {}
                    def sink(observed_id: str, tensor: torch.Tensor, metadata: Any) -> None:
                        if observed_id != task_id or delivered:
                            raise RuntimeError("batch candidate endpoint identity changed")
                        before = tensor.detach().clone(); ref = refs[endpoint]
                        if before.shape != ref.shape or before.dtype != ref.dtype:
                            raise RuntimeError("batch repair metadata mismatch")
                        if mode == "REPAIR": tensor.copy_(ref)
                        else: tensor.copy_(before)
                        delivered.update(before=summary(before), reference=summary(ref),
                                         delivered=summary(tensor), metadata=dict(metadata),
                                         changed_coordinates=int(torch.count_nonzero(before != ref)))
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    model.zero_grad(set_to_none=True)
                    observer = SameDtypeSemanticCandidateObserver(
                        modules=modules, campaign_rows=campaign["rows"],
                        inventory_rows=inventory["runtime_call_audit"]["rows"],
                        task_rows=[task], sink=sink,
                    )
                    with observer:
                        loss = candidate(values); loss.backward()
                    torch.cuda.synchronize(device); observer.validate()
                    if state_index == 0 and task_id not in selected_carriers:
                        local_candidates = (
                            qwen_local_carrier_candidates(
                                task_id, campaign_by_region, parameter_names,
                                str(task["exact_aot_endpoint_id"]),
                            )
                            if args.architecture in {"qwen", "deepseek8"}
                            else []
                        )
                        # T2 asks whether the repaired endpoint reaches a real
                        # parameter carrier, not for a fresh census of every
                        # model parameter per candidate.  Check the AOT-local
                        # carriers first and retain exhaustive discovery as a
                        # fail-closed fallback when topology cannot resolve it.
                        used_local_search = mode == "REPAIR" and bool(local_candidates)
                        gradient_delta = (
                            selected_gradient_delta_summary(
                                model, baseline_gradients, set(local_candidates)
                            )
                            if used_local_search
                            else gradient_delta_summary(model, baseline_gradients)
                        )
                        local_search_hit = bool(gradient_delta["parameters"])
                        if used_local_search and not local_search_hit:
                            gradient_delta = gradient_delta_summary(model, baseline_gradients)
                        if mode == "REPAIR":
                            carrier_discovery[task_id] = list(gradient_delta["parameters"])
                            carrier_discovery_modes[task_id] = (
                                "AOT_LOCAL_NUMERICAL_GRADIENT_DELTA"
                                if used_local_search and local_search_hit
                                else "EXHAUSTIVE_PARAMETER_GRADIENT_DELTA"
                            )
                    else:
                        carrier = selected_carriers[task_id]
                        gradient_delta = selected_gradient_delta_summary(
                            model, baseline_gradients, {carrier})
                    if mode == "REPAIR" and task_id not in selected_carriers:
                        reachable = [row for row in gradient_delta["parameters"]
                                     if row.get("status") != "PRESENCE_CHANGED"]
                        if not reachable:
                            terminal_failures[task_id] = {
                                "reason": "REPAIR_REACHES_NO_CONCRETE_PARAMETER_CARRIER",
                                "state_id": state_id,
                                "endpoint_changed_coordinates": int(
                                    delivered["changed_coordinates"]),
                            }
                            terminal = True
                            arms[mode] = {
                                "identity": {
                                    "loss": tensor_digest(loss),
                                    "gradient_identity_basis": "delta_against_cloned_baseline",
                                },
                                "endpoint": delivered,
                                "gradient_delta": gradient_delta,
                            }
                            break
                        local = next(
                            (name for name in local_candidates
                             if any(str(row["parameter"]) == name for row in reachable)),
                            None,
                        )
                        if local is not None:
                            chosen = local
                        else:
                            chosen_row = min(
                                reachable,
                                key=lambda row: (int(row["parameter_numel"]),
                                                 str(row["parameter"])),
                            )
                            chosen = str(chosen_row["parameter"])
                        selected_carriers[task_id] = chosen
                        # Stability compares like with like.  The exhaustive
                        # discovery scan selects the carrier but is not itself
                        # compared with the carrier-only repeat summary.
                        gradient_delta = selected_gradient_delta_summary(
                            model, baseline_gradients, {selected_carriers[task_id]})
                    arms[mode] = {
                        "identity": {
                            "loss": tensor_digest(loss),
                            "gradient_identity_basis": "delta_against_cloned_baseline",
                        },
                        "endpoint": delivered,
                        "gradient_delta": gradient_delta,
                    }
                if terminal:
                    break
                if state_index < args.repeat_check_states and repeat == 0 and (
                    arms["SHAM"]["identity"]["loss"] != baseline_identity["loss"]
                    or arms["SHAM"]["gradient_delta"]["changed_parameter_count"] != 0
                ):
                    raise RuntimeError("batch matched sham perturbed full step")
                signature = canonical(arms["REPAIR"])
                if frozen is None: frozen = signature
                elif frozen != signature:
                    # A T2 repeat mismatch is an observation-instability
                    # rejection, not a queue-level infrastructure failure.
                    # Preserve it in the batch artifact so the remaining
                    # endpoints can continue and the funnel remains fail-closed.
                    terminal_failures[task_id] = {
                        "reason": "REPAIR_REPEAT_CHANGED",
                        "state_id": state_id,
                        "repeat": repeat,
                        "first_signature": frozen,
                        "repeat_signature": signature,
                    }
                    terminal = True
                    break
                repeats.append({"repeat": repeat, "arms": arms})
            if terminal:
                task_states[task_id].append({
                    "state_id": state_id,
                    "terminal_failure": terminal_failures[task_id],
                    "repeats": [{"repeat": 0, "arms": arms}],
                    "repair_repeats": 1,
                    "repair_changed_endpoint": bool(
                        arms["REPAIR"]["endpoint"]["changed_coordinates"]),
                    "repair_reached_parameter_gradients": False,
                })
                continue
            repair = repeats[0]["arms"]["REPAIR"]
            task_states[task_id].append({
                "state_id": state_id, "baseline_identity": baseline_identity,
                "repeats": repeats,
                "repair_repeats": repair_repeats,
                "frozen_carrier": selected_carriers[task_id],
                "repair_changed_endpoint": repair["endpoint"]["changed_coordinates"] > 0,
                "repair_reached_parameter_gradients": repair["gradient_delta"]["changed_parameter_count"] > 0,
            })
        write(args.output.with_name("." + args.output.name + ".partial"), json_safe({
            "schema": "kernel-analyzer-same-dtype-causal-repair-batch-checkpoint-v1",
            "status": "PARTIAL_FAIL_CLOSED", "task_ids": selected,
            "states_complete": state_index + 1, "task_states": task_states,
        }))
        del baseline_gradients, refs, values
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id,
                          "tasks": len(selected)}), flush=True)

    rows = []
    for task_id in selected:
        state_rows = task_states[task_id]
        if task_id in terminal_failures:
            gates = {
                "exact_wrapper_identity": True,
                "same_dtype_reference": True,
                "exact_aot_semantic_endpoint": True,
                "pilot_matched_sham_exact": True,
                "pilot_two_repairs_stable": False,
                "remaining_states_single_repair": False,
                "repair_nonnull_every_state": bool(
                    state_rows[0]["repair_changed_endpoint"]),
                "repair_reaches_parameter_gradients_every_state": False,
            }
            rows.append({
                "task_id": task_id,
                "states": state_rows,
                "gates": gates,
                "carrier_discovery_first_state": carrier_discovery.get(task_id, []),
                "carrier_discovery_mode": carrier_discovery_modes.get(task_id),
                "terminal_failure": terminal_failures[task_id],
                "causal_t2_positive": False,
                "causal_t2_t3_positive": False,
                "disposition": "REJECT_T2_NO_PARAMETER_CARRIER",
            })
            continue
        gates = {
            "exact_wrapper_identity": True, "same_dtype_reference": True,
            "exact_aot_semantic_endpoint": True,
            "pilot_matched_sham_exact": all(
                "SHAM" in row["repeats"][0]["arms"]
                for row in state_rows[:args.repeat_check_states]
            ),
            "pilot_two_repairs_stable": all(
                row["repair_repeats"] == 2
                for row in state_rows[:args.repeat_check_states]
            ),
            "remaining_states_single_repair": all(
                row["repair_repeats"] == 1
                for row in state_rows[args.repeat_check_states:]
            ),
            "repair_nonnull_every_state": all(row["repair_changed_endpoint"] for row in state_rows),
            "repair_reaches_parameter_gradients_every_state": all(
                row["repair_reached_parameter_gradients"] for row in state_rows),
        }
        rows.append({"task_id": task_id, "states": state_rows, "gates": gates,
                     "carrier_discovery_first_state": carrier_discovery.get(task_id, []),
                     "carrier_discovery_mode": carrier_discovery_modes.get(task_id),
                     "causal_t2_positive": all(gates.values()),
                     "causal_t2_t3_positive": all(gates.values()),  # legacy reader compatibility
                     "disposition": ("PASS_T2_CAUSAL_REACH" if all(gates.values())
                                     else "REJECT_T2_CAUSAL_REACH")})
    payload = {
        "schema": "kernel-analyzer-same-dtype-causal-repair-batch-v2",
        "status": "COMPLETE_T2_CAUSAL_REACH_BATCH",
        "architecture": ARCHITECTURES[args.architecture],
        "states": args.states, "repeats": args.repeat, "task_count": len(selected),
        "full_coordinate_t1_sha256": t1["result_sha256"],
        "release_capture_sha256": capture["result_sha256"], "rows": rows,
        "counts": dict(__import__("collections").Counter(row["disposition"] for row in rows)),
        "claim_boundary": (
            "Independent exact-endpoint sham/repair T2 and real-parameter reachability only. "
            "Cross-state complete-carrier T3 and paired accumulation T4 remain required."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    write(args.output, json_safe(payload))
    args.output.with_name("." + args.output.name + ".partial").unlink(missing_ok=True)
    print(json.dumps({"status": payload["status"], "counts": payload["counts"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
