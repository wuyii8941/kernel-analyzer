#!/usr/bin/env python3
"""Recheck Gemma RMS normalization with an FP32 reduction-order intervention.

The candidate is the real generated Gemma training graph.  The only changed
choice between the two reference conditions is the feature order used by the
FP32 square-sum reference.  This runner writes compact scalar statistics and
does not overwrite the historical capture.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "results/property/training_numerical_analysis_v2/recovery/gemma_original_trainability_scope/runtime_release"
DEFAULT_INPUT_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"
DEFAULT_STATE_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory32.json"
DEFAULT_SELECTION = ROOT / "results/property/numerical_coverage_v1/gemma_rms_forward_capture_selection_v1.json"
DEFAULT_MANIFEST = ROOT / "results/property/numerical_coverage_v1/gemma_rms_forward_registered_scan_v1.json"
DEFAULT_OUTPUT = ROOT / "results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/result.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_INPUT_BANK)
    parser.add_argument("--state-bank", type=Path, default=DEFAULT_STATE_BANK)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _summary(effects: list[Any], references: list[Any], *, torch: Any) -> dict[str, Any]:
    effect_arrays = [value.detach().float().cpu().reshape(-1) for value in effects]
    reference_arrays = [value.detach().float().cpu().reshape(-1) for value in references]
    x_values = [float(torch.dot(effect, effect)) for effect in effect_arrays]
    b_values = [float(torch.dot(reference, reference)) for reference in reference_arrays]
    a_values = [float(torch.dot(effect, reference)) for effect, reference in zip(effect_arrays, reference_arrays)]
    total_x = math.fsum(x_values)
    total_b = math.fsum(b_values)
    mean_effect = torch.stack(effect_arrays).mean(dim=0)
    mean_b = total_b / len(b_values)
    calibration = effect_arrays[: len(effect_arrays) // 2]
    confirmation = effect_arrays[len(effect_arrays) // 2 :]
    calibration_mean = torch.stack(calibration).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(calibration_mean))
    direction_interval = None
    if direction_norm > 1e-30 and confirmation:
        direction = calibration_mean / direction_norm
        projections = [float(torch.dot(effect, direction)) for effect in confirmation]
        projection_mean = math.fsum(projections) / len(projections)
        centered = [value - projection_mean for value in projections]
        standard_error = (
            math.sqrt(math.fsum(value * value for value in centered) / (len(projections) - 1) / len(projections))
            if len(projections) > 1 else None
        )
        direction_interval = {
            "mean": projection_mean,
            "values": projections,
            "standard_error": standard_error,
            "interval": ([projection_mean - 2.571 * standard_error,
                           projection_mean + 2.571 * standard_error]
                          if standard_error is not None else None),
            "critical_value_note": "descriptive t critical value for df=15; fixed-suite, not population inference",
        }
    return {
        "state_count": len(effects),
        "total_effect_rms": math.sqrt(total_x / total_b) if total_b > 0 else None,
        "mean_effect_over_repair_rms": (
            float(torch.linalg.vector_norm(mean_effect)) / math.sqrt(mean_b)
            if mean_b > 0 else None
        ),
        "aligned_ratio_of_sums": math.fsum(a_values) / total_b if total_b > 0 else None,
        "direction": {
            "calibration_norm": direction_norm,
            "confirmation_projection": direction_interval,
            "status": "IDENTIFIED" if direction_interval is not None else "NOT_IDENTIFIABLE",
        },
    }


def main() -> None:
    args = _args()
    if args.steps < 4 or args.steps % 2:
        raise SystemExit("--steps must be an even integer >= 4")
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    import torch
    from torch._inductor.codecache import PyCodeCache

    from kernel_analyzer.source_reference_registry import get_reference
    from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime, text_step_values
    from scripts.run_frozen_candidate_fp32_screen import wrapper_modules
    from scripts.run_generated_fp32_screen import load_model
    from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver

    selection = _read_json(args.selection)
    manifest = _read_json(args.manifest)
    contracts_by_symbol = {}
    for source in manifest["sources"]:
        for row in source["rows"]:
            if row.get("status") == "SOURCE_CHECKED":
                contracts_by_symbol[str(row["symbol"])] = dict(row["contract"])
    cases = [
        case for case in selection["cases"]
        if str(case["reference_contract_symbol"]) in contracts_by_symbol
    ]
    if not cases:
        raise RuntimeError("RMS selection contains no source-checked cases")
    plan = _read_gzip(args.release / "same_dtype_tasks.json.gz")
    task_rows = {str(row["task_id"]): row for row in plan["rows"]}
    campaign = _read_gzip(args.release / "campaign.json.gz")
    inventory = _read_gzip(args.release / "inventory.json.gz")
    bank = _read_json(args.input_bank)
    states = _read_json(args.state_bank)["states"][: args.steps]
    warm_states = [state for state in bank["states"] if state.get("role") == "ENGINEERING"]
    if not warm_states:
        raise RuntimeError("input bank has no ENGINEERING warm-up state")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("gemma4", args.model, device)
    parameters = dict(model.named_parameters())
    carriers = {str(case["carrier"]) for case in cases}
    if not carriers <= set(parameters):
        raise RuntimeError("declared carrier is absent from Gemma model")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in carriers)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm_values = text_step_values(warm_states[0], device)
    model.zero_grad(set_to_none=True)
    candidate(*warm_values).backward()
    torch.cuda.synchronize(device)
    modules = [module for module, _phase in wrapper_modules(list(PyCodeCache.modules)[start:])]

    specification = get_reference("RMS_FORWARD_NORMALIZED")
    variants = ("FP32_NATIVE", "FP32_REVERSE_FEATURE_ORDER")
    runtime_campaign = campaign["rows"]
    runtime_inventory = inventory["runtime_call_audit"]["rows"]

    def run_state(
        state: dict[str, Any], *, variant: str, repair: bool,
        selected_task: dict[str, Any], expected_task_id: str,
    ) -> dict[str, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}

        def sink(task_id: str, tensor: torch.Tensor, metadata: dict[str, Any]) -> None:
            if task_id in captured:
                raise RuntimeError("RMS endpoint delivered more than once")
            candidate_value = tensor.detach().clone()
            if repair:
                symbol = str(metadata["symbol"])
                contract = contracts_by_symbol[symbol]
                reference = specification.evaluate(metadata, tensor, contract, variant=variant)
                tensor.copy_(reference)
                captured[task_id] = reference.detach().clone()
            else:
                captured[task_id] = candidate_value

        observer = SameDtypeSemanticCandidateObserver(
            modules=modules,
            campaign_rows=runtime_campaign,
            inventory_rows=runtime_inventory,
            task_rows=[selected_task],
            sink=sink,
            include_unresolved_tasks=True,
        )
        values = text_step_values(state, device)
        model.zero_grad(set_to_none=True)
        with observer:
            loss = candidate(*values)
            loss.backward()
        torch.cuda.synchronize(device)
        observer.validate()
        if set(captured) != {expected_task_id}:
            raise RuntimeError("RMS endpoint capture is incomplete")
        return captured

    # Run each endpoint in a separate observer.  The two selected calls share a
    # carrier parameter, so observing both at once would make a per-endpoint
    # gradient attribution invalid.
    outputs_by_case: dict[str, dict[str, dict[str, dict[str, list[Any]]]]] = {}
    for case in cases:
        task_id = str(case["task_id"])
        carrier = str(case["carrier"])
        selected_task = task_rows[task_id]
        outputs: dict[str, dict[str, dict[str, list[Any]]]] = {
            variant: {
                task_id: {
                    "effects": [],
                    "references": [],
                    "candidate_outputs": [],
                    "gradient_effects": [],
                    "gradient_references": [],
                }
            }
            for variant in variants
        }
        for index, state in enumerate(states):
            baseline = run_state(
                state, variant="FP32_NATIVE", repair=False,
                selected_task=selected_task, expected_task_id=task_id,
            )
            baseline_gradient = parameters[carrier].grad.detach().float().cpu().clone()
            for variant in variants:
                repaired = run_state(
                    state, variant=variant, repair=True,
                    selected_task=selected_task, expected_task_id=task_id,
                )
                reference = repaired[task_id]
                candidate_output = baseline[task_id]
                outputs[variant][task_id]["candidate_outputs"].append(candidate_output)
                outputs[variant][task_id]["references"].append(reference)
                outputs[variant][task_id]["effects"].append(candidate_output - reference)
                gradient = parameters[carrier].grad.detach().float().cpu().clone()
                outputs[variant][task_id]["gradient_effects"].append(
                    baseline_gradient - gradient
                )
                outputs[variant][task_id]["gradient_references"].append(gradient)
            print(json.dumps({
                "event": "RMS_ORDER_INTERVENTION_STATE",
                "task_id": task_id,
                "index": index,
                "state_id": state["state_id"],
            }), flush=True)
        outputs_by_case[task_id] = outputs

    result: dict[str, Any] = {
        "schema": "gemma-rms-fp32-order-intervention-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "state_count": len(states),
        "state_ids": [str(state["state_id"]) for state in states],
        "cases": {},
        "scope": "REAL_GEMMA_COMPILED_TRAINING_GRAPH_FIXED_INPUT_STATES",
        "prediction": (
            "If feature reduction order is the source of the observed RMS effect, "
            "the reverse-order reference should materially change the endpoint, "
            "gradient, or write difference relative to native FP32 reference."
        ),
        "training_outcome": "NOT_MEASURED",
    }
    for case in cases:
        task_id = str(case["task_id"])
        per_case: dict[str, Any] = {
            "task_id": task_id,
            "symbol": str(case["reference_contract_symbol"]),
            "carrier": str(case["carrier"]),
            "variants": {},
        }
        for variant in variants:
            data = outputs_by_case[task_id][variant][task_id]
            per_case["variants"][variant] = {
                "endpoint": _summary(data["effects"], data["references"], torch=torch),
                "parameter_gradient": _summary(
                    data["gradient_effects"],
                    data["gradient_references"],
                    torch=torch,
                ),
            }
        native = per_case["variants"]["FP32_NATIVE"]["endpoint"]
        reverse = per_case["variants"]["FP32_REVERSE_FEATURE_ORDER"]["endpoint"]
        native_data = outputs_by_case[task_id]["FP32_NATIVE"][task_id]
        reverse_data = outputs_by_case[task_id]["FP32_REVERSE_FEATURE_ORDER"][task_id]
        reference_endpoint_intervention = _summary(
            [reverse_value - native_value
             for reverse_value, native_value in zip(
                 reverse_data["references"], native_data["references"]
             )],
            native_data["references"],
            torch=torch,
        )
        reference_gradient_intervention = _summary(
            [reverse_value - native_value
             for reverse_value, native_value in zip(
                 reverse_data["gradient_references"], native_data["gradient_references"]
             )],
            native_data["gradient_references"],
            torch=torch,
        )
        per_case["source_intervention"] = {
            "changed_choice": "FP32 feature reduction order in the reference mean-square sum",
            "reference_endpoint_difference": reference_endpoint_intervention,
            "reference_parameter_gradient_difference": reference_gradient_intervention,
            "endpoint_total_rms_absolute_difference": abs(
                float(native["total_effect_rms"] or 0.0) - float(reverse["total_effect_rms"] or 0.0)
            ),
            "endpoint_aligned_ratio_absolute_difference": abs(
                float(native["aligned_ratio_of_sums"] or 0.0) - float(reverse["aligned_ratio_of_sums"] or 0.0)
            ),
            "prediction_result": "NO_MATERIAL_CHANGE" if (
                float(reference_endpoint_intervention["total_effect_rms"] or 0.0) < 1e-5
                and float(reference_gradient_intervention["total_effect_rms"] or 0.0) < 1e-5
            ) else "CHANGED",
        }
        result["cases"][task_id] = per_case

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
