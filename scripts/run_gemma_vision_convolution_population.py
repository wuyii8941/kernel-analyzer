#!/usr/bin/env python3
"""Complete-Gram F+B test on the convolution unit's own parameter block."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoProcessor

from qwen_candidate_step import configure_candidate_runtime
from scripts.generated_nontriton_fp32_observer import GeneratedNonTritonFP32Observer
from scripts.run_generated_fp32_screen import Gemma3ImageLossStep, load_model, prepare_values
from scripts.runtime_schedule_binding import bind_runtime_schedule


FORWARD_PARAMETERS = (
    "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
)


def dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return sum(float(torch.dot(left[name].reshape(-1), right[name].reshape(-1))) for name in left)


def sign_flip_p_value(gram: np.ndarray) -> float:
    observed = float(gram.sum() - np.trace(gram))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(gram)):
        vector = np.asarray(signs)
        values.append(float(vector @ gram @ vector - np.trace(gram)))
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=("forward", "backward"), default="forward")
    args = parser.parse_args()
    states = json.loads(args.input_bank.read_text())["states"]
    evaluation = [row for row in states if row["role"] == "SCREENING"]
    if len(evaluation) != 8:
        raise RuntimeError("population protocol requires eight screening states")
    probe = json.loads(args.probe.read_text())
    if probe["status"] != "COMPLETE" or probe["changed_parameter_gradient_count"] == 0:
        raise RuntimeError("convolution engineering repair is not closed")

    device = torch.device(args.device)
    configure_candidate_runtime(28000)
    model = load_model("gemma3", args.model, device)
    model.gradient_checkpointing_enable()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    candidate = torch.compile(Gemma3ImageLossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm, warm_digests = prepare_values(
        states[2], modality="IMAGE_TEXT", model_path=args.model, device=device, processor=processor,
    )
    model.zero_grad(set_to_none=True); candidate(*warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules)
    inventory_path = args.work_dir.with_name(f"{args.work_dir.name}_inventory.json.gz")
    campaign_path = args.work_dir.with_name(f"{args.work_dir.name}_campaign.json.gz")
    bind_runtime_schedule(
        modules=modules, work_dir=args.work_dir,
        manifest=args.work_dir.with_name(f"{args.work_dir.name}_manifest.json"),
        inventory=inventory_path, campaign=campaign_path, architecture="gemma3",
        state=states[2], input_digests=warm_digests, values=warm, modality="IMAGE_TEXT",
        gradient_checkpointing=True, allow_graph_breaks=True,
    )
    with gzip.open(inventory_path, "rt", encoding="utf-8") as handle:
        inventory = json.load(handle)
    target_function = (
        "extern_kernels.convolution" if args.target == "forward"
        else "torch.ops.aten.convolution_backward.default"
    )
    target_kind = "EXTERN" if args.target == "forward" else "DIRECT_TORCH_OP"
    target_phase = "FORWARD" if args.target == "forward" else "BACKWARD"
    target_endpoint = "output" if args.target == "forward" else "output_1"
    parameter_names = (
        FORWARD_PARAMETERS if args.target == "forward"
        else ("model.vision_tower.vision_model.embeddings.patch_embedding.weight",)
    )
    choices = [
        row for row in inventory["runtime_call_audit"]["rows"]
        if row.get("category") == "COMPUTE"
        and row.get("implementation_kind_or_helper_role") == target_kind
        and row.get("function") == target_function
        and row.get("phase") == target_phase
    ]
    if len(choices) != 1:
        raise RuntimeError(f"expected one exact convolution {args.target}, found {len(choices)}")
    target = choices[0]
    parameters = dict(model.named_parameters())
    if any(name not in parameters for name in parameter_names):
        raise RuntimeError("declared convolution parameter block changed")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    gram = np.zeros((8, 8), dtype=np.float64)
    deltas: list[dict[str, torch.Tensor]] = []
    state_rows = []
    for index, state in enumerate(evaluation):
        values, digests = prepare_values(
            state, modality="IMAGE_TEXT", model_path=args.model, device=device, processor=processor,
        )
        seed = 28000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); baseline_loss = candidate(*values)
        baseline_loss.backward(); torch.cuda.synchronize(device)
        baseline = {name: parameters[name].grad.detach().cpu().float().clone() for name in parameter_names}

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = GeneratedNonTritonFP32Observer(
            modules=modules, inventory_rows=inventory["runtime_call_audit"]["rows"],
            repair_targets={target["compute_region_id"]: [target_endpoint]},
        )
        with observer:
            repair_loss = candidate(*values); repair_loss.backward()
        torch.cuda.synchronize(device)
        delta = {
            name: parameters[name].grad.detach().cpu().float() - baseline[name]
            for name in parameter_names
        }
        for previous, other in enumerate(deltas):
            gram[index, previous] = gram[previous, index] = dot(delta, other)
        gram[index, index] = dot(delta, delta)
        deltas.append(delta)
        state_rows.append({
            "state_id": state["state_id"], "input_digests": digests,
            "baseline_loss": float(baseline_loss), "repair_loss": float(repair_loss),
        })

    diagonal = float(np.trace(gram))
    off_diagonal = float(gram.sum() - diagonal)
    amplification = float(np.sqrt(max(float(gram.sum()), 0.0) / diagonal)) if diagonal else 0.0
    payload = {
        "schema": "kernel-analyzer-gemma-vision-convolution-population-v1",
        "status": "COMPLETE_CONVOLUTION_PARAMETER_BLOCK_GRAM",
        "case_id": f"gemma_vision_convolution_{args.target}",
        "target": {key: target[key] for key in (
            "compute_region_id", "phase", "function", "source_path", "source_line",
        )},
        "endpoint": target_endpoint, "calibration_probe_sha256": probe["result_sha256"],
        "declared_parameter_coordinates": list(parameter_names),
        "coordinate_count": sum(parameters[name].numel() for name in parameter_names),
        "states": state_rows, "complete_gram": gram.tolist(),
        "diagonal_energy": diagonal, "cross_state_directional_energy": off_diagonal,
        "amplification": amplification, "exact_sign_flip_p_value": sign_flip_p_value(gram),
        "claim_boundary": (
            f"Open-loop complete Gram on the convolution F+B unit's declared {args.target} parameter block; "
            + (
                "the engineering probe separately establishes forward-output reach to all 883 parameter gradients."
                if args.target == "forward"
                else "the engineering probe establishes exact isolation to the patch-embedding weight gradient."
            )
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "coordinate_count", "amplification", "exact_sign_flip_p_value",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
