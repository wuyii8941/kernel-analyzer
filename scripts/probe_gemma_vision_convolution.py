#!/usr/bin/env python3
"""One-state exact F+B repair probe for Gemma's vision convolution.

This is deliberately separate from the Triton target probe: convolution is a
new semantic/implementation family, while the already-closed LayerNorm and
pool targets must not be counted or recomputed as new cases.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoProcessor

from qwen_candidate_step import configure_candidate_runtime
from scripts.generated_nontriton_fp32_observer import GeneratedNonTritonFP32Observer
from scripts.run_generated_fp32_screen import (
    Gemma3ImageLossStep, gradient_digest, load_model, prepare_values, tensor_digest,
)
from scripts.runtime_schedule_binding import bind_runtime_schedule


def parameter_grad_digests(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: ("NONE" if parameter.grad is None else tensor_digest(parameter.grad))
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--warm-state", type=int, default=2)
    parser.add_argument("--probe-state", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=("forward", "backward"), default="forward")
    args = parser.parse_args()

    states = json.loads(args.input_bank.read_text())["states"]
    device = torch.device(args.device)
    configure_candidate_runtime(27000)
    model = load_model("gemma3", args.model, device)
    model.gradient_checkpointing_enable()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    candidate = torch.compile(
        Gemma3ImageLossStep(model), backend="inductor", fullgraph=False, dynamic=False
    )

    warm, warm_digests = prepare_values(
        states[args.warm_state], modality="IMAGE_TEXT", model_path=args.model,
        device=device, processor=processor,
    )
    model.zero_grad(set_to_none=True)
    candidate(*warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules)
    inventory_path = args.work_dir.with_name(f"{args.work_dir.name}_inventory.json.gz")
    campaign_path = args.work_dir.with_name(f"{args.work_dir.name}_campaign.json.gz")
    bind_runtime_schedule(
        modules=modules, work_dir=args.work_dir,
        manifest=args.work_dir.with_name(f"{args.work_dir.name}_manifest.json"),
        inventory=inventory_path, campaign=campaign_path, architecture="gemma3",
        state=states[args.warm_state], input_digests=warm_digests, values=warm,
        modality="IMAGE_TEXT", gradient_checkpointing=True, allow_graph_breaks=True,
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
    choices = sorted(
        (
            row for row in inventory["runtime_call_audit"]["rows"]
            if row.get("category") == "COMPUTE"
            and row.get("implementation_kind_or_helper_role") == target_kind
            and row.get("function") == target_function
            and row.get("phase") == target_phase
        ),
        key=lambda row: (row["source_path"], row["source_line"], row["compute_region_id"]),
    )
    if len(choices) != 1:
        raise RuntimeError(f"expected one exact vision convolution {args.target}, found {len(choices)}")
    target = choices[0]

    probe, probe_digests = prepare_values(
        states[args.probe_state], modality="IMAGE_TEXT", model_path=args.model,
        device=device, processor=processor,
    )
    seed = 27000 + args.probe_state
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    baseline_loss = candidate(*probe); baseline_loss.backward(); torch.cuda.synchronize(device)
    baseline_gradients = parameter_grad_digests(model)
    baseline_full_digest = gradient_digest(model)

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    observer = GeneratedNonTritonFP32Observer(
        modules=modules,
        inventory_rows=inventory["runtime_call_audit"]["rows"],
        repair_targets={target["compute_region_id"]: [target_endpoint]},
    )
    with observer:
        repair_loss = candidate(*probe); repair_loss.backward()
    torch.cuda.synchronize(device)
    repair_gradients = parameter_grad_digests(model)
    changed = sorted(
        name for name in baseline_gradients
        if baseline_gradients[name] != repair_gradients[name]
    )
    summary = observer.summary()
    records = [
        row for row in summary["records"]
        if row["region_id"] == target["compute_region_id"]
    ]
    if len(records) != 1 or records[0]["repaired_endpoints"] != [target_endpoint]:
        raise RuntimeError("exact convolution repair was not applied exactly once")
    payload = {
        "schema": "kernel-analyzer-gemma-vision-convolution-fb-repair-probe-v1",
        "status": "COMPLETE",
        "case_id": f"gemma_vision_convolution_{args.target}",
        "model": str(args.model.resolve()),
        "probe_state": states[args.probe_state]["state_id"],
        "probe_input_digests": probe_digests,
        "target": {key: target[key] for key in (
            "compute_region_id", "phase", "function", "source_path", "source_line",
            "source_line_sha256",
        )},
        "baseline_loss_sha256": tensor_digest(baseline_loss),
        "repair_loss_sha256": tensor_digest(repair_loss),
        "baseline_gradient_sha256": baseline_full_digest,
        "repair_gradient_sha256": gradient_digest(model),
        "changed_parameter_gradients": changed,
        "changed_parameter_gradient_count": len(changed),
        "endpoint": target_endpoint,
        "endpoint_metrics": records[0]["endpoint_metrics"][target_endpoint],
        "repair_applied": records[0]["repaired_endpoints"],
        "claim_boundary": (
            f"Engineering-state exact convolution-{args.target} repair; population bias is not yet claimed."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "changed_parameter_gradient_count": len(changed),
        "endpoint_rms": payload["endpoint_metrics"]["rms"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
