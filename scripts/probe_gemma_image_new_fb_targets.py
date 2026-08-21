#!/usr/bin/env python3
"""One-state F+B repair probes for genuinely new Gemma vision operators."""

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
from scripts.generated_fp32_observer import GeneratedFP32Observer
from scripts.run_generated_fp32_screen import (
    Gemma3ImageLossStep, gradient_digest, load_model, prepare_values, tensor_digest,
)
from scripts.runtime_schedule_binding import bind_runtime_schedule


TARGETS = [
    {
        "case_id": "gemma_vision_layernorm_backward",
        "symbol_contains": "add_native_layer_norm_native_layer_norm_backward_view",
        "phase": "BACKWARD", "endpoint": "in_out_ptr0",
    },
    {
        "case_id": "gemma_vision_pool_norm_backward",
        "symbol_contains": "avg_pool2d_backward_div_expand_mul_native_layer_norm_backward",
        "phase": "BACKWARD", "endpoint": "out_ptr1",
    },
]


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
    args = parser.parse_args()
    states = json.loads(args.input_bank.read_text())["states"]
    device = torch.device(args.device)
    configure_candidate_runtime(25000)
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
    inventory = args.work_dir.with_name(f"{args.work_dir.name}_inventory.json.gz")
    campaign_path = args.work_dir.with_name(f"{args.work_dir.name}_campaign.json.gz")
    bind_runtime_schedule(
        modules=modules, work_dir=args.work_dir,
        manifest=args.work_dir.with_name(f"{args.work_dir.name}_manifest.json"),
        inventory=inventory, campaign=campaign_path, architecture="gemma3",
        state=states[args.warm_state], input_digests=warm_digests, values=warm,
        modality="IMAGE_TEXT", gradient_checkpointing=True, allow_graph_breaks=True,
    )
    with gzip.open(campaign_path, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    probe, probe_digests = prepare_values(
        states[args.probe_state], modality="IMAGE_TEXT", model_path=args.model,
        device=device, processor=processor,
    )
    seed = 25000 + args.probe_state
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    baseline_loss = candidate(*probe); baseline_loss.backward(); torch.cuda.synchronize(device)
    baseline_gradients = parameter_grad_digests(model)
    baseline_full_digest = gradient_digest(model)
    results = []
    for target in TARGETS:
        choices = sorted(
            (
                row for row in campaign["rows"]
                if row["phase"] == target["phase"]
                and target["symbol_contains"] in row["symbol"]
                and target["endpoint"] in row["output_names"]
            ),
            key=lambda row: (row["source_path"], row["source_line"], row["region_id"]),
        )
        if not choices:
            raise RuntimeError(f"target absent: {target['case_id']}")
        row = choices[0]
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = GeneratedFP32Observer(
            modules=modules, campaign_rows=[row],
            repair_targets={row["region_id"]: [target["endpoint"]]},
            allow_unlisted_calls=True,
        )
        with observer:
            repaired_loss = candidate(*probe); repaired_loss.backward()
        torch.cuda.synchronize(device)
        repaired_gradients = parameter_grad_digests(model)
        changed = sorted(
            name for name in baseline_gradients
            if baseline_gradients[name] != repaired_gradients[name]
        )
        summary = observer.summary()
        results.append({
            **target, "region_id": row["region_id"], "symbol": row["symbol"],
            "source_path": row["source_path"], "source_line": row["source_line"],
            "baseline_loss_sha256": tensor_digest(baseline_loss),
            "repair_loss_sha256": tensor_digest(repaired_loss),
            "baseline_gradient_sha256": baseline_full_digest,
            "repair_gradient_sha256": gradient_digest(model),
            "changed_parameter_gradients": changed,
            "changed_parameter_gradient_count": len(changed),
            "endpoint_metrics": summary["records"][0]["endpoint_metrics"][target["endpoint"]],
            "repair_applied": summary["records"][0]["repaired_endpoints"],
        })
    payload = {
        "schema": "kernel-analyzer-new-vision-fb-repair-probe-v1",
        "status": "COMPLETE" if all(row["repair_applied"] for row in results) else "INVALID",
        "model": str(args.model.resolve()), "probe_state": states[args.probe_state]["state_id"],
        "probe_input_digests": probe_digests, "targets": results,
        "claim_boundary": "Engineering-state exact endpoint repair; population bias is not yet claimed.",
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({row["case_id"]: row["changed_parameter_gradient_count"] for row in results}, sort_keys=True))


if __name__ == "__main__":
    main()
