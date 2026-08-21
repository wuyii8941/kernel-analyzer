#!/usr/bin/env python3
"""One-state exact F+B repair probes for new Ministral semantic regions.

Repeated transformer layers remain in the coverage denominator.  This probe
deep-measures only one deterministic representative of each new implementation
pattern: YaRN position scaling and the attention-mask/softmax fusion.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.generated_fp32_observer import GeneratedFP32Observer
from scripts.run_generated_fp32_screen import gradient_digest, load_model, tensor_digest
from scripts.runtime_schedule_binding import bind_runtime_schedule


TARGETS = {
    "yarn": {
        "case_id": "ministral_yarn_position_scaling",
        "symbol_contains": "div_floor_log_mul_unsqueeze",
        "endpoints": ["out_ptr0"],
    },
    "attention_softmax": {
        "case_id": "ministral_attention_mask_softmax",
        "symbol_contains": "new_ones_prepare_softmax_online",
        "endpoints": ["in_out_ptr0", "out_ptr2"],
    },
}


def parameter_grad_digests(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: ("NONE" if parameter.grad is None else tensor_digest(parameter.grad))
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--warm-state", type=int, default=2)
    parser.add_argument("--probe-state", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    target_spec = TARGETS[args.target]
    bank = json.loads(args.input_bank.read_text())
    states = bank["states"]
    device = torch.device(args.device)
    configure_candidate_runtime(31_000)
    model = load_model("mistral3", args.model, device)
    model.eval()
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)

    def values(index: int) -> tuple[torch.Tensor]:
        return (torch.tensor([states[index]["token_ids"]], dtype=torch.long, device=device),)

    model.zero_grad(set_to_none=True)
    candidate(*values(args.warm_state)).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules)
    inventory_path = args.work_dir.with_name(f"{args.work_dir.name}_inventory.json.gz")
    campaign_path = args.work_dir.with_name(f"{args.work_dir.name}_campaign.json.gz")
    bind_runtime_schedule(
        modules=modules, work_dir=args.work_dir,
        manifest=args.work_dir.with_name(f"{args.work_dir.name}_manifest.json"),
        inventory=inventory_path, campaign=campaign_path, architecture="mistral3",
        state=states[args.warm_state], input_digests={}, values=values(args.warm_state),
        modality="TEXT", gradient_checkpointing=False, allow_graph_breaks=True,
    )
    with gzip.open(campaign_path, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    choices = sorted(
        (
            row for row in campaign["rows"]
            if row["phase"] == "FORWARD"
            and target_spec["symbol_contains"] in row["symbol"]
            and all(endpoint in row["output_names"] for endpoint in target_spec["endpoints"])
        ),
        key=lambda row: (row["source_path"], row["source_line"], row["region_id"]),
    )
    if not choices:
        raise RuntimeError(f"target absent: {args.target}")
    row = choices[0]

    seed = 31_000 + args.probe_state
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    baseline_loss = candidate(*values(args.probe_state)); baseline_loss.backward()
    torch.cuda.synchronize(device)
    baseline_gradients = parameter_grad_digests(model)
    baseline_full_digest = gradient_digest(model)

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    observer = GeneratedFP32Observer(
        modules=modules, campaign_rows=[row],
        repair_targets={row["region_id"]: target_spec["endpoints"]},
        allow_unlisted_calls=True,
    )
    with observer:
        repaired_loss = candidate(*values(args.probe_state)); repaired_loss.backward()
    torch.cuda.synchronize(device)
    repaired_gradients = parameter_grad_digests(model)
    changed = sorted(
        name for name in baseline_gradients
        if baseline_gradients[name] != repaired_gradients[name]
    )
    summary = observer.summary()
    records = [record for record in summary["records"] if record["region_id"] == row["region_id"]]
    if len(records) != 1 or sorted(records[0]["repaired_endpoints"]) != sorted(target_spec["endpoints"]):
        raise RuntimeError("exact repair did not apply once to every declared endpoint")
    payload = {
        "schema": "kernel-analyzer-ministral-new-fb-repair-probe-v1",
        "status": "COMPLETE",
        "case_id": target_spec["case_id"],
        "model": str(args.model.resolve()),
        "probe_state": states[args.probe_state]["state_id"],
        "representative_policy": "FIRST_EXACT_REGION_OF_NEW_IMPLEMENTATION_PATTERN",
        "repeated_regions_in_coverage_denominator": len(choices),
        "target": {
            "region_id": row["region_id"], "symbol": row["symbol"],
            "source_path": row["source_path"], "source_line": row["source_line"],
            "endpoints": target_spec["endpoints"],
        },
        "baseline_loss_sha256": tensor_digest(baseline_loss),
        "repair_loss_sha256": tensor_digest(repaired_loss),
        "baseline_gradient_sha256": baseline_full_digest,
        "repair_gradient_sha256": gradient_digest(model),
        "changed_parameter_gradients": changed,
        "changed_parameter_gradient_count": len(changed),
        "endpoint_metrics": {
            endpoint: records[0]["endpoint_metrics"][endpoint]
            for endpoint in target_spec["endpoints"]
        },
        "claim_boundary": (
            "Engineering-state exact F+B repair for one deterministic representative; "
            "repeated regions remain counted but are not duplicated as mechanisms."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "case_id": payload["case_id"],
        "changed_parameter_gradient_count": len(changed),
        "repeated_regions_in_denominator": len(choices),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
