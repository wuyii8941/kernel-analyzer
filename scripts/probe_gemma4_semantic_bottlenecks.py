#!/usr/bin/env python3
"""Exact one-state F+B parameter-reach probes for frozen Gemma-4 hotspots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from scripts.generated_fp32_observer import GeneratedFP32Observer
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules
from scripts.run_generated_fp32_screen import load_model


TARGETS = {
    "loss_softcap_ce_fb": (
        ("FORWARD", "log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view", None),
        ("FORWARD", "log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward", None),
        ("BACKWARD", "log_softmax__log_softmax_backward_data", None),
    ),
    "normalization_ple_rms_fb": (
        ("FORWARD", "embedding_mean_mul_pow_view", 0),
    ),
    "attention_softmax_fb": (
        ("FORWARD", "new_ones_prepare_softmax_online", 0),
    ),
}


def tensor_digest_chunked(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(repr(tuple(value.shape)).encode())
    flat = value.detach().reshape(-1)
    for start in range(0, flat.numel(), 1 << 22):
        chunk = flat[start:start + (1 << 22)].contiguous().cpu()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_digests(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: "NONE" if parameter.grad is None else tensor_digest_chunked(parameter.grad)
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--runtime-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    state = next(row for row in bank["states"] if row["role"] == "ENGINEERING")
    device = torch.device(args.device)
    configure_candidate_runtime(20260821)
    model = load_model("gemma4", args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(values).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])

    capture = json.loads((args.runtime_release / "capture.json").read_text())
    expected = [row["sha256"] for row in capture["modules"]]
    observed = [
        hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()
        for module, _ in wrapper_modules(modules)
    ]
    if observed != expected:
        raise RuntimeError("runtime wrapper bytes differ from frozen release")
    with gzip.open(args.runtime_release / "campaign.json.gz", "rt") as handle:
        campaign = json.load(handle)

    seed = 20260821
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True); baseline_loss = candidate(values); baseline_loss.backward()
    torch.cuda.synchronize(device)
    baseline = gradient_digests(model)
    results = []
    for target_id, selectors in TARGETS.items():
        selected = []
        for phase, needle, occurrence in selectors:
            choices = sorted(
                (row for row in campaign["rows"] if row["phase"] == phase and needle in row["symbol"]),
                key=lambda row: (row["source_path"], row["source_line"], row["region_id"]),
            )
            if not choices:
                raise RuntimeError(f"missing frozen target {target_id}: {needle}")
            selected.extend(choices if occurrence is None else [choices[occurrence]])
        repair_targets = {row["region_id"]: row["output_names"] for row in selected}
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = GeneratedFP32Observer(
            modules=modules, campaign_rows=selected,
            repair_targets=repair_targets, allow_unlisted_calls=True,
        )
        with observer:
            repaired_loss = candidate(values); repaired_loss.backward()
        torch.cuda.synchronize(device)
        repaired = gradient_digests(model)
        changed = sorted(name for name in baseline if baseline[name] != repaired[name])
        records = [
            row for row in observer.summary()["records"] if row["region_id"] in repair_targets
        ]
        results.append({
            "target_id": target_id,
            "regions": [{
                "region_id": row["region_id"], "phase": row["phase"],
                "symbol": row["symbol"], "endpoints": row["output_names"],
            } for row in selected],
            "region_executions": len(records),
            "loss_changed": bool(not torch.equal(baseline_loss.detach(), repaired_loss.detach())),
            "changed_parameter_gradient_count": len(changed),
            "changed_parameter_gradients": changed,
            "parameter_reachable": bool(changed),
        })
        print(json.dumps({
            "event": "GEMMA4_REACH_PROBE", "target": target_id,
            "changed_parameter_gradients": len(changed),
        }), flush=True)
    payload = {
        "schema": "kernel-analyzer-gemma4-semantic-bottleneck-reach-v1",
        "status": "COMPLETE",
        "state_id": state["state_id"],
        "runtime_release_sha256": capture["result_sha256"],
        "results": results,
        "claim_boundary": "One-state exact F+B FP32-storage repair establishes parameter reach, not bias or persistence.",
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
