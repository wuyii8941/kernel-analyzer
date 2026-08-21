#!/usr/bin/env python3
"""Cheap parameter-reach screen for frozen random Gemma-4 screen negatives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from scripts.build_implementation_census import _campaign_rows, _identity, _read, _records
from scripts.generated_fp32_observer import GeneratedFP32Observer
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules
from scripts.run_generated_fp32_screen import load_model


def gradient_fingerprints(model: torch.nn.Module) -> dict[str, tuple[float, float]]:
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        total = 0.0; square = 0.0
        flat = parameter.grad.detach().reshape(-1)
        for start in range(0, flat.numel(), 1 << 22):
            chunk = flat[start:start + (1 << 22)].float()
            total += float(torch.sum(chunk).item())
            norm = float(torch.linalg.vector_norm(chunk).item())
            square += norm * norm
        result[name] = (total, square)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--runtime-release", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bank = json.loads(args.input_bank.read_text())
    state = next(row for row in bank["states"] if row["role"] == "ENGINEERING")
    roster = json.loads(args.eligibility.read_text())["random_screen_negative_recall_audit"]
    device = torch.device(args.device); configure_candidate_runtime(20260821)
    model = load_model("gemma4", args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(values).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.runtime_release / "capture.json").read_text())
    observed = [
        hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()
        for module, _ in wrapper_modules(modules)
    ]
    wrapper_release_exact = observed == [row["sha256"] for row in capture["modules"]]
    campaign_path = args.runtime_release / "campaign.json.gz"
    with gzip.open(campaign_path, "rt") as handle: campaign = json.load(handle)
    campaign_rows = {row["region_id"]: row for row in campaign["rows"]}
    target_keys = {
        (row["implementation_pattern_id"], row["representative_exact_implementation_id"], row["endpoint"])
        for row in roster
    }
    mapped = {}
    for screen_name in ("triton_screen.json.gz",):
        screen_path = args.runtime_release / screen_name
        screen = _read(screen_path)
        bound_campaign = _campaign_rows(screen_path, screen)
        for _, record in _records(screen):
            identity = _identity(record, bound_campaign.get(record.get("region_id"), {}))
            if identity is None: continue
            for endpoint in record.get("endpoint_metrics", {}):
                key = (identity["implementation_pattern_id"], identity["exact_implementation_id"], endpoint)
                if key in target_keys:
                    mapped[key] = record["region_id"]
    if set(mapped) != target_keys:
        raise RuntimeError(f"frozen recall targets not uniquely mapped: {len(mapped)}/{len(target_keys)}")

    seed = 20260821
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True); candidate(values).backward(); torch.cuda.synchronize(device)
    baseline = gradient_fingerprints(model)
    results = []
    for index, roster_row in enumerate(roster):
        key = (
            roster_row["implementation_pattern_id"],
            roster_row["representative_exact_implementation_id"], roster_row["endpoint"],
        )
        region_id = mapped[key]; row = campaign_rows[region_id]
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = GeneratedFP32Observer(
            modules=modules, campaign_rows=[row],
            repair_targets={region_id: [roster_row["endpoint"]]}, allow_unlisted_calls=True,
        )
        with observer: candidate(values).backward()
        torch.cuda.synchronize(device)
        repaired = gradient_fingerprints(model)
        changed = sorted(name for name in baseline if baseline[name] != repaired.get(name))
        results.append({
            "implementation_pattern_id": key[0], "exact_implementation_id": key[1],
            "semantic_family_id": roster_row["semantic_family_id"],
            "operation": roster_row["operation"], "phase": roster_row["phase"],
            "endpoint": key[2], "region_id": region_id,
            "parameter_reach_screen": bool(changed),
            "changed_parameter_fingerprint_count": len(changed),
            "changed_parameter_names": changed,
        })
        print(json.dumps({
            "event": "GEMMA4_RANDOM_RECALL_REACH", "index": index + 1,
            "parameter_reach": bool(changed),
        }), flush=True)
    payload = {
        "schema": "kernel-analyzer-gemma4-random-recall-reach-v1",
        "status": "COMPLETE_DIAGNOSTIC_FINGERPRINT_SCREEN",
        "wrapper_release_exact": wrapper_release_exact,
        "results": results,
        "denominator": {
            "frozen_random_semantic_families": len(results),
            "parameter_reachable": sum(row["parameter_reach_screen"] for row in results),
            "not_parameter_reachable": sum(not row["parameter_reach_screen"] for row in results),
        },
        "claim_boundary": (
            "Two scalar fingerprints per complete parameter gradient are a cheap reach screen. "
            "Any positive requires exact-vector and frozen-wrapper closure before becoming a "
            "scientific case. A wrapper mismatch therefore cannot create a positive verdict."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
