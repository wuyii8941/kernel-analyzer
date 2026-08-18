#!/usr/bin/env python3
"""Observe every currently referenceable seq64 Triton invocation online."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoModelForCausalLM

from qwen_candidate_step import LossStep, configure_candidate_runtime


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT))

from forkcert.generated_triton_reference_observer import (  # noqa: E402
    GeneratedTritonReferenceObserver,
)
from forkcert.generated_embedding_reference_observer import (  # noqa: E402
    GeneratedEmbeddingReferenceObserver,
)
from scripts.evolving_triton_observation import (  # noqa: E402
    discover_all_triton_symbols,
    remap_campaign_to_warmed_symbols,
)


MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"
CAMPAIGN = ROOT / "results/coverage/qwen_current_triton_reference_campaign.json.gz"
PROTOCOL = ROOT / "results/coverage/qwen_oracle_protocol.json"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    result = hashlib.sha256()
    result.update(str(tensor.dtype).encode())
    result.update(repr(tuple(tensor.shape)).encode())
    # A scalar cannot change element size directly; flatten before viewing bytes.
    result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def gradient_digest(model: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        result.update(name.encode())
        result.update(
            b"NONE" if parameter.grad is None else tensor_digest(parameter.grad).encode()
        )
    return result.hexdigest()


def parameter_gradient_digests(model: torch.nn.Module) -> dict[str, str | None]:
    """Per-parameter identities locate causal reach without retaining gradient tensors."""
    return {
        name: None if parameter.grad is None else tensor_digest(parameter.grad)
        for name, parameter in sorted(model.named_parameters())
    }


def sampled_parameter_gradients(
    model: torch.nn.Module, names: list[str], sample_size: int
) -> dict[str, dict[str, Any]]:
    parameters = dict(model.named_parameters())
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        parameter = parameters.get(name)
        if parameter is None or parameter.grad is None:
            rows[name] = {"status": "MISSING"}
            continue
        flat = parameter.grad.detach().reshape(-1)
        count = min(sample_size, flat.numel())
        if count == 1:
            indices = torch.zeros(1, dtype=torch.int64, device=flat.device)
        else:
            indices = (
                torch.arange(count, dtype=torch.int64, device=flat.device)
                * (flat.numel() - 1) // (count - 1)
            )
        rows[name] = {
            "status": "OK",
            "numel": int(flat.numel()),
            "indices": indices.cpu().tolist(),
            "values": flat.index_select(0, indices).float().cpu().tolist(),
        }
    return rows


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--metric-chunk-elements", type=int, default=1_048_576)
    parser.add_argument(
        "--intervention-arms", type=Path, default=None,
        help="optional JSON list of exact {region_id,endpoints} semantic interventions",
    )
    parser.add_argument("--carrier-parameter", action="append", default=[])
    parser.add_argument("--carrier-sample-size", type=int, default=4096)
    parser.add_argument("--state-design", type=Path, default=DESIGN)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--state-split", default="heldout")
    parser.add_argument("--length-bucket", default="seq64")
    parser.add_argument("--expected-states", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    design = json.loads(args.state_design.read_text())
    protocol = json.loads(args.protocol.read_text())
    with gzip.open(args.campaign, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    intervention_arms = []
    if args.intervention_arms is not None:
        intervention_arms = json.loads(args.intervention_arms.read_text())
        if not isinstance(intervention_arms, list) or not intervention_arms:
            raise ValueError("intervention arms must be a nonempty JSON list")
    intervention_ids = [str(row["region_id"]) for row in intervention_arms]
    intervention_endpoints = {
        str(row["region_id"]): list(row.get("endpoints", []))
        for row in intervention_arms if row.get("endpoints")
    }
    rows = list(campaign["rows"])
    ordinary = [
        row for row in rows
        if row["adapter_status"] != "UNRESOLVED_CURRENT_REFERENCE_ADAPTER"
        and not row["boundary_capture_mode"].startswith("SPECIALIZED_")
    ]
    specialized = [
        row for row in rows
        if row["adapter_status"] != "UNRESOLVED_CURRENT_REFERENCE_ADAPTER"
        and row["boundary_capture_mode"].startswith("SPECIALIZED_")
    ]
    unresolved_count = len(rows) - len(ordinary) - len(specialized)
    if (
        len(rows) != campaign["denominator"]["triton_invocations"]
        or len(ordinary) + len(specialized)
        != campaign["denominator"]["reference_adapter_exact"]
    ):
        raise RuntimeError("current exact reference denominator changed")
    if intervention_ids:
        target_rows = [row for row in rows if row["region_id"] in intervention_ids]
        if {row["region_id"] for row in target_rows} != set(intervention_ids):
            raise RuntimeError("an intervention region is absent from the frozen campaign")
        target_symbols = {row["symbol"] for row in target_rows}
        # Retain all invocations of a selected symbol so runtime invocation
        # indices remain exact, but do not compute metrics for 450 unrelated
        # regions during a targeted causal experiment.
        rows = [row for row in rows if row["symbol"] in target_symbols]
    all_states = [
        row for row in design["records"]
        if row["split"] == args.state_split and row["length_bucket"] == args.length_bucket
    ]
    if args.state_design.resolve() == DESIGN.resolve() and args.state_split == "heldout":
        expected_ids = {
            row["state_id"] for row in protocol["heldout_states"]
            if row["stratum"] == args.length_bucket
        }
        if {row["sequence_id"] for row in all_states} != expected_ids:
            raise RuntimeError(f"frozen {args.length_bucket} held-out population changed")
    if len(all_states) != args.expected_states:
        raise RuntimeError(f"requested frozen {args.length_bucket} population changed")
    states = [
        row for index, row in enumerate(all_states)
        if index % args.shard_count == args.shard_index
    ]

    if args.output.exists():
        with gzip.open(args.output, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["protocol_sha256"] != protocol["protocol_sha256"]:
            raise RuntimeError("existing shard protocol changed")
        if payload["campaign_sha256"] != campaign["result_sha256"]:
            raise RuntimeError("existing shard campaign changed")
    else:
        payload = {
            "schema": (
                "kernel-analyzer-current-qwen-semantic-intervention-v1"
                if intervention_ids else "kernel-analyzer-current-qwen-triton-heldout-v1"
            ),
            "status": "RUNNING",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "protocol_sha256": protocol["protocol_sha256"],
            "state_design_sha256": design.get("design_sha256"),
            "campaign_sha256": campaign["result_sha256"],
            "campaign_path": str(args.campaign.resolve()),
            "model_path": str(args.model.resolve()),
            "denominator": {
                "current_triton_invocations": len(rows),
                "reference_adapter_exact": len(ordinary) + len(specialized),
                "ordinary_online_observed_per_state": len(ordinary),
                "specialized_exact_not_observed_here": len(specialized),
                "unresolved_current_reference_adapter": unresolved_count,
            },
            "states": {},
            "intervention": {
                "region_ids": intervention_ids,
                "endpoints_by_region": intervention_endpoints,
                "reference_role": "semantic",
                "candidate_scale": 0.0,
            } if intervention_ids else None,
            "claim_boundary": (
                f"Same-input online numerical observations for the {len(ordinary)} ordinary "
                f"current {args.length_bucket} Triton invocations with exact adapters. No verdict is assigned "
                f"to the other {len(specialized) + unresolved_count}."
            ),
        }

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device).train()
    model.config.use_cache = False
    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)

    warm_input = torch.tensor([states[0]["input_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    warm_loss = candidate(warm_input)
    warm_loss.backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])
    rows, unmatched_warmed_symbols = remap_campaign_to_warmed_symbols(
        rows, discover_all_triton_symbols(modules),
        allow_extra_same_stem=bool(intervention_ids),
    )
    payload["unmatched_warmed_symbols"] = unmatched_warmed_symbols

    for state in states:
        state_id = state["sequence_id"]
        existing = payload["states"].get(state_id)
        if existing is not None and len(existing.get("repeats", [])) == 2:
            continue
        values = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values)
        baseline_loss.backward()
        torch.cuda.synchronize(device)
        baseline = {
            "loss_digest": tensor_digest(baseline_loss),
            "gradient_digest": None if args.carrier_parameter else gradient_digest(model),
            "parameter_gradient_digests": (
                {} if args.carrier_parameter else parameter_gradient_digests(model)
            ),
        }
        baseline_carriers = sampled_parameter_gradients(
            model, args.carrier_parameter, args.carrier_sample_size
        )
        state_row = {
            "record_sha256": state["record_sha256"],
            "cluster_id": state["cluster_id"],
            "repeats": [],
        }
        for repeat in range(2):
            observer = GeneratedTritonReferenceObserver(
                modules=modules,
                campaign_rows=rows,
                sequence=int(args.length_bucket.removeprefix("seq")),
                sample_size=args.sample_size,
                metric_chunk_elements=args.metric_chunk_elements,
                allow_unclosed_closures=True,
                intervene_region_ids=intervention_ids,
                intervene_endpoints_by_region=intervention_endpoints,
                intervention_candidate_scale=0.0,
                intervention_reference_role="semantic",
            )
            embedding_observer = GeneratedEmbeddingReferenceObserver(
                modules=modules,
                chunk_elements=args.metric_chunk_elements,
                sample_size=args.sample_size,
            )
            model.zero_grad(set_to_none=True)
            with observer, embedding_observer:
                loss = candidate(values)
                loss.backward()
            torch.cuda.synchronize(device)
            summary = observer.summary()
            observed = {
                "loss_digest": tensor_digest(loss),
                "gradient_digest": None if args.carrier_parameter else gradient_digest(model),
                "parameter_gradient_digests": (
                    {} if args.carrier_parameter else parameter_gradient_digests(model)
                ),
            }
            changed_parameters = sorted(
                name for name, value in observed["parameter_gradient_digests"].items()
                if value != baseline["parameter_gradient_digests"].get(name)
            )
            observed_carriers = sampled_parameter_gradients(
                model, args.carrier_parameter, args.carrier_sample_size
            )
            carrier_deltas = {}
            for name in args.carrier_parameter:
                before, after = baseline_carriers[name], observed_carriers[name]
                if before["status"] != "OK" or after["status"] != "OK":
                    carrier_deltas[name] = {"status": "MISSING"}
                    continue
                if before["indices"] != after["indices"]:
                    raise RuntimeError(f"carrier coordinates changed for {name}")
                delta = [a - b for a, b in zip(after["values"], before["values"])]
                carrier_deltas[name] = {
                    "status": "OK",
                    "numel": before["numel"],
                    "indices": before["indices"],
                    "intervened_minus_candidate": delta,
                }
            carrier_changed = any(
                any(value != 0.0 for value in row.get("intervened_minus_candidate", []))
                for row in carrier_deltas.values()
            )
            stable = observed == baseline and not any(
                any(value != 0.0 for value in row.get("intervened_minus_candidate", []))
                for row in carrier_deltas.values()
            )
            if summary["status"] != "COMPLETE_ORDINARY_TRITON_ONLINE_REFERENCE_CENSUS":
                raise RuntimeError(f"online reference census failed for {state_id}")
            specialized_records = list(embedding_observer.online_records)
            if len(specialized_records) != len(specialized):
                raise RuntimeError(
                    f"specialized embedding census failed for {state_id}: "
                    f"{len(specialized_records)} != {len(specialized)}"
                )
            if not stable and not intervention_ids:
                raise RuntimeError(f"observer perturbed full step for {state_id}")
            state_row["repeats"].append({
                "repeat": repeat,
                "observation_stable": stable,
                "loss_changed": observed["loss_digest"] != baseline["loss_digest"],
                "gradient_changed": (
                    observed["gradient_digest"] != baseline["gradient_digest"]
                    or carrier_changed
                ),
                "changed_parameters": changed_parameters,
                "carrier_deltas": carrier_deltas,
                "summary": summary,
                "specialized_embedding": {
                    "status": "COMPLETE_SPECIALIZED_EMBEDDING_CENSUS",
                    "record_count": len(specialized_records),
                    "records": specialized_records,
                },
            })
            payload["states"][state_id] = state_row
            write(args.output, payload)
            print(json.dumps({
                "event": "COMPLETE_REPEAT", "shard": args.shard_index,
                "state": state_id, "repeat": repeat,
                "regions": len(summary["records"]) + len(specialized_records),
            }), flush=True)

    payload["status"] = "COMPLETE_CURRENT_TRITON_HELDOUT_SHARD"
    payload["result_sha256"] = digest({k: v for k, v in payload.items() if k != "result_sha256"})
    write(args.output, payload)
    print(json.dumps({
        "event": "SHARD_COMPLETE", "shard": args.shard_index,
        "states": len(payload["states"]), "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
