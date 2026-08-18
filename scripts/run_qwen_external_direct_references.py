#!/usr/bin/env python3
"""Observe all Qwen seq64 external GEMMs and the direct embedding VJP call."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoModelForCausalLM

from run_qwen_current_triton_references import DESIGN, gradient_digest, tensor_digest, write
from qwen_candidate_step import LossStep, configure_candidate_runtime
from forkcert.embedding_gradient_semantic_reference import EmbeddingGradientSemanticReference
from forkcert.generated_direct_aten_reference_observer import GeneratedDirectAtenReferenceObserver
from forkcert.generated_embedding_reference_observer import GeneratedEmbeddingReferenceObserver
from forkcert.generated_external_reference_observer import GeneratedExternalReferenceObserver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-design", type=Path, default=DESIGN)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--metric-chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--length-bucket", default="seq64")
    args = parser.parse_args()

    design = json.loads(args.state_design.read_text())
    states = [
        row for index, row in enumerate(design["records"])
        if row["split"] == "heldout" and row["length_bucket"] == args.length_bucket
        and index % args.shard_count == args.shard_index
    ]
    with gzip.open(args.inventory, "rt", encoding="utf-8") as handle:
        inventory = json.load(handle)
    direct_rows = inventory["direct_runtime_calls"]["rows"]
    expected_external = inventory["runtime_call_audit"]["denominator"]["compute_kind_counts"]["EXTERN"]
    if expected_external < 0:
        raise RuntimeError("invalid external denominator")
    phase_sets = {}
    for row in inventory["runtime_call_audit"]["rows"]:
        if row.get("category") != "COMPUTE" or row.get(
            "implementation_kind_or_helper_role"
        ) != "EXTERN":
            continue
        phase_sets.setdefault(row["source_line_sha256"], set()).add(row["phase"])
    if any(len(phases) != 1 for phases in phase_sets.values()):
        raise RuntimeError("external source-line digest has conflicting phases")
    phase_by_digest = {
        value: next(iter(phases)) for value, phases in phase_sets.items()
    }

    device = torch.device("cuda:0")
    configure_candidate_runtime(24000)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device).train()
    model.config.use_cache = False
    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0]["input_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])

    payload = {
        "schema": "kernel-analyzer-qwen-external-direct-heldout-v1",
        "status": "RUNNING",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "inventory_sha256": inventory["result_sha256"],
        "denominator": {"external": expected_external, "direct_aten": len(direct_rows)},
        "states": {},
    }
    for state in states:
        state_id = state["sequence_id"]
        values = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        base_loss = candidate(values)
        base_loss.backward()
        torch.cuda.synchronize(device)
        baseline = (tensor_digest(base_loss), gradient_digest(model))
        repeats = []
        for repeat in range(2):
            semantic = EmbeddingGradientSemanticReference()
            embedding = GeneratedEmbeddingReferenceObserver(
                modules=modules, chunk_elements=args.metric_chunk_elements,
                sample_size=args.sample_size,
                embedding_gradient_semantic_reference=semantic,
            )
            external = GeneratedExternalReferenceObserver(
                modules=modules, sample_size=args.sample_size,
                metric_chunk_elements=args.metric_chunk_elements,
                phase_by_source_line_sha256=phase_by_digest,
            )
            direct = GeneratedDirectAtenReferenceObserver(
                modules=modules, inventory_rows=direct_rows,
                semantic_reference=semantic, sample_size=args.sample_size,
                metric_chunk_elements=args.metric_chunk_elements,
            )
            model.zero_grad(set_to_none=True)
            with embedding, external, direct:
                loss = candidate(values)
                loss.backward()
            torch.cuda.synchronize(device)
            external_summary = external.summary()
            direct_summary = direct.summary()
            stable = (tensor_digest(loss), gradient_digest(model)) == baseline
            if external_summary["record_count"] != expected_external:
                raise RuntimeError(f"external census incomplete: {external_summary['record_count']}")
            if direct_summary["status"] != "COMPLETE_DIRECT_ATEN_INDEX_PUT_REFERENCE":
                raise RuntimeError(
                    f"direct ATen census incomplete: {direct_summary}"
                )
            if not stable:
                raise RuntimeError("generated observers perturbed the full step")
            repeats.append({
                "repeat": repeat, "observation_stable": stable,
                "external": external_summary, "direct_aten": direct_summary,
                "specialized_embedding_record_count": len(embedding.online_records),
            })
            payload["states"][state_id] = {"record_sha256": state["record_sha256"], "repeats": repeats}
            write(args.output, payload)
            print(json.dumps({"event": "COMPLETE_REPEAT", "state": state_id, "repeat": repeat, "external": expected_external}), flush=True)
    payload["status"] = "COMPLETE_EXTERNAL_DIRECT_HELDOUT_SHARD"
    write(args.output, payload)


if __name__ == "__main__":
    main()
