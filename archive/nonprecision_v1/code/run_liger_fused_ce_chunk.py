#!/usr/bin/env python3
"""Run fixed-dtype Liger fused-CE base versus zero-padded chunk geometry."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from run_lmhead_layout import deterministic_sketch_and_metrics, tensor_digest


DATA_ROOT = Path("/data1/tzh").resolve()
ENDPOINTS = ("loss", "active_dH", "dW")


def checked(path: Path) -> Path:
    result = path.resolve()
    if DATA_ROOT not in (result, *result.parents):
        raise RuntimeError(f"path must stay under /data1/tzh: {result}")
    return result


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compact_metrics(candidate: Any, reference: Any) -> dict[str, Any]:
    result = deterministic_sketch_and_metrics(candidate, reference)
    result.pop("residual_countsketch8192")
    return result


def run_region(module: Any, hidden: Any, weight: Any, labels: Any, physical_tokens: int) -> dict[str, Any]:
    import torch

    active = hidden.shape[-2]
    if physical_tokens < active:
        raise RuntimeError("physical token count truncates active rows")
    flat = hidden.detach().reshape(-1, hidden.shape[-1])
    pad_rows = physical_tokens - active
    if pad_rows:
        flat = torch.cat(
            [flat, torch.zeros((pad_rows, flat.shape[-1]), device=flat.device, dtype=flat.dtype)],
            dim=0,
        )
        target = torch.cat(
            [labels, torch.full((pad_rows,), -100, device=labels.device, dtype=labels.dtype)],
            dim=0,
        )
    else:
        target = labels
    value = flat.detach().clone().requires_grad_(True)
    loss = module(weight, value, target)
    d_h, d_w = torch.autograd.grad(loss, (value, weight))
    tail = d_h[active:]
    return {
        "endpoints": {
            "loss": loss.detach(),
            "active_dH": d_h[:active].detach().reshape_as(hidden),
            "dW": d_w.detach(),
        },
        "tail_dH_exact_zero": bool(tail.numel() == 0 or torch.count_nonzero(tail).item() == 0),
        "tail_rows": pad_rows,
        "input_active_digest": tensor_digest(value[:active]),
        "labels_active_digest": tensor_digest(target[:active]),
        "loss_grad_fn": type(loss.grad_fn).__name__,
    }


def repeats_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import torch

    return all(torch.equal(left["endpoints"][name], right["endpoints"][name]) for name in ENDPOINTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DATA_ROOT / "models/Qwen/Qwen3-1.7B")
    parser.add_argument("--phase", choices=("pilot", "discovery", "confirmation"), required=True)
    parser.add_argument("--state-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = checked(args.protocol)
    design_path = checked(args.design)
    output_dir = checked(args.output_dir)
    model_path = checked(args.model)
    protocol = json.loads(protocol_path.read_text())
    design = json.loads(design_path.read_text())
    if protocol["status"] != "FROZEN_BEFORE_ANY_PADDED_CHUNK_VALUES_AFTER_BASELINE_CASE":
        raise RuntimeError("chunk protocol differs")
    if protocol["bindings"]["state_design"]["sha256"] != sha256(design_path):
        raise RuntimeError("state design digest differs")
    allocations = list(protocol["state_allocations"][args.phase])
    if args.state_id:
        allocations = [row for row in allocations if row["state_id"] == args.state_id]
        if len(allocations) != 1:
            raise RuntimeError("state outside frozen allocation")
    if args.limit is not None:
        allocations = allocations[: args.limit]
    records = {row["sequence_id"]: row for row in design["records"]}

    import torch
    import transformers
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device)
    model.config.use_cache = False
    model.eval()
    weight = model.lm_head.weight
    tied = bool(weight.untyped_storage().data_ptr() == model.model.embed_tokens.weight.untyped_storage().data_ptr())
    if not tied:
        raise RuntimeError("lm_head storage is no longer tied")
    modules = {
        "bf16": LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=None).to(device),
        "fp32": LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=torch.float32).to(device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    for allocation in allocations:
        state_id = allocation["state_id"]
        output_path = output_dir / f"{state_id}.json.gz"
        if output_path.exists():
            if args.skip_existing:
                print(json.dumps({"state": state_id, "status": "SKIPPED_EXISTING"}), flush=True)
                continue
            raise FileExistsError(output_path)
        record = records[state_id]
        if record["record_sha256"] != allocation["record_sha256"]:
            raise RuntimeError("state record digest differs")
        input_ids = torch.tensor([record["input_ids"]], device=device, dtype=torch.long)
        with torch.no_grad():
            hidden_1 = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state.detach().clone()
            hidden_2 = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state.detach().clone()
        labels = torch.nn.functional.pad(input_ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        first: dict[str, dict[int, Any]] = {name: {} for name in modules}
        repeat: dict[str, dict[int, bool]] = {name: {} for name in modules}
        for name, module in modules.items():
            for physical in (128, 256):
                first[name][physical] = run_region(module, hidden_1, weight, labels, physical)
                second = run_region(module, hidden_1, weight, labels, physical)
                repeat[name][physical] = repeats_equal(first[name][physical], second)
                del second
        contrasts = {
            name: {
                endpoint: deterministic_sketch_and_metrics(
                    first[name][256]["endpoints"][endpoint], first[name][128]["endpoints"][endpoint]
                )
                for endpoint in ENDPOINTS
            }
            for name in modules
        }
        accumulator_contrasts = {
            physical: {
                endpoint: compact_metrics(
                    first["bf16"][physical]["endpoints"][endpoint],
                    first["fp32"][physical]["endpoints"][endpoint],
                )
                for endpoint in ENDPOINTS
            }
            for physical in (128, 256)
        }
        common_digests = all(
            first[name][physical]["input_active_digest"] == tensor_digest(hidden_1)
            and first[name][physical]["labels_active_digest"] == tensor_digest(labels)
            for name in modules
            for physical in (128, 256)
        )
        all_finite = all(
            contrasts[name][endpoint]["finite"] for name in modules for endpoint in ENDPOINTS
        )
        gates = {
            "natural_backbone_repeat_exact": bool(torch.equal(hidden_1, hidden_2)),
            "lmhead_embedding_storage_tied": tied,
            "all_active_inputs_and_labels_exact": common_digests,
            "all_padded_tail_dH_exact_zero": all(first[name][256]["tail_dH_exact_zero"] for name in modules),
            "all_arms_repeat_bitwise_exact": all(all(row.values()) for row in repeat.values()),
            "all_values_finite": all_finite,
        }
        artifact: dict[str, Any] = {
            "schema_version": "kernel-analyzer.liger-fused-ce-chunk-state.v1",
            "status": "COMPLETE" if all(gates.values()) else "FAILED_GATE",
            "state": {"state_id": state_id, "cluster_id": record["cluster_id"], "phase": args.phase, "record_sha256": record["record_sha256"]},
            "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "triton": __import__("triton").__version__, "gpu": torch.cuda.get_device_name(device), "dtype": "torch.bfloat16", "tf32": False},
            "chunk_schedules": {
                "128": {"physical_tokens": 128, "chunk_size": 2, "total_chunks": 64, "active_chunks": 64},
                "256": {"physical_tokens": 256, "chunk_size": 4, "total_chunks": 64, "active_chunks": 32},
            },
            "gates": gates,
            "repeat_exact": repeat,
            "padded_minus_base": contrasts,
            "bf16_minus_fp32_accum": accumulator_contrasts,
            "bindings": {
                "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
                "state_design": {"path": str(design_path), "sha256": sha256(design_path)},
                "hidden_digest": tensor_digest(hidden_1),
                "weight_digest": tensor_digest(weight),
                "labels_digest": tensor_digest(labels),
            },
        }
        artifact["artifact_sha256"] = digest(artifact)
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(artifact, handle, sort_keys=True, separators=(",", ":"))
        print(json.dumps({"state": state_id, "status": artifact["status"], "bf16": {endpoint: contrasts["bf16"][endpoint]["max_abs"] for endpoint in ENDPOINTS}, "fp32": {endpoint: contrasts["fp32"][endpoint]["max_abs"] for endpoint in ENDPOINTS}}, sort_keys=True), flush=True)
        del input_ids, hidden_1, hidden_2, labels, first, contrasts, accumulator_contrasts, artifact
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
