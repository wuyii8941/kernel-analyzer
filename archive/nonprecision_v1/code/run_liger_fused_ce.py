#!/usr/bin/env python3
"""Run natural BF16 fused-linear cross-entropy forward/actual-VJP units."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from forkcert.torch_training_semantic_adapter import capture_module_boundaries
from run_lmhead_layout import deterministic_sketch_and_metrics, hidden_from_boundary, tensor_digest


DATA_ROOT = Path("/data1/tzh").resolve()
ENDPOINTS = ("loss", "dH", "dW")
IMPLEMENTATIONS = ("default_accum", "fp32_accum")


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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compact_metrics(candidate: Any, reference: Any) -> dict[str, Any]:
    result = deterministic_sketch_and_metrics(candidate, reference)
    result.pop("residual_countsketch8192")
    return result


def eager_region(hidden: Any, weight: Any, labels: Any) -> dict[str, Any]:
    import torch

    h = hidden.detach().clone().requires_grad_(True)
    logits = torch.nn.functional.linear(h, weight)
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels,
        ignore_index=-100,
        reduction="mean",
    )
    d_h, d_w, d_logits = torch.autograd.grad(loss, (h, weight, logits))
    return {
        "endpoints": {"loss": loss.detach(), "dH": d_h.detach(), "dW": d_w.detach()},
        "d_logits": d_logits.detach(),
        "loss_grad_fn": type(loss.grad_fn).__name__,
    }


def fused_region(loss_module: Any, hidden: Any, weight: Any, labels: Any) -> dict[str, Any]:
    import torch

    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = loss_module(weight, h, labels)
    d_h, d_w = torch.autograd.grad(loss, (h, weight))
    return {
        "endpoints": {
            "loss": loss.detach(),
            "dH": d_h.detach().reshape_as(hidden),
            "dW": d_w.detach(),
        },
        "loss_grad_fn": type(loss.grad_fn).__name__,
    }


def fp32_region(hidden: Any, weight: Any, labels: Any) -> dict[str, Any]:
    import torch

    h = hidden.detach().to(torch.float32).clone().requires_grad_(True)
    w = weight.detach().to(torch.float32).clone().requires_grad_(True)
    logits = torch.nn.functional.linear(h, w)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels, ignore_index=-100, reduction="mean"
    )
    d_h, d_w = torch.autograd.grad(loss, (h, w))
    return {"loss": loss.detach(), "dH": d_h.detach(), "dW": d_w.detach()}


def compare_repeat(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, bool]:
    import torch

    return {name: bool(torch.equal(left["endpoints"][name], right["endpoints"][name])) for name in ENDPOINTS}


def capture_lmhead(model: Any, input_ids: Any) -> tuple[Any, Any, dict[str, bool]]:
    import torch

    sentinel = dict(model.named_parameters())["model.norm.weight"]
    model.zero_grad(set_to_none=True)
    baseline = model(input_ids=input_ids, labels=input_ids, use_cache=False, return_dict=True)
    baseline_loss = baseline.loss.detach().clone()
    baseline.loss.backward()
    baseline_grad = sentinel.grad.detach().clone()
    del baseline
    model.zero_grad(set_to_none=True)
    observed: list[Any] = []

    def closure() -> Any:
        result = model(input_ids=input_ids, labels=input_ids, use_cache=False, return_dict=True)
        observed.append(result.loss.detach().clone())
        return result.loss

    boundaries = capture_module_boundaries(
        model,
        module_names=("lm_head",),
        loss_closure=closure,
        capture_parameter_state=False,
        capture_parameter_digests=False,
    )
    gates = {
        "baseline_vs_hooked_loss_exact": len(observed) == 1 and bool(torch.equal(baseline_loss, observed[0])),
        "baseline_vs_hooked_sentinel_gradient_exact": bool(torch.equal(baseline_grad, sentinel.grad)),
        "exactly_one_lmhead_boundary_observed": len(boundaries) == 1,
    }
    if not all(gates.values()):
        raise RuntimeError(f"lm_head observation changed execution: {gates}")
    return boundaries[0], baseline_loss, gates


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
    if protocol["status"] != "FROZEN_BEFORE_ANY_LIGER_FUSED_CE_VALUES":
        raise RuntimeError("Liger fused CE protocol differs")
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
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    weight = model.lm_head.weight
    tied_storage = bool(
        weight.untyped_storage().data_ptr()
        == model.model.embed_tokens.weight.untyped_storage().data_ptr()
    )
    if not tied_storage or list(weight.shape) != [151936, 2048]:
        raise RuntimeError("lm_head tied-weight boundary differs")
    loss_modules = {
        "default_accum": LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=None
        ).to(device),
        "fp32_accum": LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=torch.float32
        ).to(device),
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
        boundary, full_loss, observation_gates = capture_lmhead(model, input_ids)
        hidden = hidden_from_boundary(boundary, device)
        captured_q = boundary.cotangent.detach().clone().to(device=device, dtype=torch.bfloat16)
        shifted = torch.nn.functional.pad(input_ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        eager_1 = eager_region(hidden, weight, shifted)
        eager_2 = eager_region(hidden, weight, shifted)
        candidates_1 = {name: fused_region(module, hidden, weight, shifted) for name, module in loss_modules.items()}
        candidates_2 = {name: fused_region(module, hidden, weight, shifted) for name, module in loss_modules.items()}
        higher = fp32_region(hidden, weight, shifted)
        comparisons: dict[str, Any] = {}
        for name in IMPLEMENTATIONS:
            comparisons[name] = {}
            for endpoint in ENDPOINTS:
                candidate_value = candidates_1[name]["endpoints"][endpoint]
                eager_value = eager_1["endpoints"][endpoint]
                comparisons[name][endpoint] = {
                    "candidate_minus_eager": deterministic_sketch_and_metrics(candidate_value, eager_value),
                    "candidate_minus_fp32_region": compact_metrics(candidate_value, higher[endpoint]),
                    "eager_minus_fp32_region": compact_metrics(eager_value, higher[endpoint]),
                }
        accumulator_contrast = {
            endpoint: deterministic_sketch_and_metrics(
                candidates_1["default_accum"]["endpoints"][endpoint],
                candidates_1["fp32_accum"]["endpoints"][endpoint],
            )
            for endpoint in ENDPOINTS
        }
        eager_repeat = compare_repeat(eager_1, eager_2)
        candidate_repeats = {
            name: compare_repeat(candidates_1[name], candidates_2[name]) for name in IMPLEMENTATIONS
        }
        logits_q_exact = bool(torch.equal(eager_1["d_logits"], captured_q))
        loss_exact = bool(torch.equal(eager_1["endpoints"]["loss"], full_loss))
        all_shapes = all(
            candidates_1[name]["endpoints"][endpoint].shape == eager_1["endpoints"][endpoint].shape
            and candidates_1[name]["endpoints"][endpoint].dtype == eager_1["endpoints"][endpoint].dtype
            for name in IMPLEMENTATIONS
            for endpoint in ENDPOINTS
        )
        all_finite = all(
            comparisons[name][endpoint]["candidate_minus_eager"]["finite"]
            for name in IMPLEMENTATIONS
            for endpoint in ENDPOINTS
        ) and all(row["finite"] for row in accumulator_contrast.values())
        gates = {
            **observation_gates,
            "lmhead_embedding_storage_tied": tied_storage,
            "standalone_eager_loss_equals_full_model_loss": loss_exact,
            "standalone_eager_logits_vjp_equals_captured_q": logits_q_exact,
            "common_hidden_weight_labels_retained": True,
            "all_endpoint_shapes_and_dtypes_match": all_shapes,
            "all_values_finite": all_finite,
            "all_eager_repeats_bitwise_exact": all(eager_repeat.values()),
            "all_candidate_repeats_bitwise_exact": all(all(row.values()) for row in candidate_repeats.values()),
        }
        artifact: dict[str, Any] = {
            "schema_version": "kernel-analyzer.liger-fused-ce-state.v1",
            "status": "COMPLETE" if all(gates.values()) else "FAILED_GATE",
            "state": {
                "state_id": state_id,
                "cluster_id": record["cluster_id"],
                "phase": args.phase,
                "record_sha256": record["record_sha256"],
            },
            "environment": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "triton": __import__("triton").__version__,
                "gpu": torch.cuda.get_device_name(device),
                "dtype": "torch.bfloat16",
                "tf32": False,
            },
            "boundary": {
                "hidden_shape": list(hidden.shape),
                "weight_shape": list(weight.shape),
                "labels_shape": list(shifted.shape),
                "nonignored_targets": int((shifted != -100).sum().item()),
                "hidden_digest": tensor_digest(hidden),
                "weight_digest": tensor_digest(weight),
                "labels_digest": tensor_digest(shifted),
                "captured_logits_cotangent_digest": tensor_digest(captured_q),
                "full_model_loss": float(full_loss.item()),
                "eager_loss_grad_fn": eager_1["loss_grad_fn"],
                "candidate_loss_grad_fn": {name: candidates_1[name]["loss_grad_fn"] for name in IMPLEMENTATIONS},
            },
            "chunk_schedule": {
                "tokens": 128,
                "hidden": 2048,
                "vocabulary": 151936,
                "inc_factor": math.ceil(151936 / 2048),
                "chunk_size": 2,
                "chunks": 64,
            },
            "gates": gates,
            "repeat_exact": {"eager": eager_repeat, **candidate_repeats},
            "implementations": comparisons,
            "default_minus_fp32_accum": accumulator_contrast,
            "bindings": {
                "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
                "state_design": {"path": str(design_path), "sha256": sha256(design_path)},
            },
        }
        artifact["artifact_sha256"] = digest(artifact)
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(artifact, handle, sort_keys=True, separators=(",", ":"))
        print(
            json.dumps(
                {
                    "state": state_id,
                    "status": artifact["status"],
                    "default": {endpoint: comparisons["default_accum"][endpoint]["candidate_minus_eager"]["max_abs"] for endpoint in ENDPOINTS},
                    "fp32_accum": {endpoint: comparisons["fp32_accum"][endpoint]["candidate_minus_eager"]["max_abs"] for endpoint in ENDPOINTS},
                    "accum_dW_max_abs": accumulator_contrast["dW"]["max_abs"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del boundary, hidden, captured_q, shifted, eager_1, eager_2, candidates_1, candidates_2, higher
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
