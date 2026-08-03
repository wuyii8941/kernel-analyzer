#!/usr/bin/env python3
"""Run full-step tied-weight propagation for the Liger dW accumulator case."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from run_lmhead_propagation import parameter_delta, tensor_digest
from run_silu_propagation import clone_gradients, gradients_equal


DATA_ROOT = Path("/data1/tzh").resolve()


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


def full_step(model: Any, loss_module: Any, input_ids: Any) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    outputs = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state
    observed: list[Any] = []
    hidden.register_hook(lambda gradient: observed.append(gradient.detach().clone()))
    labels = __import__("torch").nn.functional.pad(input_ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
    loss = loss_module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
    loss_value = loss.detach().clone()
    loss.backward()
    if len(observed) != 1:
        raise RuntimeError("terminal hidden VJP was not observed exactly once")
    return {
        "loss": loss_value,
        "dH": observed[0],
        "hidden_digest": tensor_digest(hidden),
        "labels_digest": tensor_digest(labels),
        "dH_digest": tensor_digest(observed[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DATA_ROOT / "models/Qwen/Qwen3-1.7B")
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
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
    if protocol["status"] != "FROZEN_BEFORE_ANY_FULL_STEP_TIED_WEIGHT_VALUES":
        raise RuntimeError("fused-CE propagation protocol differs")
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
    tied = bool(
        model.lm_head.weight.untyped_storage().data_ptr()
        == model.model.embed_tokens.weight.untyped_storage().data_ptr()
    )
    if not tied or len(list(model.named_parameters())) != 310:
        raise RuntimeError("tied 310-parameter model boundary differs")
    modules = {
        "default": LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=None
        ).to(device),
        "fp32": LigerFusedLinearCrossEntropyLoss(
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

        default_1 = full_step(model, modules["default"], input_ids)
        default_gradients = clone_gradients(model)
        default_2 = full_step(model, modules["default"], input_ids)
        default_repeat = (
            torch.equal(default_1["loss"], default_2["loss"])
            and default_1["dH_digest"] == default_2["dH_digest"]
            and default_1["hidden_digest"] == default_2["hidden_digest"]
            and gradients_equal(model, default_gradients)
        )
        model.zero_grad(set_to_none=True)

        fp32_1 = full_step(model, modules["fp32"], input_ids)
        fp32_gradients = clone_gradients(model)
        fp32_2 = full_step(model, modules["fp32"], input_ids)
        fp32_repeat = (
            torch.equal(fp32_1["loss"], fp32_2["loss"])
            and fp32_1["dH_digest"] == fp32_2["dH_digest"]
            and fp32_1["hidden_digest"] == fp32_2["hidden_digest"]
            and gradients_equal(model, fp32_gradients)
        )
        delta = parameter_delta(fp32_gradients, default_gradients, device)
        nonzero_names = [row["name"] for row in delta["parameters"] if row["nonzero"]]
        common_values = {
            "loss": bool(torch.equal(default_1["loss"], fp32_1["loss"])),
            "terminal_dH": default_1["dH_digest"] == fp32_1["dH_digest"],
            "hidden": default_1["hidden_digest"] == fp32_1["hidden_digest"],
            "labels": default_1["labels_digest"] == fp32_1["labels_digest"],
        }
        gates = {
            "model_weight_is_tied": tied,
            "default_repeat_bitwise_exact": default_repeat,
            "fp32_repeat_bitwise_exact": fp32_repeat,
            "arms_share_loss_hidden_labels_and_terminal_dH": all(common_values.values()),
            "all_310_named_parameters_retained": delta["parameter_count"] == 310,
            "only_tied_embedding_parameter_differs": nonzero_names == ["model.embed_tokens.weight"],
            "finite_loss_and_delta": all(
                math.isfinite(value)
                for value in (
                    float(default_1["loss"].item()),
                    float(fp32_1["loss"].item()),
                    delta["global_l2"],
                    delta["global_max_abs"],
                )
            ),
        }
        artifact: dict[str, Any] = {
            "schema_version": "kernel-analyzer.liger-fused-ce-propagation-state.v1",
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
            "gates": gates,
            "common_values": common_values,
            "loss": {
                "default": float(default_1["loss"].item()),
                "fp32_accum": float(fp32_1["loss"].item()),
            },
            "parameter_gradient_delta": delta,
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
                    "nonzero_parameters": delta["nonzero_parameter_count"],
                    "global_l2": delta["global_l2"],
                    "max_abs": delta["global_max_abs"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del input_ids, default_1, default_2, fp32_1, fp32_2, default_gradients, fp32_gradients, delta, artifact
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
