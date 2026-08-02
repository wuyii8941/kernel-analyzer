#!/usr/bin/env python3
"""Capture a stable AOT forward/backward graph for the round-2 VL step."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.round2_vl_smoke import prepare_step, sha256_file  # noqa: E402
from scripts.round2_vl_static import specialize_fixed_grid  # noqa: E402
from scripts.aot_capture import (  # noqa: E402
    AOTForwardBackwardCapture,
)


def gradient_digest(model: Any) -> tuple[str, dict[str, str]]:
    combined = hashlib.sha256()
    parameters: dict[str, str] = {}
    for name, parameter in sorted(model.named_parameters()):
        if parameter.grad is None:
            digest = "NONE"
        else:
            value = parameter.grad.detach().contiguous().cpu()
            digest = hashlib.sha256(
                value.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        parameters[name] = digest
        combined.update(name.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return combined.hexdigest(), parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--runtime-repeats", type=int, default=2)
    parser.add_argument("--measure-gradient-error", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()
    if args.runtime_repeats < 2:
        raise ValueError("runtime repeats must be at least two")

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    inputs, labels, input_metadata = prepare_step(
        processor,
        args.image,
        width=args.width,
        height=args.height,
    )
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dtype = getattr(torch, args.dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    values = {key: value.to(device) for key, value in inputs.items()}
    device_labels = labels.to(device)
    position_ids, _ = model.model.get_rope_index(
        values["input_ids"],
        values["image_grid_thw"],
        attention_mask=values["attention_mask"],
    )

    model.zero_grad(set_to_none=True)
    baseline = model(
        **values,
        labels=device_labels,
        use_cache=False,
        return_dict=False,
    )[0]
    baseline.backward()
    baseline_loss = baseline.detach().clone()
    baseline_digest, baseline_parameter_digests = gradient_digest(model)
    baseline_gradients = (
        {
            name: parameter.grad.detach().contiguous().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        if args.measure_gradient_error
        else {}
    )
    model.zero_grad(set_to_none=True)
    del baseline

    specialization = specialize_fixed_grid(
        model,
        values["image_grid_thw"],
        values["input_ids"],
    )
    normalized = model(
        **values,
        position_ids=position_ids,
        labels=device_labels,
        use_cache=False,
        return_dict=False,
    )[0]
    normalized.backward()
    normalized_digest, normalized_parameter_digests = gradient_digest(model)
    normalization_loss_exact = bool(
        torch.equal(normalized.detach(), baseline_loss)
    )
    normalization_gradients_exact = (
        normalized_digest == baseline_digest
        and normalized_parameter_digests == baseline_parameter_digests
    )
    model.zero_grad(set_to_none=True)
    del normalized
    if not normalization_loss_exact or not normalization_gradients_exact:
        raise RuntimeError("fixed-grid normalization changed loss or gradients")

    class LossStep(torch.nn.Module):
        def __init__(self, subject: Any) -> None:
            super().__init__()
            self.subject = subject

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor,
            image_grid_thw: torch.Tensor,
            position_ids: torch.Tensor,
            loss_labels: torch.Tensor,
        ) -> torch.Tensor:
            return self.subject(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                labels=loss_labels,
                use_cache=False,
                return_dict=False,
            )[0]

    capture = AOTForwardBackwardCapture()
    compiled = torch.compile(
        LossStep(model),
        backend=capture.backend(),
        fullgraph=True,
        dynamic=False,
    )
    compiled_args = (
        values["input_ids"],
        values["attention_mask"],
        values["pixel_values"],
        values["image_grid_thw"],
        position_ids,
        device_labels,
    )
    runs = []
    for repeat in range(args.runtime_repeats):
        model.zero_grad(set_to_none=True)
        loss = compiled(*compiled_args)
        capture.bind_user_outputs(loss)
        loss.register_hook(capture.bind_user_cotangent)
        loss.backward()
        digest, parameter_digests = gradient_digest(model)
        differing_parameters = [
            name
            for name in sorted(baseline_parameter_digests)
            if parameter_digests.get(name) != baseline_parameter_digests[name]
        ]
        gradient_error = None
        if args.measure_gradient_error:
            rows = []
            delta_l2_squared = 0.0
            reference_l2_squared = 0.0
            for name, parameter in model.named_parameters():
                if parameter.grad is None or name not in baseline_gradients:
                    continue
                reference = baseline_gradients[name].float()
                candidate = parameter.grad.detach().contiguous().cpu().float()
                delta = candidate - reference
                delta_square_sum = float(torch.sum(delta * delta))
                reference_square_sum = float(torch.sum(reference * reference))
                delta_l2_squared += delta_square_sum
                reference_l2_squared += reference_square_sum
                rows.append(
                    {
                        "name": name,
                        "numel": delta.numel(),
                        "delta_l2": delta_square_sum**0.5,
                        "delta_rms": (delta_square_sum / delta.numel()) ** 0.5,
                        "delta_max_abs": float(torch.max(torch.abs(delta))),
                        "delta_signed_mean": float(torch.mean(delta)),
                        "reference_l2": reference_square_sum**0.5,
                    }
                )
            rows.sort(key=lambda row: (-row["delta_l2"], row["name"]))
            gradient_error = {
                "global_delta_l2": delta_l2_squared**0.5,
                "global_reference_l2": reference_l2_squared**0.5,
                "relative_global_l2": (
                    (delta_l2_squared / reference_l2_squared) ** 0.5
                    if reference_l2_squared
                    else None
                ),
                "nonzero_parameters": sum(row["delta_l2"] > 0 for row in rows),
                "top_parameters_by_delta_l2": rows[:64],
            }
        runs.append(
            {
                "repeat": repeat,
                "loss_exact": bool(torch.equal(loss.detach(), baseline_loss)),
                "all_parameter_gradients_exact": (
                    digest == baseline_digest
                    and parameter_digests == baseline_parameter_digests
                ),
                "gradient_digest": digest,
                "differing_parameter_count": len(differing_parameters),
                "differing_parameters": differing_parameters,
                "gradient_error": gradient_error,
            }
        )

    captured = capture.as_dict()
    bridge = captured["cross_phase_runtime_bridge"]
    graph_gates = {
        "forward_graph_present": captured["phase_graph_counts"]["FORWARD"] == 1,
        "backward_graph_present": captured["phase_graph_counts"]["BACKWARD"] == 1,
        "all_losses_exact": all(row["loss_exact"] for row in runs),
        "all_parameter_gradients_exact": all(
            row["all_parameter_gradients_exact"] for row in runs
        ),
        "fixed_grid_normalization_loss_exact": normalization_loss_exact,
        "fixed_grid_normalization_gradients_exact": normalization_gradients_exact,
        "all_call_function_nodes_have_seq_nr": all(
            node.get("seq_nr") is not None
            for graph in captured["graphs"]
            for node in graph["nodes"]
            if node["op"] == "call_function"
        ),
        "backward_forward_origin_metadata_recorded": all(
            "fwd_source_fn_stack" in node
            and "seq_nr" in node
            and "is_gradient_acc" in node
            for graph in captured["graphs"]
            if graph["phase"] == "BACKWARD"
            for node in graph["nodes"]
            if node["op"] == "call_function"
        ),
        "cross_phase_runtime_identity_complete": (
            bridge["run_count"] == args.runtime_repeats
            and all(bridge["gates"].values())
        ),
    }
    payload = {
        "schema": "kernel-analyzer.round2-vl-aot-capture.v1",
        "status": (
            "STABLE_AOT_FORWARD_BACKWARD_CAPTURE"
            if all(graph_gates.values())
            else "UNRESOLVED"
        ),
        "scope": "one natural Qwen3-VL multimodal loss forward/backward step",
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": sha256_file(args.model / "config.json"),
        },
        "input": {
            **input_metadata,
            "image_path": str(args.image.resolve()),
            "image_sha256": sha256_file(args.image),
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "dtype": args.dtype,
            "attention": "eager",
            "tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "gates": graph_gates,
        "fixed_grid_normalization": specialization,
        "runtime_runs": runs,
        "capture": captured,
        "claim_boundary": {
            "supported": "stable functionalized AOT forward/backward graphs with exact runtime identity bridge",
            "not_yet_supported": [
                "complete per-node mathematical derivation",
                "numerical bias verdict",
                "candidate implementation correctness",
                "property induction",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "phase_graph_counts": captured["phase_graph_counts"],
                "node_counts": [
                    {
                        "phase": graph["phase"],
                        "nodes": graph["node_count"],
                        "call_function": graph["call_function_count"],
                    }
                    for graph in captured["graphs"]
                ],
                "gates": graph_gates,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
