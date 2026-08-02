#!/usr/bin/env python3
"""Test whether AOT's decomposed SiLU VJP causes the round-2 gradient delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.round2_vl_smoke import prepare_step, sha256_file  # noqa: E402
from scripts.round2_vl_static import specialize_fixed_grid  # noqa: E402
from scripts.aot_capture import AOTForwardBackwardCapture  # noqa: E402


class _DecomposedSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.ops.aten.silu.default(value)

    @staticmethod
    def backward(ctx: Any, cotangent: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        sigmoid = torch.ops.aten.sigmoid.default(value)
        one = torch.empty_like(sigmoid)
        one.fill_(1)
        one_minus = torch.ops.aten.sub.Tensor(one, sigmoid)
        scaled = torch.ops.aten.mul.Tensor(value, one_minus)
        plus_one = torch.ops.aten.add.Scalar(scaled, 1)
        derivative = torch.ops.aten.mul.Tensor(sigmoid, plus_one)
        return torch.ops.aten.mul.Tensor(cotangent, derivative)


class DecomposedSiluModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _DecomposedSilu.apply(value)


def _clone_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().contiguous().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _run(model: torch.nn.Module, call: Any, arguments: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = call(*arguments)
    loss.backward()
    return loss.detach().cpu(), _clone_gradients(model)


def _compare(
    reference: dict[str, torch.Tensor],
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, Any]:
    names = sorted(reference)
    left_norm2 = right_norm2 = residual_norm2 = dot = 0.0
    exact = 0
    rows = []
    for name in names:
        base = reference[name].float()
        left_value = left[name].float()
        right_value = right[name].float()
        left_delta = left_value - base
        right_delta = right_value - base
        residual = right_value - left_value
        ln2 = float(torch.sum(left_delta * left_delta))
        rn2 = float(torch.sum(right_delta * right_delta))
        en2 = float(torch.sum(residual * residual))
        product = float(torch.sum(left_delta * right_delta))
        left_norm2 += ln2
        right_norm2 += rn2
        residual_norm2 += en2
        dot += product
        is_exact = _digest(left[name]) == _digest(right[name])
        exact += int(is_exact)
        rows.append(
            {
                "name": name,
                "candidate_delta_l2": math.sqrt(ln2),
                "intervention_delta_l2": math.sqrt(rn2),
                "candidate_intervention_residual_l2": math.sqrt(en2),
                "candidate_intervention_exact": is_exact,
            }
        )
    rows.sort(key=lambda row: (-row["candidate_delta_l2"], row["name"]))
    return {
        "parameter_count": len(names),
        "candidate_intervention_exact_parameter_count": exact,
        "candidate_delta_l2": math.sqrt(left_norm2),
        "intervention_delta_l2": math.sqrt(right_norm2),
        "candidate_intervention_residual_l2": math.sqrt(residual_norm2),
        "candidate_intervention_delta_cosine": (
            dot / math.sqrt(left_norm2 * right_norm2)
            if left_norm2 and right_norm2
            else None
        ),
        "top_parameters": rows[:64],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    inputs, labels, input_metadata = prepare_step(
        processor, args.image, width=args.width, height=args.height
    )
    values = {key: value.to(device) for key, value in inputs.items()}
    device_labels = labels.to(device)
    dtype = getattr(torch, args.dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    position_ids, _ = model.model.get_rope_index(
        values["input_ids"],
        values["image_grid_thw"],
        attention_mask=values["attention_mask"],
    )
    specialization = specialize_fixed_grid(
        model, values["image_grid_thw"], values["input_ids"]
    )

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
            positions: torch.Tensor,
            loss_labels: torch.Tensor,
        ) -> torch.Tensor:
            return self.subject(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=positions,
                labels=loss_labels,
                use_cache=False,
                return_dict=False,
            )[0]

    step = LossStep(model)
    arguments = (
        values["input_ids"],
        values["attention_mask"],
        values["pixel_values"],
        values["image_grid_thw"],
        position_ids,
        device_labels,
    )
    eager_loss, eager_gradients = _run(model, step, arguments)

    capture = AOTForwardBackwardCapture()
    compiled = torch.compile(
        step, backend=capture.backend(), fullgraph=True, dynamic=False
    )
    model.zero_grad(set_to_none=True)
    candidate_loss = compiled(*arguments)
    capture.bind_user_outputs(candidate_loss)
    candidate_loss.register_hook(capture.bind_user_cotangent)
    candidate_loss.backward()
    candidate_loss_cpu = candidate_loss.detach().cpu()
    candidate_gradients = _clone_gradients(model)

    patched_modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Qwen3VLTextMLP":
            module.act_fn = DecomposedSiluModule()
            patched_modules.append(name)
    if len(patched_modules) != 28:
        raise RuntimeError(
            f"expected 28 text SiLU modules, found {len(patched_modules)}"
        )
    intervention_loss, intervention_gradients = _run(model, step, arguments)
    comparison = _compare(
        eager_gradients, candidate_gradients, intervention_gradients
    )
    captured = capture.as_dict()
    silu_programs = [
        node["target"]
        for graph in captured["graphs"]
        if graph["phase"] == "BACKWARD"
        for node in graph["nodes"]
        if node.get("fwd_source_fn_stack")
        and any(
            str(frame[0]).startswith("silu")
            for frame in node["fwd_source_fn_stack"]
        )
        and node["op"] == "call_function"
    ]
    payload = {
        "schema": "kernel-analyzer.round2-vl-silu-cause.v1",
        "status": "COMPLETE_CAUSAL_INTERVENTION",
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": sha256_file(args.model / "config.json"),
            "dtype": args.dtype,
        },
        "input": {
            **input_metadata,
            "image_path": str(args.image.resolve()),
            "image_sha256": sha256_file(args.image),
        },
        "fixed_grid_specialization": specialization,
        "intervention": {
            "forward": "unchanged aten.silu",
            "backward": "sigma(x); one-sigma(x); sigma(x)*(1+x*(1-sigma(x))); q times derivative",
            "patched_text_mlp_count": len(patched_modules),
            "patched_module_names": patched_modules,
        },
        "losses": {
            "eager": float(eager_loss),
            "aot": float(candidate_loss_cpu),
            "decomposed_silu_vjp_eager": float(intervention_loss),
            "eager_aot_exact": bool(torch.equal(eager_loss, candidate_loss_cpu)),
            "eager_intervention_exact": bool(torch.equal(eager_loss, intervention_loss)),
        },
        "comparison": comparison,
        "captured_silu_backward_program": silu_programs,
        "claim_boundary": {
            "supported": "one-state causal mediation test that changes only the SiLU backward formula while retaining aten.silu forward",
            "not_supported": [
                "cross-state directional bias",
                "generated-kernel correctness",
                "property induction",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"losses": payload["losses"], "comparison": comparison}, sort_keys=True))


if __name__ == "__main__":
    main()
