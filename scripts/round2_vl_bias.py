#!/usr/bin/env python3
"""Cross-state directional-bias and carrier test for the round-2 SiLU case."""

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


STATES = (
    {
        "id": "token_drop",
        "image": "/data1/tzh/DFuzz/DFuzz/megatron-lm-new/docs/source/images/moe/token_drop.png",
        "question": "What mechanism is illustrated in this model diagram?",
        "answer": "It illustrates token dropping and routing in a mixture-of-experts model.",
    },
    {
        "id": "multi_token_prediction",
        "image": "/data1/tzh/DFuzz/DFuzz/megatron-lm-new/docs/source/images/multi_token_prediction/MTP_implementation.png",
        "question": "What training design is presented in this diagram?",
        "answer": "It presents a multi-token prediction design for language-model training.",
    },
    {
        "id": "optimizer_sharding",
        "image": "/data1/tzh/DFuzz/DFuzz/megatron-lm-new/docs/source/images/distrib_optimizer/sharding_scheme.png",
        "question": "What distributed-training concept does this figure show?",
        "answer": "It shows how optimizer state is sharded across distributed workers.",
    },
    {
        "id": "fsdp_allreduce",
        "image": "/data1/tzh/DFuzz/DFuzz/megatron-lm-new/docs/source/images/custom_fsdp/FSDP_Allreduce.png",
        "question": "What communication operation is described by this figure?",
        "answer": "It describes gradient communication and reduction in a sharded training system.",
    },
    {
        "id": "context_parallel",
        "image": "/data1/tzh/DFuzz/DFuzz/megatron-lm-new/docs/source/images/context_parallel/CP_overview.png",
        "question": "What parallel-computing layout is summarized here?",
        "answer": "It summarizes a context-parallel layout for transformer computation.",
    },
    {
        "id": "optimizer_dataflow",
        "image": "/data1/tzh/DFuzz/DFuzz/megatron-lm-new/docs/source/images/distrib_optimizer/data_flow.png",
        "question": "What process is represented by this technical diagram?",
        "answer": "It represents data flow through a distributed optimizer during training.",
    },
)

ENDPOINTS = (
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.2.mlp.gate_proj.weight",
    "model.visual.patch_embed.proj.weight",
)


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().contiguous().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


class LocalSiluRecorder:
    state_id: str = ""
    rows: list[dict[str, Any]] = []
    sum_delta: dict[int, torch.Tensor] = {}
    sum_norm2: dict[int, float] = {}
    count: dict[int, int] = {}

    @classmethod
    def reset_state(cls, state_id: str) -> None:
        cls.state_id = state_id

    @classmethod
    def record(
        cls,
        layer: int,
        decomposed: torch.Tensor,
        fused: torch.Tensor,
    ) -> None:
        delta = (decomposed.float() - fused.float()).detach().cpu()
        norm2 = float(torch.sum(delta * delta))
        cls.rows.append(
            {
                "state_id": cls.state_id,
                "layer": layer,
                "delta_l2": math.sqrt(norm2),
                "delta_signed_mean": float(torch.mean(delta)),
                "delta_max_abs": float(torch.max(torch.abs(delta))),
            }
        )
        if layer not in cls.sum_delta:
            cls.sum_delta[layer] = torch.zeros_like(delta)
            cls.sum_norm2[layer] = 0.0
            cls.count[layer] = 0
        cls.sum_delta[layer].add_(delta)
        cls.sum_norm2[layer] += norm2
        cls.count[layer] += 1


class _MeasuredDecomposedSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, layer: int) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.layer = int(layer)
        return torch.ops.aten.silu.default(value)

    @staticmethod
    def backward(ctx: Any, cotangent: torch.Tensor) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        sigmoid = torch.ops.aten.sigmoid.default(value)
        one = torch.empty_like(sigmoid)
        one.fill_(1)
        one_minus = torch.ops.aten.sub.Tensor(one, sigmoid)
        scaled = torch.ops.aten.mul.Tensor(value, one_minus)
        plus_one = torch.ops.aten.add.Scalar(scaled, 1)
        derivative = torch.ops.aten.mul.Tensor(sigmoid, plus_one)
        decomposed = torch.ops.aten.mul.Tensor(cotangent, derivative)
        fused = torch.ops.aten.silu_backward.default(cotangent, value)
        LocalSiluRecorder.record(ctx.layer, decomposed, fused)
        return decomposed, None


class MeasuredDecomposedSiluModule(torch.nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _MeasuredDecomposedSilu.apply(value, self.layer)


def _run(
    model: torch.nn.Module,
    call: Any,
    arguments: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = call(*arguments)
    loss.backward()
    return loss.detach().cpu(), _gradients(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--pad-length", type=int, default=160)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    prepared = []
    for state in STATES:
        inputs, labels, metadata = prepare_step(
            processor,
            Path(state["image"]),
            width=args.width,
            height=args.height,
            question=state["question"],
            answer=state["answer"],
            pad_length=args.pad_length,
        )
        prepared.append((state, inputs, labels, metadata))
    visual_positions = [
        torch.nonzero(
            row[1]["input_ids"][0]
            == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            as_tuple=False,
        ).flatten()
        for row in prepared
    ]
    if not all(torch.equal(visual_positions[0], value) for value in visual_positions[1:]):
        raise RuntimeError("visual token positions differ across frozen states")
    if not all(row[1]["input_ids"].shape == prepared[0][1]["input_ids"].shape for row in prepared):
        raise RuntimeError("state shapes are not identical after padding")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    first_inputs = {key: value.to(device) for key, value in prepared[0][1].items()}
    specialization = specialize_fixed_grid(
        model,
        first_inputs["image_grid_thw"],
        first_inputs["input_ids"],
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
            position_ids: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            return self.subject(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
                return_dict=False,
            )[0]

    step = LossStep(model)
    capture = AOTForwardBackwardCapture()
    compiled = torch.compile(
        step, backend=capture.backend(), fullgraph=True, dynamic=False
    )

    sum_delta: dict[str, torch.Tensor] = {}
    sum_error_norm2 = 0.0
    state_rows = []
    candidate_digests: dict[str, dict[str, str]] = {}
    endpoint_deltas: dict[str, list[torch.Tensor]] = {
        name: [] for name in ENDPOINTS
    }
    device_states = []
    for state, inputs, labels, metadata in prepared:
        values = {key: value.to(device) for key, value in inputs.items()}
        device_labels = labels.to(device)
        positions, _ = model.model.get_rope_index(
            values["input_ids"],
            values["image_grid_thw"],
            attention_mask=values["attention_mask"],
        )
        call_args = (
            values["input_ids"],
            values["attention_mask"],
            values["pixel_values"],
            values["image_grid_thw"],
            positions,
            device_labels,
        )
        device_states.append((state, call_args, metadata))
        eager_loss, eager = _run(model, step, call_args)
        model.zero_grad(set_to_none=True)
        candidate_loss = compiled(*call_args)
        capture.bind_user_outputs(candidate_loss)
        candidate_loss.register_hook(capture.bind_user_cotangent)
        candidate_loss.backward()
        candidate = _gradients(model)
        candidate_digests[state["id"]] = {
            name: _digest(value) for name, value in candidate.items()
        }
        error_norm2 = reference_norm2 = dot_reference = 0.0
        signed_sum = 0.0
        numel = 0
        parameter_rows = []
        for name in sorted(eager):
            reference = eager[name].float()
            delta = candidate[name].float() - reference
            dn2 = float(torch.sum(delta * delta))
            rn2 = float(torch.sum(reference * reference))
            error_norm2 += dn2
            reference_norm2 += rn2
            dot_reference += float(torch.sum(delta * reference))
            signed_sum += float(torch.sum(delta))
            numel += delta.numel()
            if name not in sum_delta:
                sum_delta[name] = torch.zeros_like(delta)
            sum_delta[name].add_(delta)
            if name in endpoint_deltas:
                endpoint_deltas[name].append(delta.clone())
            parameter_rows.append(
                {
                    "name": name,
                    "delta_l2": math.sqrt(dn2),
                    "delta_signed_mean": float(torch.mean(delta)),
                    "reference_l2": math.sqrt(rn2),
                }
            )
        sum_error_norm2 += error_norm2
        parameter_rows.sort(key=lambda row: (-row["delta_l2"], row["name"]))
        state_rows.append(
            {
                "state_id": state["id"],
                "input": {
                    **metadata,
                    "image_sha256": sha256_file(Path(state["image"])),
                },
                "eager_loss": float(eager_loss),
                "candidate_loss": float(candidate_loss.detach().cpu()),
                "loss_exact": bool(torch.equal(eager_loss, candidate_loss.detach().cpu())),
                "error_l2": math.sqrt(error_norm2),
                "reference_l2": math.sqrt(reference_norm2),
                "relative_error_l2": math.sqrt(error_norm2 / reference_norm2),
                "error_reference_cosine": dot_reference / math.sqrt(error_norm2 * reference_norm2),
                "global_error_signed_mean": signed_sum / numel,
                "top_parameters": parameter_rows[:32],
            }
        )
        del eager, candidate

    state_count = len(state_rows)
    sum_vector_norm2 = sum(
        float(torch.sum(value * value)) for value in sum_delta.values()
    )
    cross_inner = (
        (sum_vector_norm2 - sum_error_norm2) / (state_count * (state_count - 1))
    )
    average_error_norm2 = sum_error_norm2 / state_count

    endpoint_rows = []
    for name, deltas in endpoint_deltas.items():
        if not deltas:
            continue
        endpoint_sum = torch.stack(deltas).sum(dim=0)
        endpoint_norm_sum = sum(float(torch.sum(delta * delta)) for delta in deltas)
        endpoint_cross = (
            (float(torch.sum(endpoint_sum * endpoint_sum)) - endpoint_norm_sum)
            / (state_count * (state_count - 1))
        )
        endpoint_rows.append(
            {
                "name": name,
                "cross_state_error_inner_product": endpoint_cross,
                "average_state_error_norm2": endpoint_norm_sum / state_count,
                "coherence_ratio": endpoint_cross / (endpoint_norm_sum / state_count) if endpoint_norm_sum else None,
                "state_signed_means": [float(torch.mean(delta)) for delta in deltas],
            }
        )

    patched = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Qwen3VLTextMLP":
            layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
            module.act_fn = MeasuredDecomposedSiluModule(layer)
            patched.append(name)
    if len(patched) != 28:
        raise RuntimeError(f"expected 28 text SiLU modules, found {len(patched)}")
    intervention_rows = []
    for state, call_args, _ in device_states:
        LocalSiluRecorder.reset_state(state["id"])
        loss, gradients = _run(model, step, call_args)
        exact = sum(
            _digest(value) == candidate_digests[state["id"]][name]
            for name, value in gradients.items()
        )
        intervention_rows.append(
            {
                "state_id": state["id"],
                "loss": float(loss),
                "candidate_exact_parameter_count": exact,
                "parameter_count": len(gradients),
            }
        )
        del gradients

    local_layers = []
    for layer in sorted(LocalSiluRecorder.sum_delta):
        count = LocalSiluRecorder.count[layer]
        norm_sum = LocalSiluRecorder.sum_norm2[layer]
        total = LocalSiluRecorder.sum_delta[layer]
        local_cross = (
            (float(torch.sum(total * total)) - norm_sum) / (count * (count - 1))
            if count > 1
            else 0.0
        )
        local_layers.append(
            {
                "layer": layer,
                "state_count": count,
                "cross_state_error_inner_product": local_cross,
                "average_state_error_norm2": norm_sum / count,
                "coherence_ratio": local_cross / (norm_sum / count) if norm_sum else None,
                "state_signed_means": [
                    row["delta_signed_mean"]
                    for row in LocalSiluRecorder.rows
                    if row["layer"] == layer
                ],
            }
        )

    payload = {
        "schema": "kernel-analyzer.round2-vl-bias.v1",
        "status": "COMPLETE_CROSS_STATE_CAUSAL_CAMPAIGN",
        "state_count": state_count,
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": sha256_file(args.model / "config.json"),
            "dtype": "bfloat16",
        },
        "protocol": {
            "candidate": "AOTAutograd no-op backend",
            "reference": "normalized eager",
            "forward_intervention": "none",
            "backward_intervention": "replace fused SiLU VJP with the exact AOT BF16 decomposition",
            "fixed_shape": [1, args.pad_length],
            "fixed_grid_specialization": specialization,
            "tf32": False,
            "deterministic_algorithms": True,
        },
        "states": state_rows,
        "global_direction": {
            "cross_state_error_inner_product": cross_inner,
            "average_state_error_norm2": average_error_norm2,
            "coherence_ratio": cross_inner / average_error_norm2,
            "sum_error_vector_norm2": sum_vector_norm2,
        },
        "endpoints": endpoint_rows,
        "intervention": intervention_rows,
        "local_silu_vjp": {
            "rows": LocalSiluRecorder.rows,
            "layers": local_layers,
        },
        "verdict": {
            "candidate_is_exactly_mediated_by_silu_backward_decomposition": all(row["candidate_exact_parameter_count"] == row["parameter_count"] for row in intervention_rows),
            "cross_state_error_has_positive_mean_pairwise_inner_product": cross_inner > 0,
            "all_losses_exact": all(row["loss_exact"] for row in state_rows),
            "property_stage_entered": False,
        },
        "claim_boundary": {
            "supported": "six-state natural-input directional and causal mediation evidence for the AOT SiLU backward decomposition case",
            "not_supported": [
                "optimizer-trajectory effect",
                "generated Inductor/Triton kernel correctness",
                "property induction",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"global_direction": payload["global_direction"], "verdict": payload["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
