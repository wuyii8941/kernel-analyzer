#!/usr/bin/env python3
"""Run the declared layer-23 attention-state repair for 4096 direct steps.

This preserves the historical seq1024 joint S_bwd/K repair.  A natural arm
advances one q_proj master parameter and its AdamW moments.  At every measured
state, natural and repaired gradients are evaluated at that same parameter and
moment state.  The result therefore measures the direct implementation effect;
it does not attribute closed-loop feedback or training convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OLD_SRC = ROOT / "archive" / "round1_code" / "src"
for path in (OLD_SRC, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/tile_causal_compile")

import numpy as np
import torch
import torch.nn.functional as F
from torch._dynamo.backends.registry import lookup_backend
from torch._inductor.codecache import PyCodeCache
from torch._inductor.select_algorithm import extern_kernels
import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

from kernel_analyzer.persistence_property import path_statistics_from_gram
from scripts.long_horizon_trigger import atomic_json, build_model, load_milestone, under_root


CASE_ID = "layer23_qproj_attention_state_region"
PARAMETER = "model.layers.23.self_attn.q_proj.weight"
HISTORICAL_SOURCE_SHA256 = "58955a5274f7aa8c635a5273590d37388a2664022ca813727e23a322216ac3f3"
PARAMETER_TREE = dict[str, torch.Tensor]


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    result = hashlib.sha256()
    result.update(str(tensor.dtype).encode())
    result.update(repr(tuple(tensor.shape)).encode())
    result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def tree_zeros(values: PARAMETER_TREE) -> PARAMETER_TREE:
    return {name: torch.zeros_like(value) for name, value in values.items()}


def tree_sub(left: PARAMETER_TREE, right: PARAMETER_TREE) -> PARAMETER_TREE:
    return {name: left[name] - right[name] for name in left}


def tree_dot(left: PARAMETER_TREE, right: PARAMETER_TREE) -> float:
    return float(sum(
        torch.sum(left[name].double() * right[name].double()) for name in left
    ).item())


def tree_norm(values: PARAMETER_TREE) -> float:
    return math.sqrt(max(tree_dot(values, values), 0.0))


def tree_add_(target: PARAMETER_TREE, source: PARAMETER_TREE) -> None:
    for name in target:
        target[name].add_(source[name])


def update_tree(
    gradients: PARAMETER_TREE,
    first: PARAMETER_TREE,
    second: PARAMETER_TREE,
    step: int,
    learning_rate: float,
) -> tuple[PARAMETER_TREE, PARAMETER_TREE, PARAMETER_TREE]:
    beta1, beta2, epsilon = 0.9, 0.95, 1e-8
    updates: PARAMETER_TREE = {}
    next_first: PARAMETER_TREE = {}
    next_second: PARAMETER_TREE = {}
    for name, gradient in gradients.items():
        m = first[name] * beta1 + gradient * (1.0 - beta1)
        v = second[name] * beta2 + gradient.square() * (1.0 - beta2)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        updates[name] = -learning_rate * m_hat / (v_hat.sqrt() + epsilon)
        next_first[name] = m
        next_second[name] = v
    return updates, next_first, next_second


def signed_bucket_sketch(
    values: PARAMETER_TREE, *, dimension: int, seed: int,
) -> torch.Tensor:
    output = torch.zeros(
        dimension, device=next(iter(values.values())).device, dtype=torch.float64,
    )
    for name, value in values.items():
        flat = value.detach().reshape(-1).float()
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        generator = torch.Generator(device=flat.device)
        generator.manual_seed((seed ^ name_seed) % (2**63 - 1))
        column_sign = torch.randint(
            0, 2, (dimension,), generator=generator, device=flat.device,
            dtype=torch.int8,
        ).to(torch.float32).mul_(2).sub_(1)
        blocks = flat.numel() // dimension
        if blocks:
            row_sign = torch.randint(
                0, 2, (blocks,), generator=generator, device=flat.device,
                dtype=torch.int8,
            ).to(torch.float32).mul_(2).sub_(1)
            output.add_(
                torch.mv(flat[:blocks * dimension].reshape(blocks, dimension).T, row_sign).double()
                * column_sign.double()
            )
        tail = flat[blocks * dimension:]
        if tail.numel():
            output[:tail.numel()].add_(tail.double() * column_sign[:tail.numel()].double())
    return output


class LongAccumulator:
    def __init__(
        self, template: PARAMETER_TREE, *, steps: int, window: int, sketch_dim: int,
    ) -> None:
        self.steps = steps
        self.window = window
        self.sketch_dim = sketch_dim
        self.total = tree_zeros(template)
        self.window_total = tree_zeros(template)
        self.energy = 0.0
        self.window_energy = 0.0
        self.sketches: list[np.ndarray] = []
        self.windows: list[dict[str, Any]] = []

    def add(self, value: PARAMETER_TREE, step: int) -> dict[str, float]:
        tree_add_(self.total, value)
        tree_add_(self.window_total, value)
        energy = tree_dot(value, value)
        self.energy += energy
        self.window_energy += energy
        self.sketches.append(
            signed_bucket_sketch(
                value, dimension=self.sketch_dim, seed=20_260_824,
            ).cpu().numpy().astype(np.float32)
        )
        result = {
            "l2": math.sqrt(max(energy, 0.0)),
            "cumulative_A": tree_norm(self.total) / math.sqrt(max(self.energy, 1e-30)),
        }
        if step % self.window == 0:
            self.windows.append({
                "steps": [step - self.window + 1, step],
                "coherence_amplification": (
                    tree_norm(self.window_total)
                    / math.sqrt(max(self.window_energy, 1e-30))
                ),
                "resultant_l2": tree_norm(self.window_total),
                "diffusive_scale_l2": math.sqrt(max(self.window_energy, 0.0)),
            })
            for tensor in self.window_total.values():
                tensor.zero_()
            self.window_energy = 0.0
        return result

    def finalize(self, state_ids: list[str], *, output: Path) -> dict[str, Any]:
        sketches = np.stack(self.sketches).astype(np.float64)
        tensor = torch.from_numpy(sketches)
        gram = (tensor @ tensor.T).numpy()
        sketch_stats = path_statistics_from_gram(
            gram, state_ids=state_ids, max_lag=64,
            sign_flip_draws=1000, seed=20_260_824,
        )
        trace = float(np.trace(gram))
        gram_square = float(np.square(gram).sum())
        effective_rank = trace * trace / max(gram_square, 1e-30)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, direct_update_sketches=sketches.astype(np.float32))
        return {
            "steps": self.steps,
            "exact_full_coordinate": {
                "resultant_l2": tree_norm(self.total),
                "diffusive_scale_l2": math.sqrt(max(self.energy, 0.0)),
                "coherence_amplification": (
                    tree_norm(self.total) / math.sqrt(max(self.energy, 1e-30))
                ),
                "rolling_windows": self.windows,
            },
            "sketch_diagnostics": sketch_stats,
            "sketch_effective_rank_participation_ratio": effective_rank,
            "measurement_geometry": {
                "kind": "EXACT_RESULTANT_ENERGY_PLUS_FULL_COORDINATE_SIGNED_BUCKET_SKETCH",
                "sketch_dimension": self.sketch_dim,
                "fixed_rank_one_projection_used_as_gate": False,
                "artifact": str(output),
                "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--measurement-steps", type=int, default=4096)
    parser.add_argument("--window-steps", type=int, default=32)
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="permit a 32-step engineering run; never use it as the declared result",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 1:
        raise ValueError("warmup must be positive")
    if args.measurement_steps != 4096 and not (
        args.smoke and args.measurement_steps == 32
    ):
        raise ValueError("declared result requires 4096 steps; --smoke permits only 32")
    if args.window_steps != 32 or args.learning_rate != 1e-5:
        raise ValueError("frozen layer-23 protocol requires 32-step windows and lr=1e-5")

    bank_path = under_root(args.bank, "bank")
    input_path = under_root(args.input_bank, "input-bank")
    model_path = under_root(args.model, "model")
    output_path = under_root(args.output, "output")
    states_payload = json.loads(input_path.read_text())
    states = states_payload["states"]
    required = args.warmup_steps + args.measurement_steps
    if states_payload.get("sequence_length") != 1024 or len(states) < required:
        raise RuntimeError("layer-23 long test requires at least 4224 frozen seq1024 states")
    states = states[:required]
    state_ids = [str(row["state_id"]) for row in states]
    if len(set(state_ids)) != required:
        raise RuntimeError("layer-23 state IDs are not unique")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)

    checkpoint_bank = json.loads(bank_path.read_text())
    milestone = next(
        row for row in checkpoint_bank["milestones"]
        if int(row["step"]) == args.checkpoint_step
    )
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)
    parameter = dict(model.named_parameters())[PARAMETER]

    class LossStep(torch.nn.Module):
        def __init__(self, subject: torch.nn.Module) -> None:
            super().__init__()
            self.subject = subject

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return self.subject(
                input_ids=input_ids, labels=labels,
                use_cache=False, return_dict=False,
            )[0]

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend=lookup_backend("inductor"),
        fullgraph=True, dynamic=False,
    )

    def tensors(row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        return values, values

    model.zero_grad(set_to_none=True)
    warm_inputs = tensors(states[0])
    candidate(*warm_inputs).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])
    matches: list[tuple[Path, str]] = []
    for module in modules:
        path = Path(module.__file__)
        source = path.read_text()
        if "bmm_76]" in source and "mm_267" in source:
            matches.append((path, source))
    if len(matches) != 1:
        raise RuntimeError(f"expected one historical bmm_76 source, got {len(matches)}")
    source_path, source = matches[0]
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != HISTORICAL_SOURCE_SHA256:
        raise RuntimeError(
            f"layer-23 historical source drift: {source_sha256} != {HISTORICAL_SOURCE_SHA256}"
        )
    marker = source.index("bmm_76]")
    call_start = source.rfind("def call(", 0, marker)
    target_ordinal = source[call_start:marker].count("extern_kernels.bmm(")
    if target_ordinal != 19:
        raise RuntimeError(f"layer-23 target ordinal drift: {target_ordinal} != 19")

    original_bmm = extern_kernels.bmm
    original_attention = modeling_qwen3.eager_attention_forward
    eager_capture: dict[str, torch.Tensor] = {}

    def captured_attention(
        module: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attention_mask: torch.Tensor | None, scaling: float, dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
        raw_scores = torch.matmul(query, key_states.transpose(2, 3))
        if module.layer_idx == 23:
            eager_capture["K"] = key_states.detach().reshape(16, 1024, 128).clone()

            def raw_hook(gradient: torch.Tensor) -> torch.Tensor:
                eager_capture["S"] = gradient.detach().reshape(16, 1024, 1024).clone()
                return gradient

            raw_scores.register_hook(raw_hook)
        weights = raw_scores * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :key_states.shape[-2]]
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output, weights

    master = parameter.detach().float().clone()
    first = torch.zeros_like(master)
    second = torch.zeros_like(master)

    def set_master() -> None:
        with torch.no_grad():
            parameter.copy_(master.to(parameter.dtype))

    def eager_operands(inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        eager_capture.clear()
        model.zero_grad(set_to_none=True)
        modeling_qwen3.eager_attention_forward = captured_attention
        try:
            loss = model(
                input_ids=inputs[0], labels=inputs[1],
                use_cache=False, return_dict=False,
            )[0]
            loss.backward()
            torch.cuda.synchronize(device)
        finally:
            modeling_qwen3.eager_attention_forward = original_attention
        if set(eager_capture) != {"S", "K"}:
            raise RuntimeError("eager layer-23 S/K capture failed")
        return eager_capture["S"], eager_capture["K"]

    def compiled_gradient(
        inputs: tuple[torch.Tensor, torch.Tensor],
        replacement: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[str, float, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        loss = candidate(*inputs)
        if replacement is None:
            loss.backward()
        else:
            counter = {"bmm": 0}
            observed = {"target": False}

            def wrapped_bmm(*values: Any, **kwargs: Any) -> Any:
                ordinal = counter["bmm"]
                counter["bmm"] += 1
                if ordinal != target_ordinal:
                    return original_bmm(*values, **kwargs)
                out = kwargs.get("out")
                if out is None or tuple(out.shape) != (16, 1024, 128):
                    raise RuntimeError("layer-23 bmm_76 output shape drift")
                observed["target"] = True
                return original_bmm(replacement[0], replacement[1], out=out)

            extern_kernels.bmm = wrapped_bmm
            try:
                loss.backward()
            finally:
                extern_kernels.bmm = original_bmm
            if not observed["target"]:
                raise RuntimeError("layer-23 repair was not observed")
        torch.cuda.synchronize(device)
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError("layer-23 carrier gradient missing or nonfinite")
        return tensor_digest(loss), float(loss.detach().float().item()), parameter.grad.detach().float().clone()

    warmup_losses: list[float] = []
    for index, row in enumerate(states[:args.warmup_steps]):
        set_master()
        _, loss_value, gradient = compiled_gradient(tensors(row), None)
        updates, next_first, next_second = update_tree(
            {PARAMETER: gradient}, {PARAMETER: first}, {PARAMETER: second},
            index + 1, args.learning_rate,
        )
        master.add_(updates[PARAMETER])
        first, second = next_first[PARAMETER], next_second[PARAMETER]
        warmup_losses.append(loss_value)
        if not args.quiet and (index == 0 or (index + 1) % 32 == 0):
            print(json.dumps({"event": "L23_LONG_WARMUP", "step": index + 1}), flush=True)

    accumulator = LongAccumulator(
        {PARAMETER: master}, steps=args.measurement_steps,
        window=args.window_steps, sketch_dim=args.sketch_dim,
    )
    rows: list[dict[str, Any]] = []
    measured_ids: list[str] = []
    measured = states[args.warmup_steps:required]
    for index, row in enumerate(measured):
        step = index + 1
        optimizer_step = args.warmup_steps + step
        set_master()
        inputs = tensors(row)
        natural_digest, natural_loss, gradient_n = compiled_gradient(inputs, None)
        reference_s, reference_k = eager_operands(inputs)
        repair_digest, repair_loss, gradient_r = compiled_gradient(
            inputs, (reference_s, reference_k),
        )
        if natural_digest != repair_digest:
            raise RuntimeError("backward-only layer-23 repair changed forward loss")
        update_n, next_first, next_second = update_tree(
            {PARAMETER: gradient_n}, {PARAMETER: first}, {PARAMETER: second},
            optimizer_step, args.learning_rate,
        )
        update_r, _, _ = update_tree(
            {PARAMETER: gradient_r}, {PARAMETER: first}, {PARAMETER: second},
            optimizer_step, args.learning_rate,
        )
        direct = tree_sub(update_n, update_r)
        path = accumulator.add(direct, step)
        tree_add_({PARAMETER: master}, update_n)
        first, second = next_first[PARAMETER], next_second[PARAMETER]
        measured_ids.append(str(row["state_id"]))
        rows.append({
            "measurement_step": step,
            "optimizer_step": optimizer_step,
            "state_id": str(row["state_id"]),
            "natural_loss": natural_loss,
            "repair_loss": repair_loss,
            "gradient_difference_l2": float(torch.linalg.vector_norm(gradient_n - gradient_r).item()),
            "direct_update_l2": path["l2"],
            "cumulative_A": path["cumulative_A"],
        })
        if not args.quiet and (step == 1 or step % 128 == 0):
            print(json.dumps({"event": "L23_LONG_STEP", "step": step, "A": path["cumulative_A"]}), flush=True)
        del gradient_n, gradient_r, update_n, update_r, direct, reference_s, reference_k

    sketch_output = output_path.with_suffix(".sketch.npz")
    metrics = accumulator.finalize(measured_ids, output=sketch_output)
    windows = metrics["exact_full_coordinate"]["rolling_windows"]
    late = [row for row in windows if row["steps"][0] >= 2049]
    result = {
        "schema": "kernel-analyzer-declared-layer23-direct-long-v1",
        "status": "COMPLETE_4096_DIRECT_EFFECT",
        "case_id": CASE_ID,
        "protocol": {
            "checkpoint_step": args.checkpoint_step,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "window_steps": args.window_steps,
            "optimizer": {
                "name": "AdamW", "learning_rate": args.learning_rate,
                "betas": [0.9, 0.95], "epsilon": 1e-8,
                "weight_decay": 0.0, "moments": "zero then evolved normally",
            },
            "state_path": "natural q_proj master and moments; same-state candidate/repair contrast",
            "parameter_scope": PARAMETER,
            "other_parameters_updated": False,
            "repair_boundary": "actual bmm_76: G_q = S_bwd @ K with joint eager S_bwd/K replacement",
        },
        "binding": {
            "generated_source_sha256": source_sha256,
            "historical_32_step_source_sha256": HISTORICAL_SOURCE_SHA256,
            "byte_identical_to_historical_32_step_source": (
                source_sha256 == HISTORICAL_SOURCE_SHA256
            ),
            "semantic_boundary_matches_historical": True,
            "target_ordinal": target_ordinal,
            "input_bank_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "warmup": {
            "first_loss": warmup_losses[0],
            "last_loss": warmup_losses[-1],
        },
        "metrics": metrics,
        "late_window_summary": {
            "count": len(late),
            "above_one": sum(row["coherence_amplification"] > 1.0 for row in late),
            "mean_A": float(np.mean([row["coherence_amplification"] for row in late])),
        },
        "records": rows,
        "claim_boundary": {
            "supports": "4096-step common-state direct update directionality at the historical exact attention-state repair",
            "does_not_support": ["closed-loop feedback attribution", "full-parameter training", "loss convergence"],
        },
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({
        "event": "L23_DECLARED_DIRECT_LONG_COMPLETE",
        "output": str(output_path),
        "A4096": metrics["exact_full_coordinate"]["coherence_amplification"],
        "late_windows_above_one": result["late_window_summary"]["above_one"],
        "late_window_count": result["late_window_summary"]["count"],
    }), flush=True)


if __name__ == "__main__":
    main()
