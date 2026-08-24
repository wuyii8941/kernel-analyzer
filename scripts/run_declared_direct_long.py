#!/usr/bin/env python3
"""Run 4096-step warm-state direct-update tests for declared historical cases.

The runner preserves each case's exact implementation/repair boundary while
using one common output schema.  It advances a common natural parameter and
AdamW state, then compares candidate and repair at that same state.  This is a
direct-effect experiment; feedback and convergence are deliberately not
inferred from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_analyzer.persistence_property import path_statistics_from_gram  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_qwen128_softmax_saved_p_trajectory import SavedProbabilityRepair  # noqa: E402
from scripts.run_qwen128_vproj_repair import VProjRepair  # noqa: E402
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver  # noqa: E402


Tree = dict[str, torch.Tensor]


CASE_CONFIG = {
    "qwen_lmhead_dx": {
        "case_id": "qwen_seq128_lmhead_dx",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "architecture": "qwen",
        "carriers": ["model.norm.weight"],
        "learning_rate": 1e-4,
        "contrast": "compiled BF16 lm_head dX MM vs FP32 MM + BF16 ABI cast",
    },
    "liger_fused_ce": {
        "case_id": "liger_fused_ce_t128",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "architecture": "qwen",
        "carriers": ["model.embed_tokens.weight"],
        "learning_rate": 1e-4,
        "contrast": "Liger fused CE BF16 dW accumulation vs FP32 accumulation",
    },
    "mamba_in_proj": {
        "case_id": "mamba_seq64_in_proj",
        "model": "/data1/tzh/models/state-spaces/mamba-130m-hf",
        "architecture": "mamba",
        "carriers": ["backbone.layers.0.mixer.in_proj.weight"],
        "learning_rate": 1e-5,
        "contrast": "compiled in_proj BF16 MM vs FP32 MM + BF16 ABI cast",
        "target_sha": "9c03ef3fc9b93005efed225a176c3e97efa91d33af2fae0b27cbb28e695c3cee",
    },
    "qwen_saved_p": {
        "case_id": "qwen_saved_p_seq128",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "architecture": "qwen",
        "carriers": [
            "model.layers.27.self_attn.q_proj.weight",
            "model.layers.27.self_attn.k_proj.weight",
        ],
        "learning_rate": 1e-5,
        "contrast": "compiled reconstructed saved-P vs true-forward-P softmax VJP",
    },
}


def tree_clone(values: Tree) -> Tree:
    return {name: value.detach().float().clone() for name, value in values.items()}


def tree_zeros(values: Tree) -> Tree:
    return {name: torch.zeros_like(value) for name, value in values.items()}


def tree_sub(left: Tree, right: Tree) -> Tree:
    return {name: left[name] - right[name] for name in left}


def tree_norm(values: Tree) -> float:
    square = sum(torch.sum(value.double().square()) for value in values.values())
    return float(torch.sqrt(square).item())


def tree_dot(left: Tree, right: Tree) -> float:
    return float(sum(
        torch.sum(left[name].double() * right[name].double()) for name in left
    ).item())


def tree_add_(target: Tree, source: Tree) -> None:
    for name in target:
        target[name].add_(source[name])


def structured_signed_bucket_sketch(
    values: Tree, *, dimension: int, seed: int
) -> torch.Tensor:
    """Project every coordinate through a fixed signed modulo-bucket map.

    This is a full-coordinate linear sketch, not a fitted carrier.  The exact
    resultant and energy are accumulated separately and remain the headline
    measurement; the sketch is used only for lag/null/rank diagnostics.
    """

    output = torch.zeros(dimension, device=next(iter(values.values())).device, dtype=torch.float64)
    for name, value in values.items():
        flat = value.detach().reshape(-1).float()
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        local_seed = (seed ^ name_seed) % (2**63 - 1)
        generator = torch.Generator(device=flat.device)
        generator.manual_seed(local_seed)
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
            body = flat[: blocks * dimension].reshape(blocks, dimension)
            output.add_(torch.mv(body.T, row_sign).double() * column_sign.double())
        tail = flat[blocks * dimension:]
        if tail.numel():
            output[: tail.numel()].add_(tail.double() * column_sign[: tail.numel()].double())
    return output


class LongAccumulator:
    def __init__(self, template: Tree, *, steps: int, window: int, sketch_dim: int) -> None:
        self.steps = steps
        self.window = window
        self.sketch_dim = sketch_dim
        self.total = tree_zeros(template)
        self.window_total = tree_zeros(template)
        self.energy = 0.0
        self.window_energy = 0.0
        self.sketches: list[np.ndarray] = []
        self.windows: list[dict[str, Any]] = []

    def add(self, value: Tree, step: int) -> dict[str, float]:
        tree_add_(self.total, value)
        tree_add_(self.window_total, value)
        energy = tree_dot(value, value)
        self.energy += energy
        self.window_energy += energy
        sketch = structured_signed_bucket_sketch(
            value, dimension=self.sketch_dim, seed=20_260_824
        )
        self.sketches.append(sketch.cpu().numpy().astype(np.float32))
        result = {
            "l2": math.sqrt(max(energy, 0.0)),
            "cumulative_A": tree_norm(self.total) / math.sqrt(max(self.energy, 1e-30)),
        }
        if step % self.window == 0:
            self.windows.append({
                "steps": [step - self.window + 1, step],
                "coherence_amplification": (
                    tree_norm(self.window_total) / math.sqrt(max(self.window_energy, 1e-30))
                ),
                "resultant_l2": tree_norm(self.window_total),
                "diffusive_scale_l2": math.sqrt(max(self.window_energy, 0.0)),
            })
            for value_ in self.window_total.values():
                value_.zero_()
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
        resolved_output = output.resolve()
        try:
            artifact_path = str(resolved_output.relative_to(ROOT.resolve()))
        except ValueError:
            artifact_path = str(resolved_output)
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
                "artifact": artifact_path,
                "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
        }


def update_tree(
    gradients: Tree, first: Tree, second: Tree, step: int, learning_rate: float
) -> tuple[Tree, Tree, Tree]:
    updates: Tree = {}
    next_first: Tree = {}
    next_second: Tree = {}
    for name in gradients:
        update, m, v = adam_delta(
            gradients[name], first[name], second[name], step,
            learning_rate=learning_rate, beta1=0.9, beta2=0.95,
        )
        updates[name] = update
        next_first[name] = m
        next_second[name] = v
    return updates, next_first, next_second


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_CONFIG), required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--measurement-steps", type=int, default=4096)
    parser.add_argument("--window-steps", type=int, default=32)
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sketch-output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.measurement_steps < 32 or args.measurement_steps % args.window_steps:
        raise ValueError("measurement horizon must contain complete windows")
    if args.sketch_dim < 256:
        raise ValueError("sketch dimension is too small")

    config = CASE_CONFIG[args.case]
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    required = args.warmup_steps + args.measurement_steps
    if len(states) < required:
        raise RuntimeError(f"input bank has {len(states)} rows; {required} required")
    states = states[:required]
    state_ids = [str(row.get("state_id", row.get("sequence_id", i))) for i, row in enumerate(states)]
    if len(set(state_ids)) != len(state_ids):
        raise RuntimeError("long-run state IDs are not unique")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    configure_candidate_runtime(24_000)
    model = load_model(config["architecture"], Path(config["model"]), device)
    model.eval()
    named = dict(model.named_parameters())
    carriers = {name: named[name] for name in config["carriers"]}
    # The saved-P and external-MM repairs are bound to the actual compiled
    # backward program.  Freezing unrelated parameters changes that program
    # and invalidates the endpoint identity, so retain the original autograd
    # population for those cases.  The simpler lm-head/Liger boundaries may
    # safely stop at their declared carriers.
    if args.case in {"qwen_lmhead_dx", "liger_fused_ce"}:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in carriers)

    candidate: Any | None = None
    modules: list[Any] = []
    candidate_loss: Any | None = None
    repair_loss: Any | None = None
    if args.case == "liger_fused_ce":
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        if carriers[config["carriers"][0]].untyped_storage().data_ptr() != model.lm_head.weight.untyped_storage().data_ptr():
            raise RuntimeError("Liger carrier is not tied to lm_head")
        candidate_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=None,
        ).to(device)
        repair_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
        ).to(device)
    else:
        start = len(PyCodeCache.modules)
        candidate = torch.compile(
            LossStep(model), backend="inductor",
            fullgraph=args.case == "qwen_saved_p", dynamic=False,
        )
        warm_values = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        candidate(warm_values).backward()
        torch.cuda.synchronize(device)
        modules = list(PyCodeCache.modules[start:])

    master = {name: parameter.detach().float().clone() for name, parameter in carriers.items()}
    first = tree_zeros(master)
    second = tree_zeros(master)

    def set_master(values: Tree) -> None:
        with torch.no_grad():
            for name, parameter in carriers.items():
                parameter.copy_(values[name].to(parameter.dtype))

    def gradient(state: dict[str, Any], *, repair: bool, seed: int) -> tuple[str, float, Tree, dict[str, Any]]:
        set_master(master)
        tokens = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer: Any | None = None
        if args.case == "liger_fused_ce":
            hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
            labels = torch.nn.functional.pad(tokens, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
            module = repair_loss if repair else candidate_loss
            loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
            loss.backward()
            local = {"changed_coordinates": -1, "endpoint_error_l2": None}
        else:
            if repair:
                if args.case == "qwen_lmhead_dx":
                    observer = ShapeObserver(
                        modules, "fp32", [], left_shape=(128, 151936),
                        right_shape=(151936, 2048),
                    )
                elif args.case == "mamba_in_proj":
                    observer = VProjRepair(modules, "REPAIR_FP32_CAST_BF16", config["target_sha"])
                elif args.case == "qwen_saved_p":
                    observer = SavedProbabilityRepair(modules, "REPAIR_SAVED_P")
                else:
                    raise AssertionError(args.case)
            if observer is None:
                loss = candidate(tokens)
                loss.backward()
            else:
                with observer:
                    loss = candidate(tokens)
                    loss.backward()
            if observer is None:
                local = {"changed_coordinates": 0, "endpoint_error_l2": None}
            elif args.case == "qwen_lmhead_dx":
                local = {
                    "changed_coordinates": -1,
                    "endpoint_error_l2": observer.changed_l2,
                }
            elif args.case == "mamba_in_proj":
                if observer.local is None:
                    raise RuntimeError("Mamba repair did not expose its endpoint")
                local = {
                    "changed_coordinates": observer.local["changed_coordinates"],
                    "endpoint_error_l2": observer.local["l2_intervention"],
                }
            else:
                local = {
                    "changed_coordinates": observer.changed_coordinates,
                    "endpoint_error_l2": observer.correction_l2,
                }
        torch.cuda.synchronize(device)
        gradients = {}
        for name, parameter in carriers.items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"missing/nonfinite carrier gradient: {name}")
            gradients[name] = parameter.grad.detach().float().clone()
        return tensor_digest(loss), float(loss.detach().float().item()), gradients, local

    warmup_losses: list[float] = []
    for index, state in enumerate(states[: args.warmup_steps]):
        step = index + 1
        _, loss_value, grad, _ = gradient(state, repair=False, seed=40_000 + index)
        update, first, second = update_tree(
            grad, first, second, step, config["learning_rate"]
        )
        tree_add_(master, update)
        warmup_losses.append(loss_value)
        if not args.quiet and (step == 1 or step % 32 == 0):
            print(json.dumps({"event": "DECLARED_LONG_WARMUP", "case": args.case, "step": step, "loss": loss_value}), flush=True)
        del grad, update

    accumulator = LongAccumulator(
        master, steps=args.measurement_steps, window=args.window_steps,
        sketch_dim=args.sketch_dim,
    )
    rows: list[dict[str, Any]] = []
    measured_ids: list[str] = []
    for index, state in enumerate(states[args.warmup_steps:required]):
        step = index + 1
        optimizer_step = args.warmup_steps + step
        state_id = state_ids[args.warmup_steps + index]
        natural_digest, loss_value, grad_n, _ = gradient(
            state, repair=False, seed=50_000 + index
        )
        repair_digest, repair_loss_value, grad_r, local = gradient(
            state, repair=True, seed=50_000 + index
        )
        if args.case in {"qwen_lmhead_dx", "qwen_saved_p", "liger_fused_ce"} and natural_digest != repair_digest:
            raise RuntimeError("backward-only repair changed forward loss")
        update_n, next_first, next_second = update_tree(
            grad_n, first, second, optimizer_step, config["learning_rate"]
        )
        update_r, _, _ = update_tree(
            grad_r, first, second, optimizer_step, config["learning_rate"]
        )
        direct = tree_sub(update_n, update_r)
        path = accumulator.add(direct, step)
        tree_add_(master, update_n)
        first, second = next_first, next_second
        measured_ids.append(state_id)
        rows.append({
            "measurement_step": step,
            "optimizer_step": optimizer_step,
            "state_id": state_id,
            "candidate_loss": loss_value,
            "repair_same_state_loss": repair_loss_value,
            "loss_gap": loss_value - repair_loss_value,
            "direct_update_l2": path["l2"],
            "cumulative_A": path["cumulative_A"],
            "endpoint_changed_coordinates": local["changed_coordinates"],
            "endpoint_error_l2": local["endpoint_error_l2"],
        })
        if not args.quiet and (step == 1 or step % 32 == 0):
            print(json.dumps({"event": "DECLARED_LONG_MEASUREMENT", "case": args.case, **rows[-1]}), flush=True)
        del grad_n, grad_r, update_n, update_r, direct

    sketch_output = args.sketch_output or args.output.with_suffix(".sketch.npz")
    statistics = accumulator.finalize(measured_ids, output=sketch_output)
    payload = {
        "schema": "kernel-analyzer-declared-direct-long-v1",
        "status": "COMPLETE",
        "case_id": config["case_id"],
        "protocol": {
            "input_bank": str(args.input_bank.resolve()),
            "unique_natural_states": required,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "window_steps": args.window_steps,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": config["learning_rate"],
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "initial_moments": "ZERO_THEN_WARMED_ON_NATURAL_IMPLEMENTATION",
            },
            "carriers": config["carriers"],
            "other_parameters_updated": False,
            "unrelated_parameters_require_grad_for_graph_identity": (
                args.case in {"mamba_in_proj", "qwen_saved_p"}
            ),
            "contrast": config["contrast"],
            "state_path": "NATURAL_IMPLEMENTATION_ADVANCES_COMMON_MASTER_AND_MOMENTS",
        },
        "warmup": {
            "first_loss": warmup_losses[0],
            "last_loss": warmup_losses[-1],
            "mean_loss": float(np.mean(warmup_losses)),
        },
        "statistics": statistics,
        "rows": rows,
        "claim_boundary": (
            "Warm-state same-state direct effective-update experiment on the declared "
            "parameter scope. It tests 4096-step direct directionality. It does not "
            "measure feedback, full-parameter training, or convergence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    exact = statistics["exact_full_coordinate"]
    print(json.dumps({
        "event": "DECLARED_DIRECT_LONG_COMPLETE",
        "case": args.case,
        "output": str(args.output),
        "A4096": exact["coherence_amplification"],
        "late_windows_above_one": sum(
            row["coherence_amplification"] > 1.0
            for row in exact["rolling_windows"][len(exact["rolling_windows"]) // 2:]
        ),
        "late_window_count": len(exact["rolling_windows"]) // 2,
    }), flush=True)


if __name__ == "__main__":
    main()
