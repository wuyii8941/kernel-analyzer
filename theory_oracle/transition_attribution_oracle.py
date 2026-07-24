#!/usr/bin/env python
"""Four-arm output-boundary value/Jacobian attribution for transition endpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from theory_oracle.matched_transition_oracle import (
    CompileAudit,
    artifact_manifest,
    load_subject,
    make_tracking_backend,
    mean,
    quantile,
    sha256_file,
    sha256_text,
)


SCHEMA_VERSION = "forkcert.transition-attribution-oracle.v1"
ARM_ORDER = ["A_reference", "I_value_injection", "R_value_repair", "B_candidate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=["bert_sst2", "qwen_causal"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--anchor-states", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_tensor(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"non-finite tensor in {label}")


def tensor_fingerprint(torch: Any, value: Any) -> dict[str, Any]:
    work = value.detach().float().reshape(-1)
    payload = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sum": float(work.sum(dtype=torch.float64).item()),
        "square_sum": float((work * work).sum(dtype=torch.float64).item()),
        "max_abs": float(work.abs().max().item()),
        "first": float(work[0].item()),
        "middle": float(work[work.numel() // 2].item()),
        "last": float(work[-1].item()),
    }
    payload["sha256"] = sha256_text(json.dumps(payload, sort_keys=True))
    return payload


def capture_gradient_cpu(
    torch: Any, parameters: list[tuple[str, Any]]
) -> tuple[list[Any | None], dict[str, Any]]:
    gradients: list[Any | None] = []
    total_sum = 0.0
    total_square = 0.0
    max_abs = 0.0
    weighted_sample = 0.0
    nonzero = 0
    missing = 0
    for index, (_name, parameter) in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            gradients.append(None)
            missing += 1
            continue
        finite_tensor(torch, gradient, "gradient")
        item = gradient.detach().float().cpu().clone()
        gradients.append(item)
        flat = item.reshape(-1)
        total_sum += float(flat.sum(dtype=torch.float64).item())
        total_square += float((flat * flat).sum(dtype=torch.float64).item())
        max_abs = max(max_abs, float(flat.abs().max().item()))
        weighted_sample += (index + 1) * float(
            flat[0].item() + flat[flat.numel() // 2].item() + flat[-1].item()
        )
        nonzero += int((flat != 0).sum().item())
    payload = {
        "sum": total_sum,
        "square_sum": total_square,
        "max_abs": max_abs,
        "weighted_sample": weighted_sample,
        "nonzero": nonzero,
        "missing_parameters": missing,
    }
    payload["sha256"] = sha256_text(json.dumps(payload, sort_keys=True))
    return gradients, payload


def build_boundary_core(
    torch: Any, model: Any, subject: str
) -> tuple[Callable[..., tuple[Any, Any]], Callable[[Any, tuple[Any, ...]], Any]]:
    if subject == "bert_sst2":

        def loss_from_boundary(boundary: Any, inputs: tuple[Any, ...]) -> Any:
            return torch.nn.functional.cross_entropy(boundary.float(), inputs[2])

        def core(input_ids: Any, attention_mask: Any, labels: Any) -> tuple[Any, Any]:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
            logits = logits.float()
            return torch.nn.functional.cross_entropy(logits, labels), logits

        return core, loss_from_boundary

    def loss_from_boundary(boundary: Any, inputs: tuple[Any, ...]) -> Any:
        input_ids, _attention_mask, response_mask = inputs
        targets = input_ids[:, 1:]
        target_logps = torch.nn.functional.log_softmax(boundary.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        return -(target_logps * response_mask).sum() / response_mask.sum()

    def core(input_ids: Any, attention_mask: Any, response_mask: Any) -> tuple[Any, Any]:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
        boundary = logits[:, :-1, :].float()
        return loss_from_boundary(boundary, (input_ids, attention_mask, response_mask)), boundary

    return core, loss_from_boundary


def warm_path(
    torch: Any,
    model: Any,
    function: Callable[..., tuple[Any, Any]],
    inputs: tuple[Any, ...],
) -> None:
    model.zero_grad(set_to_none=True)
    loss, boundary = function(*inputs)
    finite_tensor(torch, loss, "warm loss")
    finite_tensor(torch, boundary, "warm boundary")
    loss.backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)


def execute_arm(
    torch: Any,
    model: Any,
    function: Callable[..., tuple[Any, Any]],
    loss_from_boundary: Callable[[Any, tuple[Any, ...]], Any],
    inputs: tuple[Any, ...],
    parameters: list[tuple[str, Any]],
    audit: CompileAudit,
    arm: str,
    target_boundary: Any | None,
    keep_boundary: bool,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    before = audit.runtime_invocations
    started = time.perf_counter_ns()
    base_loss, boundary = function(*inputs)
    if target_boundary is None:
        used_boundary = boundary
        loss = base_loss
        boundary_exact = True
    else:
        # Forward bits equal target_boundary while autograd follows boundary.
        used_boundary = target_boundary.detach() + (boundary - boundary.detach())
        boundary_exact = bool(torch.equal(used_boundary.detach(), target_boundary.detach()))
        loss = loss_from_boundary(used_boundary, inputs)
    finite_tensor(torch, loss, f"{arm} loss")
    finite_tensor(torch, used_boundary, f"{arm} boundary")
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter_ns() - started
    gradients, gradient_fingerprint = capture_gradient_cpu(torch, parameters)
    result = {
        "arm": arm,
        "loss": float(loss.detach().item()),
        "base_loss": float(base_loss.detach().item()),
        "boundary_exact": boundary_exact,
        "boundary_fingerprint": tensor_fingerprint(torch, used_boundary),
        "native_boundary_fingerprint": tensor_fingerprint(torch, boundary),
        "gradient_fingerprint": gradient_fingerprint,
        "gradients": gradients,
        "compiled_runtime_invocations": audit.runtime_invocations - before,
        "elapsed_ns": elapsed,
        "boundary": used_boundary.detach().clone() if keep_boundary else None,
    }
    return result


def vector_effect(
    torch: Any,
    parameters: list[tuple[str, Any]],
    left: list[Any | None],
    right: list[Any | None],
    state_id: str,
    repeat: int,
    effect_name: str,
    emit_blocks: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left_square = 0.0
    right_square = 0.0
    diff_square = 0.0
    dot = 0.0
    max_abs = 0.0
    exact = True
    blocks = []
    for index, ((name, parameter), left_item, right_item) in enumerate(
        zip(parameters, left, right, strict=True)
    ):
        if left_item is None and right_item is None:
            continue
        if left_item is None:
            left_item = torch.zeros_like(right_item)
        if right_item is None:
            right_item = torch.zeros_like(left_item)
        difference = right_item - left_item
        ls = float((left_item * left_item).sum(dtype=torch.float64).item())
        rs = float((right_item * right_item).sum(dtype=torch.float64).item())
        ds = float((difference * difference).sum(dtype=torch.float64).item())
        block_max = float(difference.abs().max().item())
        left_square += ls
        right_square += rs
        diff_square += ds
        dot += float((left_item * right_item).sum(dtype=torch.float64).item())
        max_abs = max(max_abs, block_max)
        exact = exact and bool(torch.equal(left_item, right_item))
        if emit_blocks:
            blocks.append(
                {
                    "state_id": state_id,
                    "repeat": repeat,
                    "effect": effect_name,
                    "parameter_index": index,
                    "parameter_name": name,
                    "coordinates": parameter.numel(),
                    "left_l2": math.sqrt(ls),
                    "right_l2": math.sqrt(rs),
                    "effect_l2": math.sqrt(ds),
                    "max_abs_effect": block_max,
                }
            )
    left_norm = math.sqrt(left_square)
    right_norm = math.sqrt(right_square)
    effect_norm = math.sqrt(diff_square)
    denominator = left_norm * right_norm
    return {
        "effect_l2": effect_norm,
        "relative_to_left_l2": effect_norm / left_norm if left_norm else None,
        "left_l2": left_norm,
        "right_l2": right_norm,
        "cosine_similarity": dot / denominator if denominator else None,
        "max_abs_effect": max_abs,
        "exact_equal": exact,
    }, blocks


def interaction_effect(
    torch: Any,
    parameters: list[tuple[str, Any]],
    arms: dict[str, dict[str, Any]],
    state_id: str,
    repeat: int,
    emit_blocks: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_square = 0.0
    max_abs = 0.0
    blocks = []
    for index, (name, parameter) in enumerate(parameters):
        items = []
        for arm in ARM_ORDER:
            item = arms[arm]["gradients"][index]
            if item is None:
                template = next(
                    candidate["gradients"][index]
                    for candidate in arms.values()
                    if candidate["gradients"][index] is not None
                )
                item = torch.zeros_like(template)
            items.append(item)
        a, i, r, b = items
        interaction = b - i - r + a
        square = float((interaction * interaction).sum(dtype=torch.float64).item())
        block_max = float(interaction.abs().max().item())
        total_square += square
        max_abs = max(max_abs, block_max)
        if emit_blocks:
            blocks.append(
                {
                    "state_id": state_id,
                    "repeat": repeat,
                    "effect": "interaction_B_minus_I_minus_R_plus_A",
                    "parameter_index": index,
                    "parameter_name": name,
                    "coordinates": parameter.numel(),
                    "effect_l2": math.sqrt(square),
                    "max_abs_effect": block_max,
                }
            )
    return {"effect_l2": math.sqrt(total_square), "max_abs_effect": max_abs}, blocks


def state_bootstrap_ci(
    rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float], draws: int, seed: int
) -> list[float]:
    primary = {row["state_id"]: float(getter(row)) for row in rows if row["repeat"] == 0}
    keys = sorted(primary)
    rng = random.Random(seed)
    values = [mean([primary[rng.choice(keys)] for _ in keys]) for _ in range(draws)]
    return [quantile(values, 0.025), quantile(values, 0.975)]


def repeated_variance(
    rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float]
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(float(getter(row)))
    return mean([statistics.variance(values) if len(values) > 1 else 0.0 for values in grouped.values()])


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("at least two repeats are required")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; fail instead of falling back to CPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 2

    model, _transition_core, states, subject_metadata = load_subject(torch, args)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    eager_core, loss_from_boundary = build_boundary_core(torch, model, args.subject)
    audit = CompileAudit()
    compiled_core = torch.compile(
        eager_core,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )
    anchor_rows = {
        row["state_id"]: row
        for row in read_jsonl(Path(args.anchor_states))
        if int(row["repeat"]) == 0
    }
    missing_anchors = [state.state_id for state in states if state.state_id not in anchor_rows]
    if missing_anchors:
        raise ValueError(f"missing formal transition anchors: {missing_anchors}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_unix": time.time(),
        "arguments": vars(args),
        "subject_metadata": subject_metadata,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "model_artifact": artifact_manifest(Path(args.model_path)),
        "data_artifact": {
            "path": str(Path(args.data_path).resolve()),
            "file_sha256": sha256_file(Path(args.data_path)) if Path(args.data_path).is_file() else None,
        },
        "anchor_artifact": {
            "path": str(Path(args.anchor_states).resolve()),
            "sha256": sha256_file(Path(args.anchor_states)),
        },
        "factorial": {
            "A_reference": "eager value, eager Jacobian",
            "I_value_injection": "compiled value, eager Jacobian",
            "R_value_repair": "eager value, compiled Jacobian",
            "B_candidate": "compiled value, compiled Jacobian",
            "mixed_arm_semantics": "target.detach() + (native - native.detach())",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    warm_path(torch, model, compiled_core, states[0].inputs)
    warm_path(torch, model, eager_core, states[0].inputs)
    warmup_runtime_invocations = audit.runtime_invocations
    warmup_backend_compiles = audit.backend_compiles

    rows: list[dict[str, Any]] = []
    baseline_fingerprints: dict[tuple[str, str], dict[str, Any]] = {}
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as state_handle, (
        out_dir / "blocks.jsonl"
    ).open("w", encoding="utf-8") as block_handle:
        for state_index, state in enumerate(states):
            for repeat in range(args.repeats):
                outputs: dict[str, dict[str, Any]] = {}
                base_order = ["A_reference", "B_candidate"] if repeat % 2 == 0 else ["B_candidate", "A_reference"]
                for arm in base_order:
                    function = eager_core if arm == "A_reference" else compiled_core
                    outputs[arm] = execute_arm(
                        torch,
                        model,
                        function,
                        loss_from_boundary,
                        state.inputs,
                        parameters,
                        audit,
                        arm,
                        target_boundary=None,
                        keep_boundary=True,
                    )
                eager_boundary = outputs["A_reference"]["boundary"]
                compiled_boundary = outputs["B_candidate"]["boundary"]
                mixed_order = ["I_value_injection", "R_value_repair"] if repeat % 2 == 0 else ["R_value_repair", "I_value_injection"]
                for arm in mixed_order:
                    function = eager_core if arm == "I_value_injection" else compiled_core
                    target = compiled_boundary if arm == "I_value_injection" else eager_boundary
                    outputs[arm] = execute_arm(
                        torch,
                        model,
                        function,
                        loss_from_boundary,
                        state.inputs,
                        parameters,
                        audit,
                        arm,
                        target_boundary=target,
                        keep_boundary=False,
                    )

                effects = {}
                all_blocks = []
                pairs = {
                    "AB_candidate_total": ("A_reference", "B_candidate"),
                    "AI_value_injection_under_eager_jacobian": ("A_reference", "I_value_injection"),
                    "AR_compiled_jacobian_at_eager_value": ("A_reference", "R_value_repair"),
                    "RB_value_effect_under_compiled_jacobian": ("R_value_repair", "B_candidate"),
                    "IB_compiled_jacobian_at_compiled_value": ("I_value_injection", "B_candidate"),
                }
                for effect_name, (left, right) in pairs.items():
                    effect, blocks = vector_effect(
                        torch,
                        parameters,
                        outputs[left]["gradients"],
                        outputs[right]["gradients"],
                        state.state_id,
                        repeat,
                        effect_name,
                        emit_blocks=repeat == 0,
                    )
                    effects[effect_name] = effect
                    all_blocks.extend(blocks)
                interaction, interaction_blocks = interaction_effect(
                    torch,
                    parameters,
                    outputs,
                    state.state_id,
                    repeat,
                    emit_blocks=repeat == 0,
                )
                effects["interaction_B_minus_I_minus_R_plus_A"] = interaction
                all_blocks.extend(interaction_blocks)
                for block in all_blocks:
                    block_handle.write(json.dumps(block, sort_keys=True, allow_nan=False) + "\n")
                block_handle.flush()

                self_details = {}
                for arm in ARM_ORDER:
                    current = {
                        "loss": outputs[arm]["loss"],
                        "boundary_sha256": outputs[arm]["boundary_fingerprint"]["sha256"],
                        "gradient_sha256": outputs[arm]["gradient_fingerprint"]["sha256"],
                    }
                    key = (state.state_id, arm)
                    if repeat == 0:
                        baseline_fingerprints[key] = current
                        exact = True
                    else:
                        exact = current == baseline_fingerprints[key]
                    self_details[arm] = {"metric_exact": exact, **current}

                total = effects["AB_candidate_total"]["effect_l2"]
                anchor = anchor_rows[state.state_id]
                anchor_gradient = float(anchor["gradient_difference_l2"])
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "subject": args.subject,
                    "state_id": state.state_id,
                    "state_index": state_index,
                    "state_metadata": state.metadata,
                    "repeat": repeat,
                    "execution_order": base_order + mixed_order,
                    "arm_losses": {arm: outputs[arm]["loss"] for arm in ARM_ORDER},
                    "arm_compiled_runtime_invocations": {
                        arm: outputs[arm]["compiled_runtime_invocations"] for arm in ARM_ORDER
                    },
                    "mixed_boundary_exact": {
                        "I_value_injection": outputs["I_value_injection"]["boundary_exact"],
                        "R_value_repair": outputs["R_value_repair"]["boundary_exact"],
                    },
                    "self_pair_details": self_details,
                    "self_pair_metric_exact": all(value["metric_exact"] for value in self_details.values()),
                    "effects": effects,
                    "value_injection_over_total": effects[
                        "AI_value_injection_under_eager_jacobian"
                    ]["effect_l2"]
                    / total
                    if total
                    else None,
                    "residual_after_value_repair_over_total": effects[
                        "AR_compiled_jacobian_at_eager_value"
                    ]["effect_l2"]
                    / total
                    if total
                    else None,
                    "interaction_over_total": interaction["effect_l2"] / total if total else None,
                    "formal_anchor": {
                        "reference_loss_abs_error": abs(
                            outputs["A_reference"]["loss"] - float(anchor["reference_loss"])
                        ),
                        "candidate_loss_abs_error": abs(
                            outputs["B_candidate"]["loss"] - float(anchor["candidate_loss"])
                        ),
                        "gradient_difference_l2_abs_error": abs(total - anchor_gradient),
                        "gradient_difference_l2_relative_error": abs(total - anchor_gradient)
                        / anchor_gradient
                        if anchor_gradient
                        else (0.0 if total == 0 else None),
                    },
                }
                state_handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                state_handle.flush()
                rows.append(row)
                del outputs, eager_boundary, compiled_boundary
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

    primary = [row for row in rows if row["repeat"] == 0]

    def effect_values(name: str) -> list[float]:
        return [float(row["effects"][name]["effect_l2"]) for row in primary]

    summary_effects = {}
    for index, name in enumerate(
        [
            "AB_candidate_total",
            "AI_value_injection_under_eager_jacobian",
            "AR_compiled_jacobian_at_eager_value",
            "RB_value_effect_under_compiled_jacobian",
            "IB_compiled_jacobian_at_compiled_value",
            "interaction_B_minus_I_minus_R_plus_A",
        ]
    ):
        values = effect_values(name)
        summary_effects[name] = {
            "mean_l2": mean(values),
            "max_l2": max(values),
            "state_bootstrap_mean_l2_95ci": state_bootstrap_ci(
                rows,
                lambda row, key=name: float(row["effects"][key]["effect_l2"]),
                args.bootstrap,
                args.seed + index + 1,
            ),
            "same_state_repeat_variance": repeated_variance(
                rows, lambda row, key=name: float(row["effects"][key]["effect_l2"])
            ),
        }

    self_nonzero = [row for row in rows if row["repeat"] > 0 and not row["self_pair_metric_exact"]]
    mixed_failures = [
        row
        for row in rows
        if not all(bool(value) for value in row["mixed_boundary_exact"].values())
    ]
    candidate_failures = [
        row
        for row in rows
        if int(row["arm_compiled_runtime_invocations"]["B_candidate"]) <= 0
        or int(row["arm_compiled_runtime_invocations"]["R_value_repair"]) <= 0
    ]
    measurement_compiles = audit.backend_compiles - warmup_backend_compiles
    summary = {
        "schema_version": SCHEMA_VERSION,
        "subject": args.subject,
        "sampling": {
            "states": len(states),
            "repeats": args.repeats,
            "bootstrap_unit": "matched state/minibatch",
        },
        "validity": {
            "candidate_calls_valid": not candidate_failures,
            "mixed_boundary_exact": not mixed_failures,
            "self_pair_metric_nonzero_count": len(self_nonzero),
            "backend_compiles": audit.backend_compiles,
            "backend_compiles_during_measurement": measurement_compiles,
            "no_graph_proliferation_after_warmup": measurement_compiles == 0,
            "measurement_runtime_invocations": audit.runtime_invocations - warmup_runtime_invocations,
            "graph_code_sha256": audit.graph_code_sha256,
            "graph_node_counts": audit.graph_node_counts,
        },
        "effects": summary_effects,
        "ratios": {
            "mean_value_injection_over_total": mean(
                [float(row["value_injection_over_total"]) for row in primary]
            ),
            "mean_residual_after_value_repair_over_total": mean(
                [float(row["residual_after_value_repair_over_total"]) for row in primary]
            ),
            "mean_interaction_over_total": mean(
                [float(row["interaction_over_total"]) for row in primary]
            ),
        },
        "formal_anchor_integrity": {
            "max_reference_loss_abs_error": max(
                row["formal_anchor"]["reference_loss_abs_error"] for row in primary
            ),
            "max_candidate_loss_abs_error": max(
                row["formal_anchor"]["candidate_loss_abs_error"] for row in primary
            ),
            "max_gradient_difference_l2_abs_error": max(
                row["formal_anchor"]["gradient_difference_l2_abs_error"] for row in primary
            ),
            "max_gradient_difference_l2_relative_error": max(
                row["formal_anchor"]["gradient_difference_l2_relative_error"] for row in primary
            ),
            "exact_monolithic_endpoint_preserved": all(
                row["formal_anchor"]["reference_loss_abs_error"] == 0.0
                and row["formal_anchor"]["candidate_loss_abs_error"] == 0.0
                and row["formal_anchor"]["gradient_difference_l2_abs_error"] == 0.0
                for row in primary
            ),
        },
        "claim_scope": {
            "supported": "output-boundary value/Jacobian factorial under the declared stop-gradient intervention",
            "not_supported": [
                "unique operator root cause",
                "necessity or sufficiency without interaction assumptions",
                "historical optimizer replay",
                "mathematical/compiler correctness failure",
            ],
        },
    }
    if candidate_failures or mixed_failures or self_nonzero or measurement_compiles:
        raise RuntimeError(
            "attribution validity failure: "
            f"candidate={len(candidate_failures)}, mixed={len(mixed_failures)}, "
            f"self={len(self_nonzero)}, measurement_compiles={measurement_compiles}"
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

