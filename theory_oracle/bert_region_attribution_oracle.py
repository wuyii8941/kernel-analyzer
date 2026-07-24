#!/usr/bin/env python
"""Segmented BERT prefix/suffix repair-injection attribution at fixed boundaries."""

from __future__ import annotations

import argparse
import gc
import json
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
)
from theory_oracle.transition_attribution_oracle import (
    ARM_ORDER,
    capture_gradient_cpu,
    interaction_effect,
    tensor_fingerprint,
    vector_effect,
)


SCHEMA_VERSION = "forkcert.bert-region-attribution-oracle.v1"
BOUNDARIES = {"embeddings": -1, "encoder_layer_0": 0, "encoder_layer_1": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--anchor-states", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--boundary", choices=sorted(BOUNDARIES), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_segments(
    torch: Any, model: Any, boundary_index: int
) -> tuple[Callable[..., tuple[Any, Any]], Callable[..., tuple[Any, Any]]]:
    layers = model.bert.encoder.layer
    if boundary_index >= len(layers):
        raise ValueError(f"boundary {boundary_index} outside {len(layers)} encoder layers")

    def prefix(input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model.bert.embeddings(input_ids=input_ids)
            prepared_mask, _ = model.bert._create_attention_masks(
                attention_mask=attention_mask,
                encoder_attention_mask=None,
                embedding_output=hidden,
                encoder_hidden_states=None,
                past_key_values=None,
            )
            for index in range(boundary_index + 1):
                hidden = layers[index](hidden, prepared_mask, None)
        return hidden, prepared_mask

    def suffix(hidden: Any, prepared_mask: Any, labels: Any) -> tuple[Any, Any]:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for index in range(boundary_index + 1, len(layers)):
                hidden = layers[index](hidden, prepared_mask, None)
            pooled = model.bert.pooler(hidden)
            pooled = model.dropout(pooled)
            logits = model.classifier(pooled)
        logits = logits.float()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss, logits

    return prefix, suffix


def warm_arm(
    torch: Any,
    model: Any,
    prefix: Callable[..., tuple[Any, Any]],
    suffix: Callable[..., tuple[Any, Any]],
    inputs: tuple[Any, ...],
) -> None:
    model.zero_grad(set_to_none=True)
    hidden, mask = prefix(inputs[0], inputs[1])
    loss, logits = suffix(hidden, mask, inputs[2])
    if not bool(torch.isfinite(loss).all().item()) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("non-finite segmented warm output")
    loss.backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)


def execute_arm(
    torch: Any,
    model: Any,
    prefix: Callable[..., tuple[Any, Any]],
    suffix: Callable[..., tuple[Any, Any]],
    inputs: tuple[Any, ...],
    parameters: list[tuple[str, Any]],
    audit: CompileAudit,
    arm: str,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    before = audit.runtime_invocations
    started = time.perf_counter_ns()
    hidden, mask = prefix(inputs[0], inputs[1])
    loss, logits = suffix(hidden, mask, inputs[2])
    if not bool(torch.isfinite(loss).all().item()) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"non-finite output in {arm}")
    loss.backward()
    torch.cuda.synchronize()
    gradients, gradient_fingerprint = capture_gradient_cpu(torch, parameters)
    return {
        "arm": arm,
        "loss": float(loss.detach().item()),
        "logits_fingerprint": tensor_fingerprint(torch, logits),
        "boundary_fingerprint": tensor_fingerprint(torch, hidden),
        "gradient_fingerprint": gradient_fingerprint,
        "gradients": gradients,
        "compiled_runtime_invocations": audit.runtime_invocations - before,
        "elapsed_ns": time.perf_counter_ns() - started,
    }


def state_bootstrap_ci(
    rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float], draws: int, seed: int
) -> list[float]:
    primary = {row["state_id"]: float(getter(row)) for row in rows if row["repeat"] == 0}
    keys = sorted(primary)
    rng = random.Random(seed)
    estimates = [mean([primary[rng.choice(keys)] for _ in keys]) for _ in range(draws)]
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


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
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 4

    subject_args = argparse.Namespace(
        subject="bert_sst2",
        model_path=args.model_path,
        data_path=args.data_path,
        start=0,
        count=args.count,
        repeats=args.repeats,
        sequence_length=args.sequence_length,
        minibatch_size=1,
        learning_rate=args.learning_rate,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    model, _core, states, subject_metadata = load_subject(torch, subject_args)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    eager_prefix, eager_suffix = build_segments(torch, model, BOUNDARIES[args.boundary])
    audit = CompileAudit()
    compiled_prefix = torch.compile(
        eager_prefix,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )
    compiled_suffix = torch.compile(
        eager_suffix,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )
    anchor_rows = {
        row["state_id"]: row
        for row in read_jsonl(Path(args.anchor_states))
        if int(row["repeat"]) == 0
    }
    missing = [state.state_id for state in states if state.state_id not in anchor_rows]
    if missing:
        raise ValueError(f"missing formal anchors: {missing}")

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
        "data_artifact": {"path": str(Path(args.data_path).resolve())},
        "anchor_artifact": {
            "path": str(Path(args.anchor_states).resolve()),
            "sha256": sha256_file(Path(args.anchor_states)),
        },
        "composition": {
            "A_reference": "eager prefix + eager suffix",
            "I_value_injection": "compiled prefix + eager suffix",
            "R_value_repair": "eager prefix + compiled suffix",
            "B_candidate": "compiled prefix + compiled suffix",
            "boundary": args.boundary,
            "warning": "segmentation may alter fusion/scheduling; monolithic parity is audited",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    warm_arm(torch, model, compiled_prefix, compiled_suffix, states[0].inputs)
    warm_arm(torch, model, eager_prefix, eager_suffix, states[0].inputs)
    warmup_runtime_invocations = audit.runtime_invocations
    warmup_backend_compiles = audit.backend_compiles

    rows: list[dict[str, Any]] = []
    baseline_fingerprints: dict[tuple[str, str], dict[str, Any]] = {}
    arm_functions = {
        "A_reference": (eager_prefix, eager_suffix),
        "I_value_injection": (compiled_prefix, eager_suffix),
        "R_value_repair": (eager_prefix, compiled_suffix),
        "B_candidate": (compiled_prefix, compiled_suffix),
    }
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as state_handle, (
        out_dir / "blocks.jsonl"
    ).open("w", encoding="utf-8") as block_handle:
        for state_index, state in enumerate(states):
            for repeat in range(args.repeats):
                order = ARM_ORDER if repeat % 2 == 0 else list(reversed(ARM_ORDER))
                outputs = {}
                for arm in order:
                    prefix, suffix = arm_functions[arm]
                    outputs[arm] = execute_arm(
                        torch, model, prefix, suffix, state.inputs, parameters, audit, arm
                    )
                effects = {}
                all_blocks = []
                pairs = {
                    "AB_segmented_total": ("A_reference", "B_candidate"),
                    "AI_compiled_prefix_injection": ("A_reference", "I_value_injection"),
                    "AR_eager_prefix_repair": ("A_reference", "R_value_repair"),
                    "RB_compiled_prefix_effect_under_compiled_suffix": ("R_value_repair", "B_candidate"),
                    "IB_compiled_suffix_effect_under_compiled_prefix": ("I_value_injection", "B_candidate"),
                }
                for name, (left, right) in pairs.items():
                    effect, blocks = vector_effect(
                        torch,
                        parameters,
                        outputs[left]["gradients"],
                        outputs[right]["gradients"],
                        state.state_id,
                        repeat,
                        name,
                        emit_blocks=repeat == 0,
                    )
                    effects[name] = effect
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
                        "logits_sha256": outputs[arm]["logits_fingerprint"]["sha256"],
                        "gradient_sha256": outputs[arm]["gradient_fingerprint"]["sha256"],
                    }
                    key = (state.state_id, arm)
                    if repeat == 0:
                        baseline_fingerprints[key] = current
                        exact = True
                    else:
                        exact = current == baseline_fingerprints[key]
                    self_details[arm] = {"metric_exact": exact, **current}

                total = effects["AB_segmented_total"]["effect_l2"]
                anchor = anchor_rows[state.state_id]
                anchor_gradient = float(anchor["gradient_difference_l2"])
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "boundary": args.boundary,
                    "state_id": state.state_id,
                    "state_index": state_index,
                    "state_metadata": state.metadata,
                    "repeat": repeat,
                    "execution_order": order,
                    "arm_losses": {arm: outputs[arm]["loss"] for arm in ARM_ORDER},
                    "arm_runtime_invocations": {
                        arm: outputs[arm]["compiled_runtime_invocations"] for arm in ARM_ORDER
                    },
                    "self_pair_details": self_details,
                    "self_pair_metric_exact": all(value["metric_exact"] for value in self_details.values()),
                    "effects": effects,
                    "compiled_prefix_injection_over_segmented_total": effects[
                        "AI_compiled_prefix_injection"
                    ]["effect_l2"]
                    / total
                    if total
                    else None,
                    "residual_after_eager_prefix_repair_over_segmented_total": effects[
                        "AR_eager_prefix_repair"
                    ]["effect_l2"]
                    / total
                    if total
                    else None,
                    "interaction_over_segmented_total": interaction["effect_l2"] / total if total else None,
                    "monolithic_anchor": {
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
                del outputs
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

    primary = [row for row in rows if row["repeat"] == 0]
    names = [
        "AB_segmented_total",
        "AI_compiled_prefix_injection",
        "AR_eager_prefix_repair",
        "RB_compiled_prefix_effect_under_compiled_suffix",
        "IB_compiled_suffix_effect_under_compiled_prefix",
        "interaction_B_minus_I_minus_R_plus_A",
    ]
    summary_effects = {}
    for index, name in enumerate(names):
        values = [float(row["effects"][name]["effect_l2"]) for row in primary]
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
    candidate_failures = [
        row
        for row in rows
        if int(row["arm_runtime_invocations"]["I_value_injection"]) <= 0
        or int(row["arm_runtime_invocations"]["R_value_repair"]) <= 0
        or int(row["arm_runtime_invocations"]["B_candidate"]) < 2
    ]
    measurement_compiles = audit.backend_compiles - warmup_backend_compiles
    summary = {
        "schema_version": SCHEMA_VERSION,
        "boundary": args.boundary,
        "sampling": {"states": len(states), "repeats": args.repeats, "bootstrap_unit": "SST-2 state"},
        "validity": {
            "candidate_calls_valid": not candidate_failures,
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
            "mean_compiled_prefix_injection_over_segmented_total": mean(
                [float(row["compiled_prefix_injection_over_segmented_total"]) for row in primary]
            ),
            "mean_residual_after_eager_prefix_repair_over_segmented_total": mean(
                [float(row["residual_after_eager_prefix_repair_over_segmented_total"]) for row in primary]
            ),
            "mean_interaction_over_segmented_total": mean(
                [float(row["interaction_over_segmented_total"]) for row in primary]
            ),
        },
        "monolithic_anchor_integrity": {
            "max_reference_loss_abs_error": max(
                row["monolithic_anchor"]["reference_loss_abs_error"] for row in primary
            ),
            "max_candidate_loss_abs_error": max(
                row["monolithic_anchor"]["candidate_loss_abs_error"] for row in primary
            ),
            "max_gradient_difference_l2_abs_error": max(
                row["monolithic_anchor"]["gradient_difference_l2_abs_error"] for row in primary
            ),
            "max_gradient_difference_l2_relative_error": max(
                row["monolithic_anchor"]["gradient_difference_l2_relative_error"] for row in primary
            ),
            "reference_exact": all(
                row["monolithic_anchor"]["reference_loss_abs_error"] == 0.0 for row in primary
            ),
            "candidate_segmented_endpoint_exact": all(
                row["monolithic_anchor"]["candidate_loss_abs_error"] == 0.0
                and row["monolithic_anchor"]["gradient_difference_l2_abs_error"] == 0.0
                for row in primary
            ),
        },
        "claim_scope": {
            "supported": "segmented BERT prefix/suffix intervention-dependent attribution",
            "not_supported": [
                "unique source-operator causal effect",
                "original monolithic graph attribution when candidate parity fails",
                "additive contribution percentages",
            ],
        },
    }
    if candidate_failures or self_nonzero or measurement_compiles:
        raise RuntimeError(
            f"validity failure: candidate={len(candidate_failures)}, self={len(self_nonzero)}, "
            f"measurement_compiles={measurement_compiles}"
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

