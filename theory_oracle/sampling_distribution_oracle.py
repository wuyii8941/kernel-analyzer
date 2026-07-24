#!/usr/bin/env python
"""Exact categorical-distribution and RNG-calibration Oracle for Qwen sampling."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from theory_oracle.real_model_oracle import (
    CompileAudit,
    artifact_manifest,
    load_subject,
    make_tracking_backend,
    mean,
    quantile,
    sha256_file,
)


SCHEMA_VERSION = "forkcert.sampling-distribution-oracle.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=166)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model-repeats", type=int, default=2)
    parser.add_argument("--rng-replicates", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1024)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def tensor_exact(torch: Any, left: Any, right: Any) -> bool:
    return bool(torch.equal(left, right))


def execute(
    torch: Any,
    function: Any,
    inputs: tuple[Any, ...],
    position: int,
    audit: CompileAudit,
    path: str,
) -> dict[str, Any]:
    before = audit.runtime_invocations
    started = time.perf_counter_ns()
    with torch.no_grad():
        logits = function(*inputs)
    torch.cuda.synchronize()
    if logits.ndim != 3 or position < 0 or position >= logits.shape[1]:
        raise ValueError(f"invalid logits/position: shape={tuple(logits.shape)}, position={position}")
    selected = logits[0, position].detach().clone()
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError(f"non-finite {path} logits")
    return {
        "logits": selected,
        "elapsed_ns": time.perf_counter_ns() - started,
        "compiled_runtime_invocations": audit.runtime_invocations - before,
    }


def exact_distribution_metrics(
    torch: Any, reference_logits: Any, candidate_logits: Any, temperature: float
) -> tuple[dict[str, Any], Any, Any]:
    ref_logp = torch.log_softmax(reference_logits.double() / temperature, dim=-1)
    alt_logp = torch.log_softmax(candidate_logits.double() / temperature, dim=-1)
    ref_prob = ref_logp.exp()
    alt_prob = alt_logp.exp()
    midpoint = 0.5 * (ref_prob + alt_prob)
    midpoint_log = torch.log(midpoint)
    tv = 0.5 * (ref_prob - alt_prob).abs().sum()
    kl_ref_alt = (ref_prob * (ref_logp - alt_logp)).sum()
    kl_alt_ref = (alt_prob * (alt_logp - ref_logp)).sum()
    js = 0.5 * (
        (ref_prob * (ref_logp - midpoint_log)).sum()
        + (alt_prob * (alt_logp - midpoint_log)).sum()
    )
    ref_entropy = -(ref_prob * ref_logp).sum()
    alt_entropy = -(alt_prob * alt_logp).sum()
    ref_top = int(torch.argmax(ref_prob).item())
    alt_top = int(torch.argmax(alt_prob).item())
    ref_top5 = torch.topk(ref_prob, k=5).indices
    alt_top5 = torch.topk(alt_prob, k=5).indices
    ref_top_event_shift = alt_prob[ref_top] - ref_prob[ref_top]
    alt_top_event_shift = alt_prob[alt_top] - ref_prob[alt_top]
    ref_top5_mass_shift = alt_prob[ref_top5].sum() - ref_prob[ref_top5].sum()
    alt_top5_mass_shift = alt_prob[alt_top5].sum() - ref_prob[alt_top5].sum()
    metrics = {
        "total_variation": float(tv.item()),
        "minimal_paired_disagreement_maximal_coupling": float(tv.item()),
        "kl_reference_to_candidate": float(kl_ref_alt.item()),
        "kl_candidate_to_reference": float(kl_alt_ref.item()),
        "jensen_shannon": float(js.item()),
        "reference_entropy": float(ref_entropy.item()),
        "candidate_entropy": float(alt_entropy.item()),
        "entropy_signed_delta": float((alt_entropy - ref_entropy).item()),
        "reference_top1_token": ref_top,
        "candidate_top1_token": alt_top,
        "argmax_disagreement": ref_top != alt_top,
        "reference_top1_probability": float(ref_prob[ref_top].item()),
        "candidate_probability_of_reference_top1": float(alt_prob[ref_top].item()),
        "reference_top1_probability_signed_delta": float(ref_top_event_shift.item()),
        "candidate_top1_probability_signed_delta": float(alt_top_event_shift.item()),
        "reference_top5_mass_signed_delta": float(ref_top5_mass_shift.item()),
        "candidate_top5_mass_signed_delta": float(alt_top5_mass_shift.item()),
        "reference_within_distribution_independent_draw_disagreement": float(
            (1.0 - (ref_prob * ref_prob).sum()).item()
        ),
        "candidate_within_distribution_independent_draw_disagreement": float(
            (1.0 - (alt_prob * alt_prob).sum()).item()
        ),
        "cross_distribution_independent_draw_disagreement": float(
            (1.0 - (ref_prob * alt_prob).sum()).item()
        ),
    }
    return metrics, ref_prob, alt_prob


def rng_calibration(
    torch: Any,
    reference_prob: Any,
    candidate_prob: Any,
    reference_top: int,
    state_index: int,
    replicates: int,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    ref_cdf = torch.cumsum(reference_prob, dim=-1)
    alt_cdf = torch.cumsum(candidate_prob, dim=-1)
    ref_cdf[-1] = 1.0
    alt_cdf[-1] = 1.0
    for replicate in range(replicates):
        ref_seed = seed + state_index * 100_000 + replicate * 10 + 1
        alt_seed = seed + state_index * 100_000 + replicate * 10 + 2
        coupling_seed = seed + state_index * 100_000 + replicate * 10 + 3
        ref_generator = torch.Generator(device="cuda").manual_seed(ref_seed)
        alt_generator = torch.Generator(device="cuda").manual_seed(alt_seed)
        coupling_generator = torch.Generator(device="cuda").manual_seed(coupling_seed)
        ref_draws = torch.multinomial(reference_prob, draws, replacement=True, generator=ref_generator)
        alt_draws = torch.multinomial(candidate_prob, draws, replacement=True, generator=alt_generator)
        uniforms = torch.rand(draws, device="cuda", dtype=torch.float64, generator=coupling_generator)
        ref_coupled = torch.searchsorted(ref_cdf, uniforms)
        alt_coupled = torch.searchsorted(alt_cdf, uniforms)
        rows.append(
            {
                "replicate": replicate,
                "draws": draws,
                "reference_seed": ref_seed,
                "candidate_seed": alt_seed,
                "coupling_seed": coupling_seed,
                "reference_top1_event_frequency": float((ref_draws == reference_top).double().mean().item()),
                "candidate_reference_top1_event_frequency": float(
                    (alt_draws == reference_top).double().mean().item()
                ),
                "independent_stream_token_match_rate": float((ref_draws == alt_draws).double().mean().item()),
                "common_uniform_inverse_cdf_disagreement_rate": float(
                    (ref_coupled != alt_coupled).double().mean().item()
                ),
                "common_uniform_is_coupling_dependent": True,
            }
        )
    return rows


def state_bootstrap_ci(
    rows: list[dict[str, Any]], field: str, draws: int, seed: int
) -> list[float]:
    values = {row["state_id"]: float(row[field]) for row in rows}
    keys = sorted(values)
    rng = random.Random(seed)
    estimates = [sum(values[rng.choice(keys)] for _ in keys) / len(keys) for _ in range(draws)]
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def main() -> None:
    args = parse_args()
    if args.temperature <= 0 or args.model_repeats < 2 or args.rng_replicates < 2 or args.draws <= 0:
        raise ValueError("invalid temperature/repeat/draw contract")
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
    torch._dynamo.config.recompile_limit = 2

    subject_args = argparse.Namespace(
        subject="qwen_causal",
        model_path=args.model_path,
        data_path=args.data_path,
        out_dir=args.out_dir,
        start=args.start,
        count=args.count,
        repeats=args.model_repeats,
        sequence_length=args.sequence_length,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    model, eager_function, states, subject_metadata = load_subject(torch, subject_args)
    audit = CompileAudit()
    compiled_function = torch.compile(
        eager_function,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )
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
            "sha256": sha256_file(Path(args.data_path)),
        },
        "sampling_contract": {
            "state_position": "final valid response prediction position",
            "probability_arithmetic": "float64",
            "single_sample_is_oracle": False,
            "total_variation_is_coupling_independent": True,
            "inverse_cdf_token_order_coupling": "diagnostic only",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    warm_position = int(states[0].observation_positions[-1])
    execute(torch, compiled_function, states[0].inputs, warm_position, audit, "compiled")
    execute(torch, eager_function, states[0].inputs, warm_position, audit, "eager")
    warmup_runtime_invocations = audit.runtime_invocations
    warmup_backend_compiles = audit.backend_compiles

    state_rows = []
    replicate_rows = []
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as state_handle, (
        out_dir / "replicates.jsonl"
    ).open("w", encoding="utf-8") as replicate_handle:
        for state_index, state in enumerate(states):
            position = int(state.observation_positions[-1])
            baselines = {}
            outputs = {}
            self_exact = {"eager": True, "compiled": True}
            runtime_calls = 0
            execution_orders = []
            for repeat in range(args.model_repeats):
                order = ["eager", "compiled"] if repeat % 2 == 0 else ["compiled", "eager"]
                execution_orders.append(order)
                current = {}
                for path in order:
                    function = eager_function if path == "eager" else compiled_function
                    current[path] = execute(torch, function, state.inputs, position, audit, path)
                runtime_calls += int(current["compiled"]["compiled_runtime_invocations"])
                if repeat == 0:
                    baselines = {path: current[path]["logits"] for path in ["eager", "compiled"]}
                    outputs = current
                else:
                    for path in ["eager", "compiled"]:
                        self_exact[path] = self_exact[path] and tensor_exact(
                            torch, baselines[path], current[path]["logits"]
                        )
            metrics, ref_prob, alt_prob = exact_distribution_metrics(
                torch, baselines["eager"], baselines["compiled"], args.temperature
            )
            rng_rows = rng_calibration(
                torch,
                ref_prob,
                alt_prob,
                int(metrics["reference_top1_token"]),
                state_index,
                args.rng_replicates,
                args.draws,
                args.seed,
            )
            exact_ref_top = float(metrics["reference_top1_probability"])
            exact_alt_top = float(metrics["candidate_probability_of_reference_top1"])
            for item in rng_rows:
                item.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "state_id": state.state_id,
                        "state_index": state_index,
                        "reference_top1_event_frequency_error": item[
                            "reference_top1_event_frequency"
                        ]
                        - exact_ref_top,
                        "candidate_top1_event_frequency_error": item[
                            "candidate_reference_top1_event_frequency"
                        ]
                        - exact_alt_top,
                    }
                )
                replicate_handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
                replicate_rows.append(item)
            replicate_handle.flush()
            row = {
                "schema_version": SCHEMA_VERSION,
                "state_id": state.state_id,
                "state_index": state_index,
                "state_metadata": state.metadata,
                "token_prediction_position": position,
                "temperature": args.temperature,
                "model_execution_orders": execution_orders,
                "candidate_compiled_runtime_invocations": runtime_calls,
                "candidate_execution_valid": runtime_calls == args.model_repeats,
                "model_self_pair_exact": self_exact,
                "rng_reference_top1_frequency_between_replicate_variance": statistics.variance(
                    item["reference_top1_event_frequency"] for item in rng_rows
                ),
                "rng_candidate_top1_frequency_between_replicate_variance": statistics.variance(
                    item["candidate_reference_top1_event_frequency"] for item in rng_rows
                ),
                "mean_common_uniform_inverse_cdf_disagreement_rate": mean(
                    [item["common_uniform_inverse_cdf_disagreement_rate"] for item in rng_rows]
                ),
                **metrics,
            }
            state_handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            state_handle.flush()
            state_rows.append(row)

    measurement_compiles = audit.backend_compiles - warmup_backend_compiles
    self_failures = [
        row for row in state_rows if not all(row["model_self_pair_exact"].values())
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "subject": "qwen_causal_final_position_sampling",
        "sampling": {
            "states": len(state_rows),
            "rng_replicates_per_state_path": args.rng_replicates,
            "draws_per_replicate": args.draws,
            "state_bootstrap_unit": "sequence state",
        },
        "validity": {
            "candidate_calls_valid": all(row["candidate_execution_valid"] for row in state_rows),
            "self_pair_nonzero_count": len(self_failures),
            "backend_compiles": audit.backend_compiles,
            "backend_compiles_during_measurement": measurement_compiles,
            "no_graph_proliferation_after_warmup": measurement_compiles == 0,
            "measurement_runtime_invocations": audit.runtime_invocations - warmup_runtime_invocations,
            "graph_code_sha256": audit.graph_code_sha256,
            "graph_node_counts": audit.graph_node_counts,
        },
        "distribution": {
            "mean_total_variation": mean([row["total_variation"] for row in state_rows]),
            "mean_total_variation_state_bootstrap_95ci": state_bootstrap_ci(
                state_rows, "total_variation", args.bootstrap, args.seed + 1
            ),
            "max_total_variation": max(row["total_variation"] for row in state_rows),
            "mean_jensen_shannon": mean([row["jensen_shannon"] for row in state_rows]),
            "mean_entropy_signed_delta": mean(
                [row["entropy_signed_delta"] for row in state_rows]
            ),
            "mean_entropy_signed_delta_state_bootstrap_95ci": state_bootstrap_ci(
                state_rows, "entropy_signed_delta", args.bootstrap, args.seed + 2
            ),
            "argmax_disagreements": sum(row["argmax_disagreement"] for row in state_rows),
            "mean_reference_top1_probability_signed_delta": mean(
                [row["reference_top1_probability_signed_delta"] for row in state_rows]
            ),
        },
        "algorithmic_rng": {
            "mean_reference_within_distribution_independent_draw_disagreement": mean(
                [
                    row["reference_within_distribution_independent_draw_disagreement"]
                    for row in state_rows
                ]
            ),
            "mean_candidate_within_distribution_independent_draw_disagreement": mean(
                [
                    row["candidate_within_distribution_independent_draw_disagreement"]
                    for row in state_rows
                ]
            ),
            "mean_cross_distribution_independent_draw_disagreement": mean(
                [row["cross_distribution_independent_draw_disagreement"] for row in state_rows]
            ),
            "mean_common_uniform_inverse_cdf_disagreement_rate": mean(
                [row["mean_common_uniform_inverse_cdf_disagreement_rate"] for row in state_rows]
            ),
            "mean_reference_top1_frequency_between_replicate_variance": mean(
                [
                    row["rng_reference_top1_frequency_between_replicate_variance"]
                    for row in state_rows
                ]
            ),
            "mean_candidate_top1_frequency_between_replicate_variance": mean(
                [
                    row["rng_candidate_top1_frequency_between_replicate_variance"]
                    for row in state_rows
                ]
            ),
            "mean_abs_reference_top1_frequency_error": mean(
                [abs(row["reference_top1_event_frequency_error"]) for row in replicate_rows]
            ),
            "mean_abs_candidate_top1_frequency_error": mean(
                [abs(row["candidate_top1_event_frequency_error"]) for row in replicate_rows]
            ),
        },
        "claim_scope": {
            "supported": "exact implementation-relative categorical distribution shift and separately calibrated sampling RNG variability",
            "not_supported": [
                "single sampled token as a correctness Oracle",
                "temperature-independent generalization",
                "normative correctness of either distribution",
            ],
        },
    }
    if self_failures or measurement_compiles or not summary["validity"]["candidate_calls_valid"]:
        raise RuntimeError(
            f"validity failure: self={len(self_failures)}, compiles={measurement_compiles}, "
            f"candidate={summary['validity']['candidate_calls_valid']}"
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

