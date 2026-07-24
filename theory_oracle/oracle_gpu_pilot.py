#!/usr/bin/env python
"""Deterministic-first matched-state pilot for the discrepancy Oracle.

This is deliberately a small controlled subject, not a benchmark and not a
correctness test.  It exercises the proposed measurement structure on CUDA:

* eager/eager, compiled/compiled, and eager/compiled paired executions;
* continuous, semantic-event, gradient, and one-step update endpoints;
* nested state/case/repeat units;
* fail-closed evidence that every candidate call reached an Inductor-compiled
  full graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "forkcert.oracle-gpu-pilot.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--states-per-stratum", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--clip-threshold", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


@dataclass
class CompileAudit:
    backend_compiles: int = 0
    runtime_invocations: int = 0
    graph_code_sha256: list[str] = field(default_factory=list)
    graph_node_counts: list[int] = field(default_factory=list)


def make_tracking_backend(torch: Any, audit: CompileAudit) -> Callable[..., Any]:
    def backend(graph_module: Any, example_inputs: list[Any]) -> Callable[..., Any]:
        audit.backend_compiles += 1
        audit.graph_code_sha256.append(sha256_text(graph_module.code))
        audit.graph_node_counts.append(sum(1 for _ in graph_module.graph.nodes))
        compiled = torch._inductor.compile(graph_module, example_inputs)

        def counted(*args: Any) -> Any:
            audit.runtime_invocations += 1
            return compiled(*args)

        return counted

    return backend


def subject(torch: Any, x: Any, w1: Any, w2: Any, bias: Any, target: Any) -> tuple[Any, Any]:
    # The matmul, GELU, second matmul, pointwise bias, and loss reduction give
    # Inductor legal opportunities to choose different floating-point graphs.
    hidden = torch.nn.functional.gelu(x @ w1, approximate="tanh")
    logits = hidden @ w2 + bias
    loss = torch.nn.functional.cross_entropy(logits.float(), target)
    return logits, loss


def make_base_parameters(torch: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    w1 = torch.randn(args.input_dim, args.hidden_dim, generator=generator, dtype=torch.float32)
    w1 = (w1 / math.sqrt(args.input_dim)).to(dtype=torch.float16, device="cuda")
    base = torch.randn(args.hidden_dim, generator=generator, dtype=torch.float32)
    base = base / math.sqrt(args.hidden_dim)
    w2 = torch.randn(args.hidden_dim, args.classes, generator=generator, dtype=torch.float32)
    w2 = w2 / math.sqrt(args.hidden_dim)
    # Classes 0 and 1 are deliberately close but not identical.  This creates a
    # predeclared near-boundary stratum without selecting states after seeing an
    # eager/compiled discrepancy.
    w2[:, 0] = base
    w2[:, 1] = base + 0.002 * torch.randn(args.hidden_dim, generator=generator)
    return w1, w2.to(dtype=torch.float16, device="cuda")


def state_tensors(torch: Any, args: argparse.Namespace, stratum: str, state_index: int) -> tuple[Any, Any, Any]:
    seed = args.seed + {"near": 10_000, "far": 20_000, "natural": 30_000}[stratum] + state_index
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(args.batch_size, args.input_dim, generator=generator, dtype=torch.float32)
    bias = torch.zeros(args.batch_size, args.classes, dtype=torch.float32)
    if stratum == "near":
        offsets = torch.linspace(-0.02, 0.02, args.batch_size)
        offsets += 0.002 * torch.randn(args.batch_size, generator=generator)
        bias[:, 0] = offsets / 2
        bias[:, 1] = -offsets / 2
        if args.classes > 2:
            bias[:, 2:] = -3.0
    elif stratum == "far":
        signs = torch.where(torch.arange(args.batch_size) % 2 == 0, 1.0, -1.0)
        offsets = signs * (0.5 + 0.1 * torch.rand(args.batch_size, generator=generator))
        bias[:, 0] = offsets / 2
        bias[:, 1] = -offsets / 2
        if args.classes > 2:
            bias[:, 2:] = -3.0
    elif stratum == "natural":
        bias = 0.2 * torch.randn(args.batch_size, args.classes, generator=generator)
    else:
        raise ValueError(stratum)
    target = torch.randint(args.classes, (args.batch_size,), generator=generator)
    return (
        x.to(dtype=torch.float16, device="cuda"),
        bias.to(dtype=torch.float16, device="cuda"),
        target.to(device="cuda"),
    )


def execute(
    torch: Any,
    fn: Callable[..., Any],
    path: str,
    audit: CompileAudit,
    x: Any,
    base_w1: Any,
    base_w2: Any,
    bias: Any,
    target: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    w1 = base_w1.detach().clone().requires_grad_(True)
    w2 = base_w2.detach().clone().requires_grad_(True)
    before = audit.runtime_invocations
    started = time.perf_counter_ns()
    logits, loss = fn(x, w1, w2, bias, target)
    grad1, grad2 = torch.autograd.grad(loss, (w1, w2))
    torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - started
    runtime_delta = audit.runtime_invocations - before
    if path == "compiled" and runtime_delta <= 0:
        raise RuntimeError("fail-closed: candidate call did not reach the tracked compiled callable")
    if path == "eager" and runtime_delta != 0:
        raise RuntimeError("eager execution unexpectedly changed the compiled runtime counter")

    grad_norm = torch.sqrt(grad1.float().square().sum() + grad2.float().square().sum())
    clip_coef = torch.clamp(args.clip_threshold / (grad_norm + 1e-6), max=1.0)
    update1 = -args.lr * grad1.float() * clip_coef
    update2 = -args.lr * grad2.float() * clip_coef
    update = torch.cat((update1.flatten(), update2.flatten()))
    logits32 = logits.detach().float()
    top2 = torch.topk(logits32, k=2, dim=-1).indices
    return {
        "path": path,
        "elapsed_ns": elapsed_ns,
        "compiled_runtime_invocations": runtime_delta,
        "loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm.detach().item()),
        "clip_event": bool((grad_norm > args.clip_threshold).item()),
        "logits": logits32.cpu().tolist(),
        "logits_sha256": tensor_sha256(logits),
        "argmax": logits32.argmax(dim=-1).cpu().tolist(),
        "top2": top2.cpu().tolist(),
        "class01_margin": (logits32[:, 0] - logits32[:, 1]).cpu().tolist(),
        "class01_event": (logits32[:, 0] > logits32[:, 1]).cpu().tolist(),
        "update": update.detach().cpu().tolist(),
        "update_sha256": tensor_sha256(update),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    left_logits = np.asarray(left["logits"], dtype=np.float64)
    right_logits = np.asarray(right["logits"], dtype=np.float64)
    left_update = np.asarray(left["update"], dtype=np.float64)
    right_update = np.asarray(right["update"], dtype=np.float64)
    logit_delta = right_logits - left_logits
    update_delta = right_update - left_update
    cases = []
    for index in range(left_logits.shape[0]):
        lmargin = float(left["class01_margin"][index])
        rmargin = float(right["class01_margin"][index])
        levent = bool(left["class01_event"][index])
        revent = bool(right["class01_event"][index])
        cases.append(
            {
                "case_index": index,
                "reference_margin": lmargin,
                "candidate_margin": rmargin,
                "signed_margin_delta": rmargin - lmargin,
                "distance_to_boundary": abs(lmargin),
                "reference_event": levent,
                "candidate_event": revent,
                "up_fork": (not levent) and revent,
                "down_fork": levent and (not revent),
                "argmax_reference": int(left["argmax"][index]),
                "argmax_candidate": int(right["argmax"][index]),
                "argmax_disagreement": left["argmax"][index] != right["argmax"][index],
                "top2_reference": left["top2"][index],
                "top2_candidate": right["top2"][index],
                "top2_set_disagreement": set(left["top2"][index]) != set(right["top2"][index]),
                "logit_mean_signed_delta": float(logit_delta[index].mean()),
                "logit_mean_abs_delta": float(np.abs(logit_delta[index]).mean()),
                "logit_max_abs_delta": float(np.abs(logit_delta[index]).max()),
            }
        )
    return {
        "reference_loss": left["loss"],
        "candidate_loss": right["loss"],
        "loss_signed_delta": right["loss"] - left["loss"],
        "reference_grad_norm": left["grad_norm"],
        "candidate_grad_norm": right["grad_norm"],
        "grad_norm_signed_delta": right["grad_norm"] - left["grad_norm"],
        "reference_clip_event": left["clip_event"],
        "candidate_clip_event": right["clip_event"],
        "clip_event_disagreement": left["clip_event"] != right["clip_event"],
        "reference_update_l2": float(np.linalg.norm(left_update)),
        "candidate_update_l2": float(np.linalg.norm(right_update)),
        "update_l2_delta": float(np.linalg.norm(update_delta)),
        "update_max_abs_delta": float(np.abs(update_delta).max()),
        "left_runtime_invocations": left["compiled_runtime_invocations"],
        "right_runtime_invocations": right["compiled_runtime_invocations"],
        "cases": cases,
    }


def bootstrap_state_ci(state_values: dict[str, float], draws: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    keys = sorted(state_values)
    if not keys:
        return [float("nan"), float("nan")]
    estimates = []
    for _ in range(draws):
        sample = [state_values[rng.choice(keys)] for _ in keys]
        estimates.append(mean(sample))
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def build_summary(records: list[dict[str, Any]], args: argparse.Namespace, audit: CompileAudit) -> dict[str, Any]:
    # Repeat 0 is the predeclared eager/compiled comparison.  Other repeats
    # estimate within-state runtime variability and self-pair disagreement.
    primary = [row for row in records if row["repeat"] == 0 and row["pair"] == "eager_compiled"]
    cases = [dict(case, state_id=row["state_id"], stratum=row["stratum"]) for row in primary for case in row["comparison"]["cases"]]
    state_case_deltas: dict[str, list[float]] = {}
    state_disagreement: dict[str, list[float]] = {}
    for case in cases:
        state_case_deltas.setdefault(case["state_id"], []).append(case["signed_margin_delta"])
        state_disagreement.setdefault(case["state_id"], []).append(float(case["reference_event"] != case["candidate_event"]))
    state_mean_delta = {key: mean(value) for key, value in state_case_deltas.items()}
    state_mean_disagreement = {key: mean(value) for key, value in state_disagreement.items()}

    repeat_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in records:
        if row["pair"] != "eager_compiled":
            continue
        for case in row["comparison"]["cases"]:
            repeat_groups.setdefault((row["state_id"], row["stratum"], case["case_index"]), []).append(case)
    within_variances = []
    for group in repeat_groups.values():
        values = [case["signed_margin_delta"] for case in group]
        within_variances.append(statistics.variance(values) if len(values) > 1 else 0.0)

    argmax_ref = Counter(case["argmax_reference"] for case in cases)
    argmax_alt = Counter(case["argmax_candidate"] for case in cases)
    total_cases = len(cases)
    categories = set(argmax_ref) | set(argmax_alt)
    argmax_tv = 0.5 * sum(abs(argmax_ref[value] - argmax_alt[value]) for value in categories) / total_cases

    self_pairs = [row for row in records if row["pair"] in {"eager_self", "compiled_self"}]
    self_nonzero = []
    for row in self_pairs:
        comparison = row["comparison"]
        if comparison["update_l2_delta"] != 0 or comparison["loss_signed_delta"] != 0:
            self_nonzero.append({"state_id": row["state_id"], "pair": row["pair"], "repeat": row["repeat"]})
        elif any(case["logit_max_abs_delta"] != 0 for case in comparison["cases"]):
            self_nonzero.append({"state_id": row["state_id"], "pair": row["pair"], "repeat": row["repeat"]})

    signed = [case["signed_margin_delta"] for case in cases]
    up = sum(case["up_fork"] for case in cases)
    down = sum(case["down_fork"] for case in cases)
    disagreements = up + down
    state_var = statistics.variance(state_mean_delta.values()) if len(state_mean_delta) > 1 else 0.0
    by_stratum = {}
    for stratum in ["near", "far", "natural"]:
        subset = [case for case in cases if case["stratum"] == stratum]
        by_stratum[stratum] = {
            "cases": len(subset),
            "mean_signed_margin_delta": mean([case["signed_margin_delta"] for case in subset]),
            "mean_abs_margin_delta": mean([abs(case["signed_margin_delta"]) for case in subset]),
            "class01_disagreement": mean([float(case["reference_event"] != case["candidate_event"]) for case in subset]),
            "argmax_disagreement": mean([float(case["argmax_disagreement"]) for case in subset]),
            "top2_set_disagreement": mean([float(case["top2_set_disagreement"]) for case in subset]),
            "median_reference_boundary_distance": quantile([case["distance_to_boundary"] for case in subset], 0.5),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "validity": {
            "all_candidate_calls_reached_compiled_callable": all(
                row["candidate_execution_valid"] for row in records if "candidate_execution_valid" in row
            ),
            "backend_compiles": audit.backend_compiles,
            "runtime_invocations": audit.runtime_invocations,
            "graph_code_sha256": audit.graph_code_sha256,
            "graph_node_counts": audit.graph_node_counts,
            "self_pair_nonzero_count": len(self_nonzero),
            "self_pair_nonzero_examples": self_nonzero[:10],
        },
        "sampling_units": {
            "states": len(state_mean_delta),
            "cases": total_cases,
            "repeats": args.repeats,
            "bootstrap_unit": "state",
        },
        "numerical": {
            "mean_signed_margin_delta": mean(signed),
            "mean_signed_margin_delta_state_bootstrap_95ci": bootstrap_state_ci(
                state_mean_delta, args.bootstrap, args.seed + 1
            ),
            "mean_abs_margin_delta": mean([abs(value) for value in signed]),
            "max_abs_margin_delta": max((abs(value) for value in signed), default=0.0),
            "mean_logit_mean_abs_delta": mean([case["logit_mean_abs_delta"] for case in cases]),
            "max_logit_max_abs_delta": max((case["logit_max_abs_delta"] for case in cases), default=0.0),
            "mean_signed_loss_delta": mean([row["comparison"]["loss_signed_delta"] for row in primary]),
            "mean_abs_loss_delta": mean([abs(row["comparison"]["loss_signed_delta"]) for row in primary]),
            "state_mean_delta_variance": state_var,
            "mean_within_state_case_repeat_variance": mean(within_variances),
            "interpretation": {
                "state_mean_delta_variance": "state heterogeneity, not runtime noise",
                "mean_within_state_case_repeat_variance": "same state/case repeated execution variability",
                "bootstrap_ci": "finite-state sampling uncertainty",
            },
        },
        "semantic": {
            "class01_up_forks": up,
            "class01_down_forks": down,
            "directional_shift": (up - down) / total_cases,
            "paired_disagreement": disagreements / total_cases,
            "paired_disagreement_state_bootstrap_95ci": bootstrap_state_ci(
                state_mean_disagreement, args.bootstrap, args.seed + 2
            ),
            "argmax_disagreement": mean([float(case["argmax_disagreement"]) for case in cases]),
            "top2_set_disagreement": mean([float(case["top2_set_disagreement"]) for case in cases]),
            "argmax_marginal_total_variation": argmax_tv,
            "coupling_note": "paired disagreement is coupling-dependent and upper-bounds marginal total variation",
        },
        "transition": {
            "mean_update_l2_delta": mean([row["comparison"]["update_l2_delta"] for row in primary]),
            "max_update_l2_delta": max((row["comparison"]["update_l2_delta"] for row in primary), default=0.0),
            "mean_reference_update_l2": mean(
                [row["comparison"]["reference_update_l2"] for row in primary]
            ),
            "mean_relative_update_l2_delta": mean(
                [
                    row["comparison"]["update_l2_delta"] / row["comparison"]["reference_update_l2"]
                    if row["comparison"]["reference_update_l2"] > 0
                    else 0.0
                    for row in primary
                ]
            ),
            "mean_reference_grad_norm": mean(
                [row["comparison"]["reference_grad_norm"] for row in primary]
            ),
            "mean_abs_grad_norm_delta": mean(
                [abs(row["comparison"]["grad_norm_signed_delta"]) for row in primary]
            ),
            "reference_clip_rate": mean(
                [float(row["comparison"]["reference_clip_event"]) for row in primary]
            ),
            "candidate_clip_rate": mean(
                [float(row["comparison"]["candidate_clip_event"]) for row in primary]
            ),
            "clip_event_disagreement": mean(
                [float(row["comparison"]["clip_event_disagreement"]) for row in primary]
            ),
        },
        "conditional": by_stratum,
        "claim_scope": {
            "supported": "calibrated implementation-relative discrepancy and one-step impact on a controlled CUDA subject",
            "not_supported": [
                "compiler correctness failure",
                "application harm",
                "long-run training effect",
                "generalization to arbitrary operators, models, or state distributions",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    if args.classes < 3:
        raise ValueError("--classes must be at least 3")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to estimate runtime variability")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this pilot must not silently fall back to CPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 2

    base_w1, base_w2 = make_base_parameters(torch, args)
    audit = CompileAudit()

    def eager_fn(x: Any, w1: Any, w2: Any, bias: Any, target: Any) -> tuple[Any, Any]:
        return subject(torch, x, w1, w2, bias, target)

    compiled_fn = torch.compile(
        eager_fn,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_unix": time.time(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "comparison_contract": {
            "reference": "PyTorch eager on CUDA",
            "candidate": "torch.compile fullgraph=True, backend=tracked Inductor",
            "target_state_distribution": "equal mixture of predeclared near, far, and natural strata",
            "observable_profile": [
                "logits and signed class-0/class-1 margin",
                "class-0-vs-class-1 event, argmax, and top-2 set",
                "loss, gradient norm, clipping event",
                "clipped one-step SGD update",
            ],
            "randomness_protocol": "deterministic algorithms; no dropout or sampling RNG; repeated identical execution",
            "acceptance_semantics": "descriptive pilot only; no application tolerance or correctness specification",
            "execution_identity": "tracked Inductor callable invocation on every candidate execution; fullgraph and fail-closed",
        },
        "arguments": vars(args),
        "parameter_hashes": {"w1": tensor_sha256(base_w1), "w2": tensor_sha256(base_w2)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    records: list[dict[str, Any]] = []
    for stratum in ["near", "far", "natural"]:
        for state_index in range(args.states_per_stratum):
            state_id = f"{stratum}-{state_index:04d}"
            x, bias, target = state_tensors(torch, args, stratum, state_index)
            eager_runs = []
            compiled_runs = []
            for repeat in range(args.repeats):
                eager_runs.append(execute(torch, eager_fn, "eager", audit, x, base_w1, base_w2, bias, target, args))
                compiled_runs.append(
                    execute(torch, compiled_fn, "compiled", audit, x, base_w1, base_w2, bias, target, args)
                )
                comparison = summarize_pair(eager_runs[-1], compiled_runs[-1])
                records.append(
                    {
                        "state_id": state_id,
                        "stratum": stratum,
                        "state_index": state_index,
                        "repeat": repeat,
                        "pair": "eager_compiled",
                        "candidate_execution_valid": comparison["right_runtime_invocations"] > 0,
                        "comparison": comparison,
                    }
                )
            # Self-pairs use repeat 0 as the fixed anchor and every later repeat.
            for repeat in range(1, args.repeats):
                records.append(
                    {
                        "state_id": state_id,
                        "stratum": stratum,
                        "state_index": state_index,
                        "repeat": repeat,
                        "pair": "eager_self",
                        "comparison": summarize_pair(eager_runs[0], eager_runs[repeat]),
                    }
                )
                compiled_self = summarize_pair(compiled_runs[0], compiled_runs[repeat])
                records.append(
                    {
                        "state_id": state_id,
                        "stratum": stratum,
                        "state_index": state_index,
                        "repeat": repeat,
                        "pair": "compiled_self",
                        "candidate_execution_valid": (
                            compiled_runs[0]["compiled_runtime_invocations"] > 0
                            and compiled_runs[repeat]["compiled_runtime_invocations"] > 0
                        ),
                        "comparison": compiled_self,
                    }
                )
            del x, bias, target

    summary = build_summary(records, args, audit)
    if not summary["validity"]["all_candidate_calls_reached_compiled_callable"]:
        raise RuntimeError("fail-closed: at least one candidate record lacks compiled execution evidence")
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest["completed_unix"] = time.time()
    manifest["audit"] = {
        "backend_compiles": audit.backend_compiles,
        "runtime_invocations": audit.runtime_invocations,
        "graph_code_sha256": audit.graph_code_sha256,
        "graph_node_counts": audit.graph_node_counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
