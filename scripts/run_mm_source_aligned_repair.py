#!/usr/bin/env python3
"""Run source-aligned MM repairs without conflating accumulation and rounding.

The old intervention ``fp32_mm(...).to(low_dtype)`` is intentionally named
``KERNEL_ONLY`` here: it retains deterministic output rounding.  Rounding arms
use coordinate-wise unbiased stochastic materialization through the original
BF16/FP16 ABI.  Consequently they are source-debiasing interventions, not an
exact FP32-shadow claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_analyzer.precision import (  # noqa: E402
    decompose_low_precision_output,
    source_aligned_mm_output,
)
from kernel_analyzer.bias_formation_v22 import (  # noqa: E402
    ConditionalPolicy,
    aggregate_conditional_debias,
    summarize_conditional_gram,
)
from kernel_analyzer.seup import adamw_effective_update_delta  # noqa: E402
from kernel_analyzer.streaming import StreamingGramAccumulator  # noqa: E402
from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest,
    load_model,
    tensor_digest,
)


RANDOMIZED = {"ROUNDING_ONLY", "JOINT"}
ALLOWED_ARMS = {"KERNEL_ONLY", *RANDOMIZED}


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def vector_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().float().reshape(-1)
    return {
        "coordinates": flat.numel(),
        "nonzero": int(torch.count_nonzero(flat).item()),
        "signed_mean": float(flat.double().mean().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "max_abs": float(flat.abs().max().item()),
    }


def validate_target_call(modules: list[Any], target_sha: str) -> dict[str, Any]:
    """Bind the runtime callsite without requiring unrelated wrapper bytes.

    A whole generated wrapper can change because of unrelated scheduling or
    cache metadata.  The intervention needs the exact call expression and an
    observed single execution, not byte identity for thousands of unrelated
    lines.
    """

    matches = []
    for module in modules:
        path = Path(module.__file__)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            source = line.strip()
            if hashlib.sha256(source.encode()).hexdigest() == target_sha:
                matches.append({
                    "phase": getattr(module, "__name__", "generated"),
                    "line": line_number,
                    "call": source,
                })
    if len(matches) != 1 or "extern_kernels.mm" not in matches[0]["call"]:
        raise RuntimeError(
            f"runtime wrapper contains {len(matches)} copies of the exact target call"
        )
    return matches[0]


class SourceAlignedMMRepair:
    """Replace one compiler-bound external MM output at its real F+B boundary."""

    def __init__(
        self,
        modules: list[Any],
        target_sha: str,
        mode: str,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        if mode not in {"SHAM", *ALLOWED_ARMS}:
            raise ValueError(mode)
        if mode in RANDOMIZED and rounding_seed is None:
            raise ValueError("randomized repair requires rounding_seed")
        self.modules = modules
        self.target_sha = target_sha
        self.mode = mode
        self.rounding_seed = rounding_seed
        self.restores: list[tuple[Any, Any]] = []
        self.calls = 0
        self.vectors: dict[str, torch.Tensor] = {}
        self.summary: dict[str, Any] | None = None

    def __enter__(self) -> "SourceAlignedMMRepair":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                _, _, digest = _source_identity()
                result = _original(*args, **kwargs)
                if digest != self.target_sha:
                    return result
                actual = kwargs.get("out", result)
                if not isinstance(actual, torch.Tensor):
                    raise RuntimeError("target MM has no tensor output")
                actual_before = actual.detach().clone()
                high = fp32_external_reference("mm", args, kwargs)
                generator = None
                if self.mode in RANDOMIZED:
                    generator = torch.Generator(device=actual.device)
                    assert self.rounding_seed is not None
                    generator.manual_seed(self.rounding_seed)
                repaired = source_aligned_mm_output(
                    actual_before, high, self.mode, generator=generator,
                )
                actual.copy_(repaired.delivered)

                terms = decompose_low_precision_output(actual_before, high)
                if self.mode == "KERNEL_ONLY":
                    natural_source = terms["kernel"]
                    repaired_source = repaired.delivered.float() - high.to(actual.dtype).float()
                elif self.mode == "ROUNDING_ONLY":
                    natural_source = terms["output_rounding"]
                    repaired_source = repaired.base.float() - high
                elif self.mode == "JOINT":
                    natural_source = terms["total"]
                    repaired_source = repaired.delivered.float() - high
                else:  # exact reconstruction sham
                    natural_source = torch.zeros_like(high)
                    repaired_source = repaired.delivered.float() - actual_before.float()

                natural_kernel = terms["kernel"]
                realized_kernel = repaired.delivered.float() - repaired.base.float()
                self.vectors = {
                    "natural_source": natural_source.detach().cpu(),
                    "repaired_source": repaired_source.detach().cpu(),
                    "source_removed": (natural_source - repaired_source).detach().cpu(),
                    "delivered_total_error": (repaired.delivered.float() - high).detach().cpu(),
                }
                self.summary = {
                    "mode": self.mode,
                    "natural_source": vector_summary(natural_source),
                    "repaired_source": vector_summary(repaired_source),
                    "intervention": vector_summary(actual_before.float() - repaired.delivered.float()),
                    "delivered_total_error": vector_summary(repaired.delivered.float() - high),
                    "kernel_residual_preservation_error": vector_summary(
                        realized_kernel - natural_kernel
                    ),
                    "low_dtype": str(actual.dtype),
                    "rounding_seed": self.rounding_seed,
                }
                self.calls += 1
                return result

            namespace.mm = wrapped
            self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        for namespace, original in self.restores:
            namespace.mm = original
        if self.calls != 1 or self.summary is None:
            raise RuntimeError(f"target source-aligned repair executed {self.calls} times")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--arms", default="ROUNDING_ONLY")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument(
        "--state-start", type=int, default=0,
        help="Start index in the frozen input bank; used for condition-disjoint shards.",
    )
    parser.add_argument("--rounding-repeats", type=int, default=8)
    parser.add_argument("--rounding-seed-base", type=int, default=910_000)
    parser.add_argument(
        "--conditional-debias", action="store_true",
        help=(
            "Within each fixed state, retain replicate-level local/F+B/update "
            "vectors and certify conditional means. Cross-state direction remains "
            "supplementary."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--temp-root", type=Path,
        default=Path("/data1/tzh/cache/kernel_analyzer_contrasts/source_aligned_repair"),
    )
    args = parser.parse_args()
    arms = tuple(dict.fromkeys(part.strip() for part in args.arms.split(",") if part.strip()))
    if not arms or not set(arms) <= ALLOWED_ARMS:
        raise ValueError(f"invalid repair arms: {arms}")
    if args.states < 2 or args.rounding_repeats < 2 or args.state_start < 0:
        raise ValueError("at least two states and two stochastic repeats are required")
    if args.conditional_debias and args.rounding_repeats < 4:
        raise ValueError("conditional debiasing requires at least four repeats")
    if not (0.0 < args.learning_rate < 1.0):
        raise ValueError("learning rate must lie in (0, 1)")

    queue = json.loads((ROOT / "results/coverage/bias_candidate_queue.json").read_text())
    bound = next(row for row in queue["candidates"] if row["candidate_id"] == args.candidate_id)
    exact_call = bound["exact_generated_call"]
    if exact_call["function"] != "extern_kernels.mm" or exact_call["source_line_sha256"] != args.target_sha:
        raise RuntimeError("candidate ID does not bind the declared external MM")
    decomposition = json.loads(args.decomposition.read_text())
    if decomposition["candidate_id"] != args.candidate_id or not all(decomposition["gates"].values()):
        raise RuntimeError("precision decomposition is absent or invalid")
    coherent = set(decomposition["coherent_sources"])
    required_by_arm = {
        "KERNEL_ONLY": {"kernel"},
        "ROUNDING_ONLY": {"output_rounding"},
        "JOINT": {"kernel", "output_rounding"},
    }
    for arm in arms:
        if not required_by_arm[arm] <= coherent:
            raise RuntimeError(
                f"{arm} is not source-aligned with coherent sources {sorted(coherent)}"
            )

    bank = json.loads(args.input_bank.read_text())
    all_states = bank.get("states", bank.get("records"))
    states = all_states[args.state_start:args.state_start + args.states]
    if len(states) != args.states:
        raise RuntimeError("input bank shorter than requested population")
    capture = json.loads((args.release_dir / "capture.json").read_text())
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank differs from frozen runtime release")

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    parameters = dict(model.named_parameters())
    if args.carrier not in parameters:
        raise RuntimeError("declared carrier parameter is absent")
    target = parameters[args.carrier]
    target_cpu = target.detach().float().cpu().clone()
    zero_moment = torch.zeros_like(target_cpu)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not bool(capture.get("allow_graph_breaks", False)), dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    runtime_target = validate_target_call(
        [module for module, _phase in wrapper_modules(modules)], args.target_sha
    )

    spool = args.temp_root / hashlib.sha256(args.candidate_id.encode()).hexdigest()[:20]
    grams = {
        (arm, role): StreamingGramAccumulator(spool, f"{args.candidate_id}_{arm}_{role}")
        for arm in arms for role in ("repaired_source", "source_removed", "carrier_removed")
    }

    def execute(
        tokens: list[int], state_index: int, mode: str | None, repeat: int = 0,
    ) -> tuple[torch.Tensor, dict[str, str], SourceAlignedMMRepair | None]:
        model_seed = 24000 + state_index
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        observer = None if mode is None else SourceAlignedMMRepair(
            modules, args.target_sha, mode,
            rounding_seed=(
                args.rounding_seed_base + 10_000 * state_index + repeat
                if mode in RANDOMIZED else None
            ),
        )
        if observer is None:
            loss = candidate(values); loss.backward()
        else:
            with observer:
                loss = candidate(values); loss.backward()
        torch.cuda.synchronize(device)
        if target.grad is None:
            raise RuntimeError("declared carrier gradient is absent")
        carrier = target.grad.detach().float().cpu().clone()
        identity = {"loss": tensor_digest(loss)}
        return carrier, identity, observer

    rows: list[dict[str, Any]] = []
    for local_state_index, state in enumerate(states):
        state_index = args.state_start + local_state_index
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        natural_grad, natural_identity, _ = execute(tokens, state_index, None)
        sham_grad, sham_identity, sham = execute(tokens, state_index, "SHAM")
        if sham is None or sham_identity != natural_identity or not torch.equal(sham_grad, natural_grad):
            raise RuntimeError("source-repair sham changed the F+B endpoint")

        arm_rows: dict[str, Any] = {}
        for arm in arms:
            repeats = args.rounding_repeats if arm in RANDOMIZED else 1
            conditional_accumulators = None
            if args.conditional_debias and arm in RANDOMIZED:
                conditional_root = spool / "conditional" / hashlib.sha256(
                    f"{state_id}\0{arm}".encode()
                ).hexdigest()[:20]
                conditional_accumulators = {
                    role: StreamingGramAccumulator(
                        conditional_root,
                        f"{args.candidate_id}_{state_id}_{arm}_{role}",
                    )
                    for role in (
                        "repair_local_residual",
                        "candidate_local_effect_removed",
                        "candidate_gradient_effect_removed",
                        "candidate_sgd_update_effect_removed",
                        "candidate_adamw_zero_update_effect_removed",
                    )
                }
            mean_grad = torch.zeros_like(natural_grad)
            mean_vectors: dict[str, torch.Tensor] = {}
            summaries = []
            identities = []
            for repeat in range(repeats):
                repair_grad, identity, observer = execute(tokens, state_index, arm, repeat)
                assert observer is not None and observer.summary is not None
                mean_grad.add_(repair_grad, alpha=1.0 / repeats)
                for name, value in observer.vectors.items():
                    if name not in mean_vectors:
                        mean_vectors[name] = torch.zeros_like(value)
                    mean_vectors[name].add_(value, alpha=1.0 / repeats)
                summaries.append(observer.summary)
                identities.append(identity)
                if conditional_accumulators is not None:
                    replicate_id = f"repeat_{repeat:04d}"
                    gradient_effect = natural_grad - repair_grad
                    sgd_update_effect = gradient_effect.mul(-args.learning_rate)
                    adamw_update_effect = adamw_effective_update_delta(
                        natural_grad,
                        repair_grad,
                        zero_moment,
                        zero_moment,
                        target_cpu,
                        step=1,
                        learning_rate=args.learning_rate,
                        betas=(0.9, 0.95),
                        epsilon=1.0e-8,
                        weight_decay=0.0,
                    )["value"]
                    values_by_role = {
                        "repair_local_residual": observer.vectors["repaired_source"],
                        "candidate_local_effect_removed": observer.vectors["source_removed"],
                        "candidate_gradient_effect_removed": gradient_effect,
                        "candidate_sgd_update_effect_removed": sgd_update_effect,
                        "candidate_adamw_zero_update_effect_removed": adamw_update_effect,
                    }
                    for role, value in values_by_role.items():
                        conditional_accumulators[role].add_array(
                            replicate_id, value.numpy(),
                        )
                    del gradient_effect, sgd_update_effect, adamw_update_effect
                del repair_grad, observer

            carrier_removed = natural_grad - mean_grad
            grams[(arm, "repaired_source")].add_array(
                state_id, mean_vectors["repaired_source"].numpy()
            )
            grams[(arm, "source_removed")].add_array(
                state_id, mean_vectors["source_removed"].numpy()
            )
            grams[(arm, "carrier_removed")].add_array(
                state_id, carrier_removed.numpy()
            )
            arm_rows[arm] = {
                "repeats": repeats,
                "repair_endpoint_identities": identities,
                "mean_repaired_source": vector_summary(mean_vectors["repaired_source"]),
                "mean_source_removed": vector_summary(mean_vectors["source_removed"]),
                "mean_delivered_total_error": vector_summary(mean_vectors["delivered_total_error"]),
                "carrier_removed": vector_summary(carrier_removed),
                "repeat_local_summaries": summaries,
            }
            if conditional_accumulators is not None:
                conditional: dict[str, Any] = {}
                role_specs = {
                    "repair_local_residual": (
                        "REPAIR_RESIDUAL",
                        "EXACT_DECLARED_LOCAL_SOURCE_ZERO_SAME_OPERANDS",
                    ),
                    "candidate_local_effect_removed": (
                        "CANDIDATE_MINUS_REPAIR_ENSEMBLE",
                        "STOCHASTIC_SOURCE_DEBIASED_ENSEMBLE",
                    ),
                    "candidate_gradient_effect_removed": (
                        "CANDIDATE_MINUS_REPAIR_ENSEMBLE",
                        "STOCHASTIC_SOURCE_DEBIASED_ENSEMBLE_AFTER_REAL_BACKWARD",
                    ),
                    "candidate_sgd_update_effect_removed": (
                        "CANDIDATE_MINUS_REPAIR_ENSEMBLE",
                        "STATELESS_SGD_OF_STOCHASTIC_REPAIR_GRADIENT_ENSEMBLE",
                    ),
                    "candidate_adamw_zero_update_effect_removed": (
                        "CANDIDATE_MINUS_REPAIR_ENSEMBLE",
                        "ADAMW_ZERO_MOMENT_STEP1_OF_STOCHASTIC_REPAIR_GRADIENT_ENSEMBLE",
                    ),
                }
                for role_index, (role, (estimand, reference)) in enumerate(
                    role_specs.items()
                ):
                    raw = conditional_accumulators[role].finalize(
                        bootstrap_draws=2000,
                        seed=29_000 + 100 * state_index + 10 * role_index,
                        cleanup=True,
                    )
                    conditional[role] = summarize_conditional_gram(
                        raw["gram"],
                        condition_id=state_id,
                        coordinate_count=raw["coordinates"],
                        replicate_ids=raw["state_ids"],
                        vector_digests=[
                            raw["state_vector_sha256"][rid]
                            for rid in raw["state_ids"]
                        ],
                        estimand=estimand,
                        reference=reference,
                        policy=ConditionalPolicy(
                            min_replicates=4,
                            bootstrap_samples=2000,
                            bootstrap_seed=(
                                31_000 + 100 * state_index + 10 * role_index
                            ),
                        ),
                    )
                arm_rows[arm]["conditional_debias"] = {
                    "status": "COMPLETE",
                    "fixed_state_id": state_id,
                    "only_intervention_randomness_changes_across_replicates": True,
                    "layers": conditional,
                    "downstream_repair_residual_reference": (
                        "NOT_AVAILABLE; gradient/update certificates measure candidate "
                        "effect removed relative to the stochastic source-debiased "
                        "ensemble, not absolute repair residual bias"
                    ),
                    "optimizer_conditions": {
                        "stateless_sgd": {"learning_rate": args.learning_rate},
                        "adamw_zero_moment_step1": {
                            "learning_rate": args.learning_rate,
                            "betas": [0.9, 0.95],
                            "epsilon": 1.0e-8,
                            "weight_decay": 0.0,
                            "natural_mature_moments_measured": False,
                        },
                    },
                }
            del mean_grad, mean_vectors, carrier_removed

        rows.append({
            "state_id": state_id,
            "natural_endpoint_identity": natural_identity,
            "sham_endpoint_identity": sham_identity,
            "sham_exact": True,
            "arms": arm_rows,
        })
        write(args.output, {
            "schema": "kernel-analyzer-mm-source-aligned-repair-v1",
            "status": "RUNNING",
            "candidate_id": args.candidate_id,
            "states": rows,
        })
        del natural_grad, sham_grad
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)

    certificates: dict[str, Any] = {}
    for arm_index, arm in enumerate(arms):
        certificates[arm] = {
            role: grams[(arm, role)].finalize(
                bootstrap_draws=4000,
                seed=18_000 + 100 * arm_index + role_index,
                cleanup=True,
            )
            for role_index, role in enumerate(
                ("repaired_source", "source_removed", "carrier_removed")
            )
        }
    arm_verdicts = {}
    for arm in arms:
        cert = certificates[arm]
        analytic_source_debias = True
        arm_verdicts[arm] = {
            "source_aligned": True,
            "sham_exact": all(row["sham_exact"] for row in rows),
            "analytic_expected_repaired_source_is_zero": analytic_source_debias,
            "repair_status": "SOURCE_DEBIASED_IN_EXPECTATION",
            "finite_repeat_repaired_source_directional": cert["repaired_source"]["status"] == "PASS",
            "removed_source_directional": cert["source_removed"]["status"] == "PASS",
            "downstream_carrier_effect_directional": cert["carrier_removed"]["status"] == "PASS",
            "verdict": (
                "SOURCE_DEBIASED_IN_EXPECTATION_WITH_DIRECTIONAL_DOWNSTREAM_EFFECT"
                if analytic_source_debias and cert["source_removed"]["status"] == "PASS"
                and cert["carrier_removed"]["status"] == "PASS"
                else "SOURCE_DEBIASED_IN_EXPECTATION_LOCAL_ONLY"
                if analytic_source_debias and cert["source_removed"]["status"] == "PASS"
                else "SOURCE_DEBIASED_IN_EXPECTATION_DOWNSTREAM_UNRESOLVED"
            ),
        }
    conditional_debias_summary: dict[str, Any] = {}
    if args.conditional_debias:
        for arm in arms:
            if arm not in RANDOMIZED:
                conditional_debias_summary[arm] = {
                    "status": "NOT_APPLICABLE_DETERMINISTIC_ARM",
                }
                continue
            conditional_debias_summary[arm] = aggregate_conditional_debias({
                row["state_id"]: row["arms"][arm]["conditional_debias"]["layers"]
                for row in rows
            })
    if args.conditional_debias and args.states >= 16:
        complete_status = "COMPLETE_CONDITIONAL_DEBIAS_CONFIRMATION"
    elif args.conditional_debias:
        complete_status = "COMPLETE_CONDITIONAL_DEBIAS_ENGINEERING"
    else:
        complete_status = (
            "COMPLETE_SOURCE_ALIGNED_REPAIR_POPULATION"
            if args.states == 32 else "COMPLETE_SOURCE_ALIGNED_REPAIR_PILOT"
        )
    payload = {
        "schema": "kernel-analyzer-mm-source-aligned-repair-v1",
        "status": complete_status,
        "candidate_id": args.candidate_id,
        "architecture": args.architecture,
        "carrier_parameter": args.carrier,
        "states": rows,
        "arms": list(arms),
        "arm_verdicts": arm_verdicts,
        "conditional_debias_summary": conditional_debias_summary,
        "direction": certificates,
        "bindings": {
            "source_line_sha256": args.target_sha,
            "runtime_target": runtime_target,
            "release_capture_sha256": capture["result_sha256"],
            "input_bank_sha256": file_digest(args.input_bank),
            "decomposition_result_sha256": decomposition["result_sha256"],
            "state_start": args.state_start,
            "state_count": args.states,
            "rounding_seed_base": args.rounding_seed_base,
            "sham_scope": (
                "bitwise loss plus complete declared carrier gradient; no redundant "
                "whole-model gradient hashing"
            ),
        },
        "claim_boundary": (
            "ROUNDING_ONLY and JOINT are coordinate-wise unbiased stochastic "
            "materialization through the original low-precision ABI. They test removal "
            "of deterministic rounding bias; they are not exact FP32-shadow or global "
            "full-reference equivalence claims. Their analytic conditional source mean is "
            "zero, while the finite-repeat certificate separately reports Monte Carlo "
            "residual. When conditional_debias is enabled, gradient/update certificates "
            "identify a candidate effect removed relative to the stochastic repair "
            "ensemble. Without an exact downstream reference they do not certify that "
            "the repair itself has zero downstream bias. KERNEL_ONLY retains deterministic "
            "rounding."
        ),
        "conditional_debias_enabled": args.conditional_debias,
    }
    payload["result_sha256"] = canonical(payload)
    write(args.output, payload)
    print(json.dumps({"event": "REPAIR_COMPLETE", "verdicts": arm_verdicts}, sort_keys=True))


if __name__ == "__main__":
    main()
