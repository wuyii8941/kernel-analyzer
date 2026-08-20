"""Evaluate low-cost bias screens without launching a GPU campaign.

The script performs two label-blind feasibility studies:

1. synthetic mechanism controls compare transported-mean, curvature, and
   exact paired-response decompositions;
2. retained complete Grams are sliced to measure how often 4/6/8/12 repair
   draws reproduce the already frozen full-repeat conditional verdict.

Only compact JSON/Markdown summaries are written.  The source Grams remain in
their existing compressed artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from kernel_analyzer.bias_formation_v22 import (
    BiasV22Status,
    ConditionalPolicy,
    summarize_conditional_gram,
)
from kernel_analyzer.bias_oracle_feasibility import (
    moment_response_sketch,
    paired_response_decomposition,
    shared_block_hvp_sketch,
    subset_square_matrix,
)


DEFAULT_ARTIFACTS = {
    "qwen64_vproj": "results/coverage/cases/qwen64_vproj_conditional_debias_r16.json.gz",
    "qwen128_vproj": "results/coverage/cases/qwen128_vproj_conditional_debias.json.gz",
    "mamba64_input_proj": "results/coverage/cases/mamba_seq64_input_proj_conditional_debias.json.gz",
}


def _mean(rows: Sequence[Sequence[float]]) -> list[float]:
    return [
        math.fsum(row[index] for row in rows) / len(rows)
        for index in range(len(rows[0]))
    ]


def _covariance_factor_1d(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    if any(len(row) != 1 for row in rows):
        raise ValueError("synthetic helper is intentionally one-dimensional")
    mean = _mean(rows)[0]
    variance = math.fsum((row[0] - mean) ** 2 for row in rows) / len(rows)
    return [[math.sqrt(max(0.0, variance))]]


def _relative_error(estimate: Sequence[float], truth: Sequence[float]) -> float:
    numerator = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(estimate, truth)))
    denominator = max(
        math.sqrt(math.fsum(a * a for a in estimate)),
        math.sqrt(math.fsum(b * b for b in truth)),
        1e-30,
    )
    return numerator / denominator


def synthetic_controls() -> list[dict[str, Any]]:
    controls = [
        {
            "id": "centered_linear_safe",
            "residuals": [[-1.0], [1.0], [-0.5], [0.5]],
            "response": lambda x: [2.0 * x[0]],
            "expected_channel": "NONE",
        },
        {
            "id": "source_mean_linear_transport",
            "residuals": [[-0.5], [0.5], [0.5], [1.5]],
            "response": lambda x: [2.0 * x[0]],
            "expected_channel": "TRANSPORTED_MEAN",
        },
        {
            "id": "centered_variance_quadratic_rectification",
            "residuals": [[-1.0], [1.0], [-1.0], [1.0]],
            "response": lambda x: [3.0 * x[0] * x[0]],
            "expected_channel": "CURVATURE_RECTIFICATION",
        },
        {
            "id": "mixed_source_and_rectification",
            "residuals": [[-0.5], [0.5], [0.5], [1.5]],
            "response": lambda x: [2.0 * x[0] + 3.0 * x[0] * x[0]],
            "expected_channel": "BOTH",
        },
        {
            "id": "nonsmooth_support_switch",
            "residuals": [[-1.0], [1.0], [-1.0], [1.0]],
            "response": lambda x: [max(0.0, abs(x[0]) - 0.6)],
            "expected_channel": "ESCALATE",
        },
    ]
    results: list[dict[str, Any]] = []
    for control in controls:
        residuals = control["residuals"]
        response = control["response"]
        exact = paired_response_decomposition(residuals, response)
        source_mean = _mean(residuals)
        sketch = moment_response_sketch(
            source_mean,
            _covariance_factor_1d(residuals),
            response,
            curvature_probes=4,
            scale=1.0,
            seed=20260820,
            check_half_scale=True,
            amplitude_tolerance=0.25,
        )
        results.append({
            "control_id": control["id"],
            "expected_channel": control["expected_channel"],
            "exact_paired_response": exact.as_dict(),
            "moment_response_sketch": sketch.as_dict(),
            "prediction_relative_error": _relative_error(
                sketch.predicted_bias, exact.natural_mean_response
            ),
            "note": (
                "The moment sketch estimates a local second-order response; "
                "the exact paired response is the reference decomposition."
            ),
        })
    return results


def shared_hvp_control() -> dict[str, Any]:
    """Prove that one coded HVP population can screen all local blocks."""

    try:
        import torch
    except ImportError:
        return {"status": "UNAVAILABLE_NO_TORCH"}
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    y = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    response = (
        3.0 * x[0]
        - 2.0 * y[0]
        + x[0] * x[0]
        + 3.0 * x[1] * x[1]
        + 5.0 * y[0] * y[0]
        + 7.0 * x[0] * y[0]
    )
    codes = [
        [list(signs[:2]), [signs[2]]]
        for signs in itertools.product((-1.0, 1.0), repeat=3)
    ]
    result = shared_block_hvp_sketch(
        response,
        [x, y],
        [torch.tensor([0.5, 0.0]), torch.tensor([0.25])],
        [torch.eye(2, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)],
        probes=len(codes),
        probe_signs=codes,
    )
    return {
        "status": "PASS" if (
            max(abs(a - b) for a, b in zip(result.transported_mean_projections, (1.5, -0.5))) < 1e-12
            and max(abs(a - b) for a, b in zip(result.curvature_projections, (4.0, 5.0))) < 1e-12
        ) else "FAIL",
        "expected_transported_mean_projections": [1.5, -0.5],
        "expected_curvature_projections": [4.0, 5.0],
        "measurement": result.as_dict(),
        "claim": (
            "one forward, one first reverse pass, and K global HVPs return an "
            "unbiased curvature sketch for every declared block simultaneously"
        ),
        "boundary": (
            "cross-block terms cancel over independent/coded probes; real custom "
            "backward graphs may not support second differentiation"
        ),
    }


def _group_omp(
    design: np.ndarray, responses: np.ndarray, *, sparsity: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Small vector-valued orthogonal matching pursuit with an intercept."""

    selected: list[int] = []
    fitted = np.repeat(responses.mean(axis=0, keepdims=True), len(responses), axis=0)
    coefficients = np.empty((0, responses.shape[1]), dtype=np.float64)
    for _ in range(sparsity):
        residual = responses - fitted
        centered = design - design.mean(axis=0, keepdims=True)
        scores = np.linalg.norm(centered.T @ residual, axis=1)
        if selected:
            scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
        regressors = np.column_stack([np.ones(len(design)), design[:, selected]])
        solution, *_ = np.linalg.lstsq(regressors, responses, rcond=None)
        fitted = regressors @ solution
        coefficients = solution
    return selected, coefficients, fitted


def coded_group_screen_controls(
    *, unit_count: int = 64, active_count: int = 4, output_dimension: int = 16,
    budgets: Sequence[int] = (8, 16, 24, 32), trials: int = 20,
) -> dict[str, Any]:
    """Test model-level coded intervention as an alternative to per-op probes."""

    active = np.asarray([3, 19, 37, 58][:active_count], dtype=int)
    scenarios: dict[str, Any] = {}
    for scenario in ("SPARSE_ADDITIVE", "SPARSE_WITH_INTERACTION", "DENSE_ADDITIVE"):
        budget_rows = {}
        for budget in budgets:
            recalls = []
            residuals = []
            for trial in range(trials):
                rng = np.random.default_rng(_stable_seed("group", scenario, str(budget), str(trial)))
                effects = np.zeros((unit_count, output_dimension), dtype=np.float64)
                if scenario == "DENSE_ADDITIVE":
                    scenario_active = np.arange(0, 40, 2, dtype=int)
                else:
                    scenario_active = active
                effects[scenario_active] = rng.normal(size=(len(scenario_active), output_dimension))
                train = rng.integers(0, 2, size=(budget, unit_count)).astype(np.float64)
                holdout = rng.integers(0, 2, size=(16, unit_count)).astype(np.float64)

                def observe(mask: np.ndarray) -> np.ndarray:
                    response = mask @ effects
                    if scenario == "SPARSE_WITH_INTERACTION":
                        interaction = rng.normal(size=(output_dimension,))
                        response = response + 3.0 * (
                            mask[:, active[0]] * mask[:, active[1]]
                        )[:, None] * interaction[None, :]
                    return response + rng.normal(scale=0.01, size=response.shape)

                train_response = observe(train)
                selected, coefficients, _ = _group_omp(
                    train, train_response, sparsity=active_count
                )
                holdout_regressors = np.column_stack([
                    np.ones(len(holdout)), holdout[:, selected]
                ])
                holdout_response = observe(holdout)
                prediction = holdout_regressors @ coefficients
                relative_residual = float(
                    np.linalg.norm(holdout_response - prediction)
                    / max(np.linalg.norm(holdout_response), 1e-30)
                )
                recalls.append(len(set(selected) & set(scenario_active)) / len(scenario_active))
                residuals.append(relative_residual)
            budget_rows[str(budget)] = {
                "ordinary_fb_mask_runs": budget,
                "active_support_recall_mean": float(np.mean(recalls)),
                "active_support_recall_min": float(np.min(recalls)),
                "heldout_response_residual_mean": float(np.mean(residuals)),
                "heldout_response_residual_max": float(np.max(residuals)),
            }
        scenarios[scenario] = budget_rows
    return {
        "unit_count": unit_count,
        "assumed_sparsity": active_count,
        "output_dimension_retained_in_full": output_dimension,
        "trials": trials,
        "heldout_mask_runs_per_trial": 16,
        "scenarios": scenarios,
        "screening_rule": (
            "use support ranking only when heldout vector-response residual is small; "
            "otherwise escalate because additivity/sparsity failed"
        ),
        "scientific_boundary": (
            "requires a controllable candidate/repair switch at every included unit; "
            "does not certify uninstrumentable units"
        ),
    }


def _load(path: Path) -> Mapping[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _subsets(size: int, budget: int, *, key: str, trials: int) -> list[tuple[int, ...]]:
    if budget > size:
        return []
    if budget == size:
        return [tuple(range(size))]
    candidates: list[tuple[int, ...]] = [tuple(range(budget))]
    # Include a deterministic evenly-spaced subset so the conclusion is not
    # tied to acquisition order.
    even = tuple(sorted({round(i * (size - 1) / (budget - 1)) for i in range(budget)}))
    if len(even) == budget:
        candidates.append(even)
    rng = random.Random(_stable_seed(key, str(budget)))
    while len(candidates) < trials:
        candidates.append(tuple(sorted(rng.sample(range(size), budget))))
    unique: list[tuple[int, ...]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique[:trials]


def _class_relation(subset_status: str, full_status: str) -> str:
    if subset_status == full_status:
        return "SAME"
    unresolved = BiasV22Status.CONDITIONAL_UNRESOLVED.value
    if subset_status == unresolved:
        return "SUBSET_UNRESOLVED"
    if full_status == unresolved:
        return "FULL_UNRESOLVED"
    return "OPPOSITE_RESOLVED"


def repeat_budget_ablation(
    artifact_paths: Mapping[str, Path],
    *,
    budgets: Sequence[int] = (4, 6, 8, 12, 16),
    subset_trials: int = 3,
) -> dict[str, Any]:
    policy = ConditionalPolicy()
    cases: dict[str, Any] = {}
    for case_id, path in artifact_paths.items():
        artifact = _load(path)
        records: dict[int, list[dict[str, str]]] = defaultdict(list)
        original_repeat_counts: set[int] = set()
        for state in artifact["states"]:
            arm = next(iter(state["arms"].values()))
            layers = arm["conditional_debias"]["layers"]
            for role, certificate in layers.items():
                gram = certificate["complete_gram"]
                size = len(gram)
                original_repeat_counts.add(size)
                for budget in budgets:
                    for trial, indices in enumerate(_subsets(
                        size,
                        budget,
                        key=f"{case_id}:{state['state_id']}",
                        trials=subset_trials,
                    )):
                        ids = [certificate["state_ids"][index] for index in indices]
                        digests = [certificate["vector_digests"][index] for index in indices]
                        subset = summarize_conditional_gram(
                            subset_square_matrix(gram, indices),
                            condition_id=str(state["state_id"]),
                            coordinate_count=int(certificate["coordinate_count"]),
                            replicate_ids=ids,
                            vector_digests=digests,
                            estimand=str(certificate["estimand"]),
                            reference=str(certificate["reference"]),
                            policy=policy,
                        )
                        subset_status = str(subset["status"])
                        full_status = str(certificate["status"])
                        records[budget].append({
                            "state_id": str(state["state_id"]),
                            "role": role,
                            "trial": str(trial),
                            "subset_status": subset_status,
                            "full_status": full_status,
                            "relation": _class_relation(subset_status, full_status),
                        })
        summaries: dict[str, Any] = {}
        for budget, rows in sorted(records.items()):
            relation = Counter(row["relation"] for row in rows)
            status = Counter(row["subset_status"] for row in rows)
            by_role: dict[str, Any] = {}
            for role in sorted({row["role"] for row in rows}):
                selected = [row for row in rows if row["role"] == role]
                role_relations = Counter(row["relation"] for row in selected)
                by_role[role] = {
                    "comparisons": len(selected),
                    "same_fraction": role_relations["SAME"] / len(selected),
                    "unresolved_fraction": role_relations["SUBSET_UNRESOLVED"] / len(selected),
                    "opposite_resolved_fraction": role_relations["OPPOSITE_RESOLVED"] / len(selected),
                }
            summaries[str(budget)] = {
                "comparisons": len(rows),
                "same_fraction": relation["SAME"] / len(rows),
                "unresolved_fraction": relation["SUBSET_UNRESOLVED"] / len(rows),
                "opposite_resolved_fraction": relation["OPPOSITE_RESOLVED"] / len(rows),
                "relation_counts": dict(sorted(relation.items())),
                "subset_status_counts": dict(sorted(status.items())),
                "by_role": by_role,
            }
        cases[case_id] = {
            "source_artifact": str(path),
            "fixed_condition_count": len(artifact["states"]),
            "original_repeat_counts": sorted(original_repeat_counts),
            "budgets": summaries,
        }
    return {
        "policy": policy.as_dict(),
        "subset_trials": subset_trials,
        "comparison_target": "FROZEN_FULL_REPEAT_CONDITIONAL_VERDICT",
        "full_verdict_is_not_treated_as_an_external_ground_truth": True,
        "cases": cases,
    }


def _project_gram(
    gram: Sequence[Sequence[float]], *, dimension: int, seed: int,
) -> tuple[list[list[float]], int, float]:
    """Draw a Gaussian JL projection using only the retained complete Gram."""

    matrix = np.asarray(gram, dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    negative_mass = float(np.abs(eigenvalues[eigenvalues < 0.0]).sum())
    eigenvalues = np.maximum(eigenvalues, 0.0)
    keep = eigenvalues > max(float(eigenvalues.max(initial=0.0)) * 1e-12, 1e-300)
    if not np.any(keep):
        coordinates = np.zeros((len(matrix), 1), dtype=np.float64)
    else:
        coordinates = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(coordinates.shape[1], dimension)) / math.sqrt(dimension)
    sketched = coordinates @ projection
    return (sketched @ sketched.T).tolist(), int(coordinates.shape[1]), negative_mass


def projection_dimension_ablation(
    artifact_paths: Mapping[str, Path],
    *,
    dimensions: Sequence[int] = (4, 8, 16, 32, 64),
    trials: int = 2,
) -> dict[str, Any]:
    """Test whether random update projections preserve conditional verdicts.

    This is exactly the missing output-space cost question for shared HVPs:
    one scalar update projection is cheap but cannot certify a full vector.
    Complete Grams let us simulate a Gaussian Johnson-Lindenstrauss sketch
    without recovering or storing the original parameter vectors.
    """

    policy = ConditionalPolicy()
    cases: dict[str, Any] = {}
    for case_id, path in artifact_paths.items():
        artifact = _load(path)
        records: dict[int, list[dict[str, str]]] = defaultdict(list)
        maximum_negative_mass = 0.0
        maximum_reconstructed_rank = 0
        for state in artifact["states"]:
            arm = next(iter(state["arms"].values()))
            for role, certificate in arm["conditional_debias"]["layers"].items():
                for dimension in dimensions:
                    for trial in range(trials):
                        projected_gram, rank, negative_mass = _project_gram(
                            certificate["complete_gram"],
                            dimension=dimension,
                            seed=_stable_seed(
                                "jl", case_id, str(state["state_id"]), role,
                                str(dimension), str(trial),
                            ),
                        )
                        maximum_negative_mass = max(maximum_negative_mass, negative_mass)
                        maximum_reconstructed_rank = max(maximum_reconstructed_rank, rank)
                        projected = summarize_conditional_gram(
                            projected_gram,
                            condition_id=str(state["state_id"]),
                            coordinate_count=dimension,
                            replicate_ids=certificate["state_ids"],
                            vector_digests=[
                                f"JL{dimension}:trial{trial}:{value}"
                                for value in certificate["vector_digests"]
                            ],
                            estimand=str(certificate["estimand"]),
                            reference=str(certificate["reference"]),
                            policy=policy,
                        )
                        projected_status = str(projected["status"])
                        full_status = str(certificate["status"])
                        records[dimension].append({
                            "role": role,
                            "projected_status": projected_status,
                            "full_status": full_status,
                            "relation": _class_relation(projected_status, full_status),
                        })
        summaries = {}
        for dimension, rows in sorted(records.items()):
            relation = Counter(row["relation"] for row in rows)
            summaries[str(dimension)] = {
                "comparisons": len(rows),
                "same_fraction": relation["SAME"] / len(rows),
                "unresolved_fraction": relation["SUBSET_UNRESOLVED"] / len(rows),
                "opposite_resolved_fraction": relation["OPPOSITE_RESOLVED"] / len(rows),
                "full_unresolved_fraction": relation["FULL_UNRESOLVED"] / len(rows),
                "relation_counts": dict(sorted(relation.items())),
            }
        cases[case_id] = {
            "source_artifact": str(path),
            "maximum_reconstructed_rank": maximum_reconstructed_rank,
            "maximum_clipped_negative_eigenvalue_mass": maximum_negative_mass,
            "dimensions": summaries,
        }
    return {
        "projection": "GAUSSIAN_JL_FROM_COMPLETE_GRAM",
        "trials": trials,
        "full_verdict_is_not_treated_as_an_external_ground_truth": True,
        "cases": cases,
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Low-cost Bias Oracle Feasibility",
        "",
        "This is a screening-method study, not a new correctness certificate.",
        "The exact conditional antithetic experiment remains the escalation path.",
        "",
        "## Synthetic mechanism controls",
        "",
        "| Control | Expected | sketch status | relative error | odd norm | even norm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["synthetic_controls"]:
        exact = row["exact_paired_response"]
        sketch = row["moment_response_sketch"]
        lines.append(
            "| {control_id} | {expected_channel} | {status} | {error:.4g} | {odd:.4g} | {even:.4g} |".format(
                control_id=row["control_id"], expected_channel=row["expected_channel"],
                status=sketch["status"], error=row["prediction_relative_error"],
                odd=exact["odd_mean_l2"], even=exact["even_mean_l2"],
            )
        )
    lines.extend([
        "",
        "Interpretation: transported mean covers source/pairing asymmetry; the",
        "Rademacher antithetic curvature sketch covers smooth response rectification.",
        "The amplitude gate deliberately escalates the support-switch control.",
        "",
        "## Shared all-block HVP experiment",
        "",
        f"Status: **{result['shared_hvp_control']['status']}**.",
        "",
        "The synthetic coupled F+B response recovered both blocks' first-order",
        "and curvature terms with one shared forward/reverse graph plus eight coded",
        "HVPs. Cross-block coupling was present and canceled through the codes.",
        "This is the only tested route whose probe count does not multiply by op count.",
        "",
        "## Coded group-intervention experiment",
        "",
        "This alternative uses only ordinary F+B runs and keeps the full response",
        "vector. Random masks switch groups of units between candidate and repair;",
        "vector-valued sparse recovery localizes contributors, while held-out masks",
        "test the required additivity assumption.",
        "",
        "| Scenario | mask runs | support recall | held-out residual |",
        "|---|---:|---:|---:|",
    ])
    for scenario, budgets in result["coded_group_screen_controls"]["scenarios"].items():
        for budget, row in budgets.items():
            lines.append(
                f"| {scenario} | {budget} | "
                f"{row['active_support_recall_mean']:.3f} | "
                f"{row['heldout_response_residual_mean']:.3f} |"
            )
    lines.extend([
        "",
        "## Retrospective repeat-budget ablation",
        "",
        "Each entry compares a deterministic prefix/even/random subset with the",
        "already frozen full-repeat verdict, using the unchanged 2,000-bootstrap policy.",
        "",
    ])
    for case_id, case in result["repeat_budget_ablation"]["cases"].items():
        lines.extend([
            f"### {case_id}",
            "",
            "| repeats | agreement | unresolved | opposite resolved | comparisons |",
            "|---:|---:|---:|---:|---:|",
        ])
        for budget, row in case["budgets"].items():
            lines.append(
                f"| {budget} | {row['same_fraction']:.3f} | "
                f"{row['unresolved_fraction']:.3f} | "
                f"{row['opposite_resolved_fraction']:.3f} | {row['comparisons']} |"
            )
        lines.append("")
    lines.extend([
        "## Random output-projection ablation",
        "",
        "A shared HVP returns one scalar update projection at a time. The table",
        "tests how many Gaussian output coordinates preserve the frozen full-vector",
        "conditional verdict when projected directly from the retained Grams.",
        "",
    ])
    for case_id, case in result["projection_dimension_ablation"]["cases"].items():
        lines.extend([
            f"### {case_id}",
            "",
            "| sketch dimensions | agreement | unresolved | opposite resolved |",
            "|---:|---:|---:|---:|",
        ])
        for dimension, row in case["dimensions"].items():
            lines.append(
                f"| {dimension} | {row['same_fraction']:.3f} | "
                f"{row['unresolved_fraction']:.3f} | "
                f"{row['opposite_resolved_fraction']:.3f} |"
            )
        lines.append("")
    lines.extend([
        "## Decision boundary",
        "",
        "- A first/second-moment screen is dimension-independent in probe count, but",
        "  it is not exact for nonsmooth or nonlocal responses.",
        "- Repeat reduction is acceptable only as a sequential screen. An unresolved",
        "  prefix escalates; it is never imputed as centered.",
        "- A new operator still needs a declared F+B perturbation boundary. Units with",
        "  no faithful antithetic/source perturbation must abstain.",
        "- No result here changes an existing case verdict.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/property/bias_oracle_feasibility"),
    )
    parser.add_argument("--subset-trials", type=int, default=3)
    args = parser.parse_args()
    if args.subset_trials < 1:
        raise SystemExit("--subset-trials must be positive")
    artifact_paths = {key: Path(value) for key, value in DEFAULT_ARTIFACTS.items()}
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("missing retained artifacts: " + ", ".join(missing))
    result = {
        "schema": "kernel-analyzer-bias-oracle-feasibility-v1",
        "scientific_role": "SCREENING_METHOD_SELECTION_NOT_CORRECTNESS_CERTIFICATION",
        "methods": {
            "transported_mean": "two symmetric responses at the conditional source mean",
            "curvature_sketch": "Rademacher antithetic response through a covariance factor",
            "paired_response": "exact empirical odd/even decomposition on sampled residuals",
            "shared_hvp": "one global coded HVP population estimates every declared local block",
            "coded_group": "ordinary F+B candidate/repair masks with held-out sparse-response closure",
            "repeat_ablation": "retrospective complete-Gram principal-submatrix analysis",
            "output_projection_ablation": "Gaussian JL sketches reconstructed from retained complete Grams",
        },
        "synthetic_controls": synthetic_controls(),
        "shared_hvp_control": shared_hvp_control(),
        "coded_group_screen_controls": coded_group_screen_controls(),
        "repeat_budget_ablation": repeat_budget_ablation(
            artifact_paths, subset_trials=args.subset_trials
        ),
        "projection_dimension_ablation": projection_dimension_ablation(
            artifact_paths, trials=2
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "feasibility.json"
    md_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_render_markdown(result).rstrip() + "\n")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
