#!/usr/bin/env python
"""Produce a nested-unit and boundary-conditioned report for oracle_gpu_pilot."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def exact_mcnemar_pvalue(up: int, down: int) -> float:
    discordant = up + down
    if discordant == 0:
        return 1.0
    tail = min(up, down)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nested_variance(records: list[dict[str, Any]]) -> dict[str, float]:
    observations: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in records:
        if row["pair"] != "eager_compiled":
            continue
        for case in row["comparison"]["cases"]:
            observations[(row["state_id"], int(case["case_index"]))].append(float(case["signed_margin_delta"]))
    state_cases: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for (state_id, case_index), values in observations.items():
        state_cases[state_id][case_index] = values
    all_values = np.asarray([value for values in observations.values() for value in values], dtype=np.float64)
    grand = float(all_values.mean())
    state_means = {
        state_id: float(np.mean([value for values in cases.values() for value in values]))
        for state_id, cases in state_cases.items()
    }
    state_component = float(np.mean([(value - grand) ** 2 for value in state_means.values()]))
    case_component_terms = []
    repeat_component_terms = []
    for state_id, cases in state_cases.items():
        for values in cases.values():
            case_mean = float(np.mean(values))
            case_component_terms.append((case_mean - state_means[state_id]) ** 2)
            repeat_component_terms.extend((value - case_mean) ** 2 for value in values)
    case_component = float(np.mean(case_component_terms))
    repeat_component = float(np.mean(repeat_component_terms))
    total = float(np.var(all_values))
    reconstructed = state_component + case_component + repeat_component
    return {
        "total_observation_variance": total,
        "between_state_component": state_component,
        "within_state_between_case_component": case_component,
        "within_case_repeat_component": repeat_component,
        "reconstructed_total": reconstructed,
        "identity_abs_error": abs(total - reconstructed),
        "between_state_share": state_component / total if total else 0.0,
        "within_state_between_case_share": case_component / total if total else 0.0,
        "within_case_repeat_share": repeat_component / total if total else 0.0,
        "sampling_uncertainty_is_not_a_component": True,
    }


def primary_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        if row["pair"] != "eager_compiled" or row["repeat"] != 0:
            continue
        for case in row["comparison"]["cases"]:
            output.append({**case, "state_id": row["state_id"], "stratum": row["stratum"]})
    return output


def semantic_by_stratum(cases: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    output = {}
    for stratum in sorted({case["stratum"] for case in cases}):
        selected = [case for case in cases if case["stratum"] == stratum]
        up = sum(bool(case["up_fork"]) for case in selected)
        down = sum(bool(case["down_fork"]) for case in selected)
        n = len(selected)
        output[stratum] = {
            "cases": n,
            "up_forks": up,
            "down_forks": down,
            "directional_shift": (up - down) / n,
            "paired_disagreement": (up + down) / n,
            "directional_exact_mcnemar_pvalue": exact_mcnemar_pvalue(up, down),
            "argmax_disagreement": sum(bool(case["argmax_disagreement"]) for case in selected) / n,
            "top2_order_disagreement": sum(case["top2_reference"] != case["top2_candidate"] for case in selected) / n,
            "top2_set_disagreement": sum(bool(case["top2_set_disagreement"]) for case in selected) / n,
            "mean_signed_margin_delta": float(np.mean([case["signed_margin_delta"] for case in selected])),
            "mean_abs_margin_delta": float(np.mean([abs(case["signed_margin_delta"]) for case in selected])),
        }
    return output


def boundary_profile(cases: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    edges = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, math.inf]
    output = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = [
            case
            for case in cases
            if float(case["distance_to_boundary"]) >= lower and float(case["distance_to_boundary"]) < upper
        ]
        if not selected:
            continue
        disagreements = sum(bool(case["reference_event"] != case["candidate_event"]) for case in selected)
        toward = sum(
            float(case["signed_margin_delta"]) * (1.0 if float(case["reference_margin"]) > 0 else -1.0) < 0
            for case in selected
            if float(case["reference_margin"]) != 0
        )
        nonzero_reference = sum(float(case["reference_margin"]) != 0 for case in selected)
        output.append(
            {
                "distance_bin": f"[{lower:g}, {upper:g})" if math.isfinite(upper) else f"[{lower:g}, inf)",
                "cases": len(selected),
                "disagreements": disagreements,
                "disagreement_rate": disagreements / len(selected),
                "mean_abs_margin_delta": float(np.mean([abs(case["signed_margin_delta"]) for case in selected])),
                "toward_boundary_rate": toward / nonzero_reference if nonzero_reference else float("nan"),
            }
        )
    return output


def semantic_definition_audit(cases: list[dict[str, Any]]) -> dict[str, int | float]:
    predicate = [case for case in cases if case["reference_event"] != case["candidate_event"]]
    argmax = [case for case in cases if case["argmax_disagreement"]]
    order = [case for case in cases if case["top2_reference"] != case["top2_candidate"]]
    top2_set = [case for case in cases if case["top2_set_disagreement"]]
    ties = [case for case in cases if float(case["reference_margin"]) == 0.0]
    up = sum(bool(case["up_fork"]) for case in cases)
    down = sum(bool(case["down_fork"]) for case in cases)
    return {
        "cases": len(cases),
        "reference_exact_class01_ties": len(ties),
        "class01_strict_predicate_disagreements": len(predicate),
        "class01_up_forks": up,
        "class01_down_forks": down,
        "directional_exact_mcnemar_pvalue": exact_mcnemar_pvalue(up, down),
        "argmax_disagreements": len(argmax),
        "top2_order_disagreements": len(order),
        "top2_set_disagreements": len(top2_set),
        "predicate_disagreements_without_argmax_change": sum(not case["argmax_disagreement"] for case in predicate),
        "argmax_disagreements_without_predicate_change": sum(
            case["reference_event"] == case["candidate_event"] for case in argmax
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["source_summary"]
    variance = payload["nested_variance"]
    lines = [
        "# Deterministic CUDA Oracle Pilot",
        "",
        "## Validity",
        "",
        f"- Candidate compiled-call evidence: `{summary['validity']['all_candidate_calls_reached_compiled_callable']}`",
        f"- Backend compiles: `{summary['validity']['backend_compiles']}`",
        f"- Runtime compiled invocations: `{summary['validity']['runtime_invocations']}`",
        f"- Nonzero self-pairs: `{summary['validity']['self_pair_nonzero_count']}`",
        "",
        "## Endpoint profile",
        "",
        f"- Global mean signed margin delta: `{summary['numerical']['mean_signed_margin_delta']:.8g}`",
        f"- Mean absolute margin delta: `{summary['numerical']['mean_abs_margin_delta']:.8g}`",
        f"- Directional semantic shift: `{summary['semantic']['directional_shift']:.8g}`",
        f"- Paired semantic disagreement: `{summary['semantic']['paired_disagreement']:.8g}`",
        f"- Argmax marginal total variation: `{summary['semantic']['argmax_marginal_total_variation']:.8g}`",
        f"- Mean one-step update L2 delta: `{summary['transition']['mean_update_l2_delta']:.8g}`",
        f"- Mean relative update L2 delta: `{summary['transition'].get('mean_relative_update_l2_delta', float('nan')):.8g}`",
        "",
        "These endpoints answer different questions. No global pass/fail threshold was declared for this descriptive pilot.",
        "",
        "## Semantic-definition audit",
        "",
        f"- Strict class-0 > class-1 predicate disagreements: `{payload['semantic_definition_audit']['class01_strict_predicate_disagreements']}`",
        f"- Exact McNemar p-value for directional asymmetry: `{payload['semantic_definition_audit']['directional_exact_mcnemar_pvalue']:.6g}`",
        f"- Argmax disagreements: `{payload['semantic_definition_audit']['argmax_disagreements']}`",
        f"- Ordered top-2 disagreements: `{payload['semantic_definition_audit']['top2_order_disagreements']}`",
        f"- Top-2 set disagreements: `{payload['semantic_definition_audit']['top2_set_disagreements']}`",
        f"- Predicate disagreements without argmax change: `{payload['semantic_definition_audit']['predicate_disagreements_without_argmax_change']}`",
        f"- Argmax disagreements without predicate change: `{payload['semantic_definition_audit']['argmax_disagreements_without_predicate_change']}`",
        "",
        "A signed two-class margin does not by itself encode program tie-breaking, ordered ranking, or top-k set semantics. The event map must be declared exactly.",
        "",
        "## Three different quantities often called variance",
        "",
        "| Quantity | Estimate | Share of observed delta variance |",
        "|---|---:|---:|",
        f"| Between-state heterogeneity | {variance['between_state_component']:.8g} | {variance['between_state_share']:.4f} |",
        f"| Within-state, between-case heterogeneity | {variance['within_state_between_case_component']:.8g} | {variance['within_state_between_case_share']:.4f} |",
        f"| Same-case repeat variability | {variance['within_case_repeat_component']:.8g} | {variance['within_case_repeat_share']:.4f} |",
        "",
        "The confidence intervals in `summary.json` are finite-state sampling uncertainty, not another data-generating variance component.",
        "",
        "## Semantic results by predeclared stratum",
        "",
        "| Stratum | Cases | Up | Down | Predicate disagreement | Argmax | Top-2 order | Top-2 set | Mean signed delta | Mean abs delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["semantic_by_stratum"].items():
        lines.append(
            f"| {name} | {row['cases']} | {row['up_forks']} | {row['down_forks']} | "
            f"{row['paired_disagreement']:.6g} | {row['argmax_disagreement']:.6g} | "
            f"{row['top2_order_disagreement']:.6g} | {row['top2_set_disagreement']:.6g} | "
            f"{row['mean_signed_margin_delta']:.6g} | {row['mean_abs_margin_delta']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Boundary-conditioned profile",
            "",
            "| Reference distance | Cases | Disagreements | Rate | Mean abs delta | Toward-boundary rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["boundary_profile"]:
        lines.append(
            f"| {row['distance_bin']} | {row['cases']} | {row['disagreements']} | "
            f"{row['disagreement_rate']:.6g} | {row['mean_abs_margin_delta']:.6g} | "
            f"{row['toward_boundary_rate']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This pilot identifies implementation-relative discrepancy on a controlled state distribution. "
            "The near-boundary stratum is intentionally enriched by construction, so its disagreement rate is "
            "not an estimate for a natural model workload. There is no independent mathematical specification, "
            "application acceptance threshold, or long-run training evidence here.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    records = load_jsonl(run_dir / "records.jsonl")
    cases = primary_cases(records)
    payload = {
        "source_summary": summary,
        "nested_variance": nested_variance(records),
        "semantic_definition_audit": semantic_definition_audit(cases),
        "semantic_by_stratum": semantic_by_stratum(cases),
        "boundary_profile": boundary_profile(cases),
    }
    (run_dir / "analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (run_dir / "REPORT.md").write_text(render_markdown(payload))
    print(json.dumps({key: value for key, value in payload.items() if key != "source_summary"}, indent=2))


if __name__ == "__main__":
    main()
