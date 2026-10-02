"""Reanalyse saved observations under one new aligned endpoint, without GPU runs.

This compares A/B with A/sqrt(B), keeping observations, alpha and a zero
decision margin identical. Historical reports are never rewritten. Fixed
benchmark Student intervals are sensitivity diagnostics, not new population
certificates. Missing per-observation statistics are explicitly inventoried.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from kernel_analyzer.bias_checker import _aligned_projection_interval, _student_interval

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text())


def interval_label(interval: dict) -> str:
    if interval.get("status") != "VALID":
        return "NOT_ASSESSED"
    lo, hi = interval["interval"]
    return "POSITIVE" if lo > 0 else "NEGATIVE" if hi < 0 else "NOT_CONFIRMED"


def compare(inner: list[float], energies: list[float], *, alpha: float) -> dict:
    new = _aligned_projection_interval(inner, energies, alpha=alpha)
    # Legacy checker filtered invalid reference energies. Keep that behavior
    # visible, rather than silently attributing this change to normalization.
    old_values = [a / b for a, b in zip(inner, energies)
                  if math.isfinite(a) and math.isfinite(b) and b > 1e-20]
    old = _student_interval(old_values, alpha=alpha)
    old_label, new_label = interval_label(old), interval_label(new)
    return {
        "unit_count": len(inner), "endpoint_alpha": alpha,
        "comparison_margin": 0.0,
        "legacy_gain": {**old, "label": old_label},
        "unit_projection": {**new, "label": new_label},
        "decision_changed": old_label != new_label,
        "same_observation_set": len(old_values) == len(inner),
        "interpretation": "different estimands and units; numerical means are not directly comparable",
    }


def adamw_histories(root: Path) -> list[dict]:
    specs = [
        ("adamw8bit_independent_histories", "numerical_coverage_v1/adamw8bit_population_direction_v1/units",
         [("default", "effect_repair_inner_product")], "DECLARED_IID_HISTORY_POPULATION"),
        ("adamw8bit_source_link", "bias_proof_plan_v1/adamw8bit_source_link/units",
         [("default", "default_effect_repair_inner_product"),
          ("compensated", "compensated_effect_repair_inner_product")],
         "POST_HOC_FROZEN_HISTORY_SOURCE_LINK_NOT_NEW_CONFIRMATION"),
    ]
    results = []
    for name, relative, arms, scope in specs:
        paths = sorted((root / "results/property" / relative).glob("unit-*.json"))
        if not paths:
            results.append({"case": name, "status": "MISSING_UNIT_RECORDS"})
            continue
        rows = [read(path) for path in paths]
        for arm, field in arms:
            results.append({
                "case": name, "arm": arm, "scope": scope,
                "sources": [str(path.relative_to(root)) for path in paths],
                "unit_ids": [row["unit_id"] for row in rows],
                **compare([row[field] for row in rows], [row["repair_energy"] for row in rows], alpha=.05),
            })
    return results


def sum_banks(root: Path) -> list[dict]:
    results = []
    for path in sorted((root / "results/property").glob("tilelang_sum_spec_experiment_v*.json")):
        if ".protocol." in path.name:
            continue
        data = read(path)
        for name, row in data["results"].items():
            distribution = "asymmetric" if name.endswith("_asymmetric") else "symmetric"
            census = {item["index"]: item for item in row["census"]}
            draws = data["protocol"]["draws"][distribution]
            refs = [census[i]["exact_sum"] for i in draws]
            effects = [census[i]["signed_error"] for i in draws]
            report = row["sampled_bias_report"]["output"]
            alpha = report.get("endpoint_alpha", data["protocol"]["case_alpha"] / 2)
            comparison = compare([e*r for e, r in zip(effects, refs)], [r*r for r in refs], alpha=alpha)
            old_saved = report["aligned_statewise_gain_interval"]
            if not math.isclose(comparison["legacy_gain"]["mean"], old_saved["mean"],
                                rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"saved draws do not reproduce legacy gain: {path.name}/{name}")
            results.append({
                "source": str(path.relative_to(root)), "case": name,
                "scope": "DECLARED_FINITE_BANK_SAME_SAVED_DRAWS",
                "finite_support_signed_mean": row["finite_support_signed_mean"],
                "finite_support_signed_mean_unchanged": True,
                "archived_gain_margin": data["protocol"].get("aligned_margin"),
                "margin_note": "normalization comparison uses zero on both sides; old nonzero gain margins are not transferred",
                **comparison,
            })
    return results


def exact_benchmark(root: Path) -> tuple[list[dict], list[dict]]:
    results, excluded = [], []
    for path in sorted((root / "results/property/generalization_benchmark_v1/raw").glob("*.json")):
        data = read(path)
        for stage, variants in data.get("stages", {}).items():
            identity = {"source": str(path.relative_to(root)), "case": data.get("case_id", path.stem), "stage": stage}
            if "EXACT" not in variants:
                excluded.append({**identity, "reason": "ONLY_SKETCH_GEOMETRY_NOT_ORIGINAL_COORDINATE_STATISTICS"})
                continue
            variant = variants["EXACT"]
            gram = variant.get("profile", variant)["suite"]["joint_gram"]
            n = len(gram["repair_repair"])
            results.append({
                **identity, "scope": "FIXED_SUITE_RETROSPECTIVE_SENSITIVITY_ONLY",
                "original_report_verdict_replaced": False,
                "population_confirmation": False,
                "alpha_note": "common checker endpoint alpha for sensitivity, not a newly preregistered benchmark family",
                **compare([gram["effect_repair"][i][i] for i in range(n)],
                          [gram["repair_repair"][i][i] for i in range(n)], alpha=.025),
            })
    return results, excluded


def legacy_stages(value, path=""):
    if isinstance(value, dict):
        if "aligned_statewise_gain_interval" in value:
            yield path, value
        for key, child in value.items():
            if "stages" in value and key in ("output", "backward"):
                continue  # convenience aliases, not extra observations
            yield from legacy_stages(child, path + "/" + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from legacy_stages(child, path + "/" + str(i))


def summary_only_inventory(root: Path) -> list[dict]:
    base = root / "results/property"
    paths = set(base.glob("bias_checker_examples_*.json")) | set(base.glob("tilelang_*.json"))
    paths |= set((base / "new_problem_group_search_v1").glob("mainstream_candidate_screen*.json"))
    results = []
    for path in sorted(paths):
        if path.name.startswith("tilelang_sum_spec_experiment_"):
            continue  # independently recovered above from census and draw indices
        stages = list(legacy_stages(read(path)))
        if stages:
            results.append({
                "source": str(path.relative_to(root)),
                "status": "CANNOT_RECOMPUTE_FROM_AGGREGATES",
                "stage_count": len(stages),
                "stages": [{"path": key, "sample_count": row.get("sample_count"),
                            "recorded_decision": row.get("decision"),
                            "zero_observed_effect_energy": row.get("total_effect_energy") == 0.0}
                           for key, row in stages],
                "missing": "per-observation inner products and reference energies, or raw paired vectors",
                "action": "retain historical labels with old estimand; recollect only if a new-endpoint claim is needed",
            })
    return results


def audit(root: Path = ROOT) -> dict:
    histories = adamw_histories(root)
    banks = sum_banks(root)
    benchmark, excluded = exact_benchmark(root)
    missing = summary_only_inventory(root)
    return {
        "schema": "aligned-projection-migration-audit-v1",
        "new_gpu_measurements": False,
        "historical_artifacts_modified": False,
        "legacy_estimand": "E[dot(u,r)/dot(r,r)]",
        "selected_estimand": "E[dot(u,r)/sqrt(dot(r,r))]",
        "method_scope": "same conditional Student intervals; this migration does not validate iid, tail, or zero-variance assumptions",
        "unaffected_by_definition_change": ["source interventions", "raw effect vectors and energies",
                                            "fixed-direction projections", "paired training loss observations"],
        "summary": {
            "history_endpoint_comparisons": len(histories),
            "history_endpoint_changes": sum(row.get("decision_changed", False) for row in histories),
            "finite_bank_endpoint_comparisons": len(banks),
            "finite_bank_endpoint_changes": sum(row["decision_changed"] for row in banks),
            "exact_benchmark_sensitivity_comparisons": len(benchmark),
            "exact_benchmark_sensitivity_changes": sum(row["decision_changed"] for row in benchmark),
            "excluded_sketch_only_stages": len(excluded),
            "summary_only_artifacts": len(missing),
            "summary_only_stages": sum(row["stage_count"] for row in missing),
        },
        "history_comparisons": histories,
        "finite_bank_comparisons": banks,
        "exact_benchmark_sensitivity": benchmark,
        "excluded_sketch_only_stages": excluded,
        "unrecomputable_checker_artifacts": missing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    if not path.is_relative_to(ROOT) or path.exists():
        parser.error("output must be a new path inside this repository")
    result = audit()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
