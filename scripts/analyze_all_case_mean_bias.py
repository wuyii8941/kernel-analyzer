#!/usr/bin/env python3
"""Audit mean-bias evidence for every retained problem group.

The audit deliberately keeps two estimands separate:

* independent-history and empirical-bank mean tests retain their different
  population scopes and assumptions;
* fixed-suite Gram/profile records remain descriptive unless an explicit
  calibration/confirmation sampling protocol is present.

No nonzero norm, positive-count rule, or cumulative trajectory is promoted to
an iid population statement.  This is an analysis of retained artifacts, not a
new GPU measurement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/root_cause_closure_v1/all_case_mean_bias_audit_v1.json"


def read(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text())


def fixed_gram_metrics(gram: list[list[float]], calibration: int | None = None) -> dict[str, Any]:
    n = len(gram)
    if n == 0 or any(len(row) != n for row in gram):
        raise ValueError("Gram matrix must be nonempty and square")
    if any(not math.isfinite(float(x)) for row in gram for x in row):
        raise ValueError("Gram matrix contains a non-finite value")
    total = math.fsum(float(x) for row in gram for x in row)
    diagonal = math.fsum(float(gram[i][i]) for i in range(n))
    off_diagonal = total - diagonal
    mean_norm_sq = max(0.0, total / (n * n))
    # The off-diagonal U-statistic is the finite-suite analogue of the
    # population quantity ||E[u]||^2 when units are iid.  For a fixed suite it
    # is descriptive only; unlike the diagonal energy it is not automatically
    # positive.  Keeping it explicit prevents a nonzero RMS (which is always
    # produced by nonzero observations) from being called mean bias.
    cross_u = off_diagonal / (n * (n - 1)) if n > 1 else None
    result: dict[str, Any] = {
        "state_count": n,
        "fixed_suite_mean_vector_norm_squared": mean_norm_sq,
        "fixed_suite_mean_vector_norm": math.sqrt(mean_norm_sq),
        "fixed_suite_cross_state_u_statistic": cross_u,
        "fixed_suite_cross_state_u_positive": bool(cross_u is not None and cross_u > 0.0),
        "fixed_suite_diagonal_energy": diagonal / n,
        "scope": "DESCRIPTIVE_FIXED_SUITE_ONLY",
        "population_mean_bias_decision": "NOT_ASSESSED_FIXED_SUITE",
        "zero_population_mean_proven": False,
    }
    if calibration is None:
        return result
    confirmation = n - calibration
    if calibration <= 0 or confirmation <= 0:
        raise ValueError("calibration split must be nonempty")
    cal_sum = math.fsum(
        float(gram[i][j]) for i in range(calibration) for j in range(calibration)
    )
    cal_norm = math.sqrt(max(0.0, cal_sum / calibration**2))
    result["calibration_state_count"] = calibration
    result["confirmation_state_count"] = confirmation
    result["calibration_mean_norm"] = cal_norm
    if cal_norm == 0.0:
        result.update({
            "direction_status": "NOT_IDENTIFIABLE",
            "confirmation_projection_mean": None,
            "confirmation_positive_count": None,
            "confirmation_negative_count": None,
            "confirmation_zero_count": None,
        })
        return result
    projections = [
        math.fsum(float(gram[j][i]) for i in range(calibration))
        / (calibration * cal_norm)
        for j in range(calibration, n)
    ]
    result.update({
        "direction_status": "IDENTIFIED",
        "confirmation_projection_mean": math.fsum(projections) / confirmation,
        "confirmation_positive_count": sum(x > 0.0 for x in projections),
        "confirmation_negative_count": sum(x < 0.0 for x in projections),
        "confirmation_zero_count": sum(x == 0.0 for x in projections),
    })
    return result


def profile_metrics(profile: dict[str, Any], geometry: str) -> dict[str, Any]:
    suite = profile["suite"]
    gram = suite["joint_gram"]["effect_effect"]
    result = fixed_gram_metrics(gram, int(suite.get("calibration_state_count", 0)) or None)
    result.update({
        "geometry": geometry,
        "total_effect_rms": suite.get("total_effect_rms"),
        "repair_rms": suite.get("repair_rms"),
        "repair_aligned_effect": suite.get("repair_aligned_effect"),
        "status_in_artifact": profile.get("status"),
    })
    return result


def raw_stage_metrics(path: str, stages: tuple[str, ...] = ("PARAMETER_GRADIENT", "ADAMW_UPDATE")) -> dict[str, Any]:
    raw = read(path)
    output: dict[str, Any] = {"source": path, "state_count": len(raw.get("state_ids", []))}
    for stage in stages:
        variants = raw.get("stages", {}).get(stage, {})
        if not variants:
            continue
        geometry = "EXACT" if "EXACT" in variants else sorted(variants)[0]
        output[stage] = profile_metrics(variants[geometry]["profile"], geometry)
    return output


def benchmark_fixed_suite_mean_evidence() -> list[dict[str, Any]]:
    """Collect exact-Gram mean-vector descriptions for every frozen case.

    These records are deliberately kept separate from the iid population
    tests below.  A complete 32-state Gram determines the observed-suite mean
    vector norm, but the frozen benchmark states are not a random sample of a
    larger training population.
    """
    summary_path = ROOT / "results/property/generalization_benchmark_v1/summary.json"
    raw_dir = ROOT / "results/property/generalization_benchmark_v1/raw"
    if not summary_path.is_file() or not raw_dir.is_dir():
        return []
    summary = read(str(summary_path.relative_to(ROOT)))
    output: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        data = read(str(path.relative_to(ROOT)))
        case_id = str(data.get("case_id", path.stem))
        case_meta = summary.get("cases", {}).get(case_id, {})
        for stage, variants in data.get("stages", {}).items():
            if not isinstance(variants, dict) or "EXACT" not in variants:
                continue
            profile = variants["EXACT"].get("profile", variants["EXACT"])
            try:
                metric = profile_metrics(profile, "EXACT")
            except (KeyError, TypeError, ValueError):
                continue
            metric.update({
                "case_id": case_id,
                "family": case_meta.get("family"),
                "model": case_meta.get("model"),
                "carrier": data.get("carrier"),
                "stage": stage,
                "source": str(path.relative_to(ROOT)),
                "mean_vector_nonzero_on_observed_suite": bool(
                    metric["fixed_suite_mean_vector_norm"] > 0.0
                ),
                "scope": "DESCRIPTIVE_FIXED_BENCHMARK_SUITE_ONLY",
            })
            output.append(metric)
    return output


def benchmark_empirical_bank_mean_tests(
    fixed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Test a frozen direction on the finite benchmark bank.

    The benchmark states are treated as a finite, declared empirical
    distribution and sampled with replacement.  This is useful for checking
    directional recurrence without pretending that the benchmark is a
    random sample of all training states.  The test is intentionally run for
    every exact-Gram row (not only rows that look positive) to avoid outcome
    based filtering.
    """
    import numpy as np
    from scipy.stats import t

    calibration_count = 32
    confirmation_count = 64
    base_seed = 20261000
    results: list[dict[str, Any]] = []
    for row_index, row in enumerate(fixed_rows):
        source = str(row["source"])
        data = read(source)
        profile = data["stages"][row["stage"]]["EXACT"].get("profile", data["stages"][row["stage"]]["EXACT"])
        gram = np.asarray(profile["suite"]["joint_gram"]["effect_effect"], dtype=np.float64)
        n_bank = int(gram.shape[0])
        rng = np.random.default_rng(base_seed + row_index)
        indices = rng.integers(0, n_bank, size=calibration_count + confirmation_count)
        calibration = indices[:calibration_count]
        confirmation = indices[calibration_count:]
        calibration_gram = gram[np.ix_(calibration, calibration)]
        direction_norm_sq = float(calibration_gram.sum() / calibration_count**2)
        item: dict[str, Any] = {
            "case_id": row["case_id"],
            "family": row.get("family"),
            "model": row.get("model"),
            "stage": row["stage"],
            "source": source,
            "bank_state_count": n_bank,
            "calibration_count": calibration_count,
            "confirmation_count": confirmation_count,
            "draw_seed": base_seed + row_index,
            "scope": "RETROSPECTIVE_IID_WITH_REPLACEMENT_FROM_DECLARED_BENCHMARK_BANK",
            "assumptions": [
                "the retained complete original-coordinate Gram represents the declared finite bank",
                "with-replacement index draws are independent conditional on that bank",
                "the calibration direction is fixed before confirmation",
                "finite variance and Student-t approximation for the signed projection",
            ],
        }
        if not math.isfinite(direction_norm_sq) or direction_norm_sq <= 0:
            item.update({"status": "DIRECTION_NOT_IDENTIFIABLE", "implies_vector_mean_nonzero": False})
            results.append(item)
            continue
        direction_norm = math.sqrt(direction_norm_sq)
        values = np.asarray([
            float(gram[int(index), calibration].sum() / (calibration_count * direction_norm))
            for index in confirmation
        ], dtype=np.float64)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        two_sided_half = float(t.ppf(0.975, confirmation_count - 1)) * sd / math.sqrt(confirmation_count)
        one_sided_half = float(t.ppf(0.95, confirmation_count - 1)) * sd / math.sqrt(confirmation_count)
        two_sided = [mean - two_sided_half, mean + two_sided_half]
        one_sided_lower = mean - one_sided_half
        item.update({
            "status": "PROJECTED_MEAN_SUPPORTED" if one_sided_lower > 0 else "PROJECTED_MEAN_NOT_CONFIRMED",
            "calibration_direction_norm": direction_norm,
            "confirmation_projection_mean": mean,
            "confirmation_projection_sd": sd,
            "confirmation_projection_two_sided_95": two_sided,
            "confirmation_projection_one_sided_95_lower": one_sided_lower,
            "two_sided_95_nonzero_supported": bool(two_sided[0] > 0 or two_sided[1] < 0),
            "positive_mean_supported_one_sided": bool(one_sided_lower > 0),
            "confirmation_positive_count": int(np.count_nonzero(values > 0)),
            "confirmation_negative_count": int(np.count_nonzero(values < 0)),
            "confirmation_zero_count": int(np.count_nonzero(values == 0)),
            "finite_bank_mean_vector_norm": math.sqrt(max(0.0, float(gram.sum() / n_bank**2))),
            "implies_vector_mean_nonzero_if_bound_positive": bool(one_sided_lower > 0),
        })
        results.append(item)
    return results


def mean_of_aligned_ratios(paths: list[str]) -> dict[str, Any]:
    values: list[float] = []
    for path in paths:
        data = read(path)
        values.append(float(data["aligned_ratio"]))
    if not values:
        raise ValueError("no population units")
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
    sd = math.sqrt(variance)
    # scipy is an optional project extra; this report is run in the statistics
    # environment and fails loudly instead of silently substituting a normal
    # quantile for a small-sample t interval.
    from scipy.stats import t
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    one_sided_half = float(t.ppf(0.95, n - 1)) * sd / math.sqrt(n)
    signs = {"positive": sum(x > 0 for x in values),
             "negative": sum(x < 0 for x in values),
             "zero": sum(x == 0 for x in values)}
    return {
        "estimand": "MEAN_STATEWISE_ALIGNED_WRITE_PROJECTION",
        "independent_unit_count": n,
        "mean": mean,
        "sample_sd": sd,
        "two_sided_95_interval": [mean - half, mean + half],
        "one_sided_95_lower_bound": mean - one_sided_half,
        "two_sided_95_nonzero_supported": bool(mean - half > 0.0 or mean + half < 0.0),
        "positive_mean_supported_one_sided": bool(mean - one_sided_half > 0.0),
        "sign_counts": signs,
        "mean_positive_under_student_t": bool(mean - one_sided_half > 0.0),
        "scope": "DECLARED_IID_HISTORY_POPULATION",
        "assumptions": [
            "independent histories sampled with replacement under the declared protocol",
            "finite variance and Student-t approximation for the signed aligned ratio",
            "directional aligned estimand, not a vector mean or total-energy certificate",
        ],
    }


def mean_of_probe_aligned_values(
    path: str,
    variant: str,
    *,
    scope: str = "DECLARED_IID_WITH_REPLACEMENT_FROM_DEEPSEEK_TRAJECTORY_BANK",
) -> dict[str, Any]:
    """Summarize a with-replacement natural-state probe's aligned projections.

    The probe has already sampled indices from a declared finite bank. This
    computes the signed mean and interval without treating that bank as an
    unrestricted natural-training population.
    """
    data = read(path)
    rows = data.get("rows", [])
    values = [float(row[f"write_aligned_over_{variant}"]) for row in rows]
    if len(values) < 2:
        raise ValueError("probe must contain at least two aligned observations")
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
    sd = math.sqrt(variance)
    from scipy.stats import t
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    one_sided_half = float(t.ppf(0.95, n - 1)) * sd / math.sqrt(n)
    interval = [mean - half, mean + half]
    return {
        "estimand": "MEAN_STATEWISE_ALIGNED_WRITE_PROJECTION",
        "independent_unit_count": n,
        "mean": mean,
        "sample_sd": sd,
        "two_sided_95_interval": interval,
        "one_sided_95_lower_bound": mean - one_sided_half,
        "one_sided_95_upper_bound": mean + one_sided_half,
        "two_sided_95_nonzero_supported": bool(interval[0] > 0.0 or interval[1] < 0.0),
        "positive_mean_supported_one_sided": bool(mean - one_sided_half > 0.0),
        "negative_mean_supported_one_sided": bool(mean + one_sided_half < 0.0),
        "sign_counts": {
            "positive": sum(x > 0.0 for x in values),
            "negative": sum(x < 0.0 for x in values),
            "zero": sum(x == 0.0 for x in values),
        },
        "scope": scope,
        "source": path,
        "variant": variant,
        "assumptions": [
            "the retained trajectory bank is the declared finite empirical population",
            "the listed state indices are sampled with replacement conditional on that bank",
            "the signed aligned projection is the estimand; this is not an unrestricted natural-training population claim",
            "finite variance and Student-t approximation for the signed projection",
        ],
    }


def build() -> dict[str, Any]:
    population_units = sorted(
        str(p) for p in (ROOT / "results/property/numerical_coverage_v1/adamw8bit_population_direction_v1/units").glob("*.json")
    )
    # Keep the group-level table aligned with the active root-cause ledger.
    ledger = read("results/property/case_causal_audit_v1/root_cause_closure_current.json")
    rows: list[dict[str, Any]] = []
    for group in ledger["rows"]:
        rows.append({
            "problem_group": group["problem_group"],
            "root_closure": group["closure"],
            "case_role": group.get("case_role", "ACTIVE_BIAS_CASE"),
        })
    by_group = {row["problem_group"]: row for row in rows}

    by_group["adamw8bit_moment_quantization"].update({
        "mean_bias_status": "SUPPORTED_ALIGNED_MEAN_UNDER_DECLARED_IID_HISTORY",
        "population_test": mean_of_aligned_ratios([
            str(p) for p in sorted((ROOT / "results/property/numerical_coverage_v1/adamw8bit_population_direction_v1/units").glob("*.json"))
        ]),
        "interpretation": "The declared independent history population has a positive mean statewise aligned write projection; this is not a vector-mean proof.",
    })

    # Existing exact original-coordinate Gram/profile records.  These are
    # intentionally reported as fixed-suite descriptions, even when an old
    # artifact used an inference label.
    fixed_sources = {
        "liger_fused_linear_ce_dw_accumulation": [
            ("results/property/liger_fp32_chunk_order_v1/length64_confirmation.json", "PARAMETER_GRADIENT"),
            ("results/property/liger_fp32_chunk_order_v1/length64_confirmation.json", "ADAMW_UPDATE"),
            ("results/property/liger_fp32_chunk_order_v1/length256_confirmation.json", "PARAMETER_GRADIENT"),
            ("results/property/liger_fp32_chunk_order_v1/length256_confirmation.json", "ADAMW_UPDATE"),
        ],
        "softmax_saved_state_backward": [
            ("results/coverage/cases/qwen128_softmax_fb_formal.json", "semantic_total"),
            ("results/coverage/cases/qwen128_softmax_fb_formal.json", "saved_state_reconstruction"),
        ],
        "granite_moe_expert_contribution_order": [
            ("results/property/training_numerical_analysis_v2/granite_expert_order_confirmation_v2/raw.json", "PARAMETER_WRITE"),
        ],
    }
    # Additional exact MM/source decompositions are retained as a compact
    # family-level appendix, not merged into a cross-model population.
    mm_sources = [
        ("results/coverage/cases/qwen64_vproj.json", "precision_mechanism", "kernel"),
        ("results/coverage/cases/qwen64_vproj.json", "precision_mechanism", "output_rounding"),
        ("results/coverage/cases/mamba_seq64_input_proj.json", "precision_mechanism", "kernel"),
        ("results/coverage/cases/mamba_seq64_input_proj.json", "precision_mechanism", "output_rounding"),
        ("results/coverage/cases/phi4_seq64_lmhead_dx.json", "precision_mechanism", "kernel"),
        ("results/coverage/cases/phi4_seq64_lmhead_dx.json", "causal_repair", "final_norm_weight_carrier"),
    ]
    mm_metrics = []
    for path, parent, key in mm_sources:
        parent_value = read(path).get(parent, {})
        # Some historical decomposition artifacts store only scalar
        # U-statistics at this path (for example Qwen's output-rounding
        # record), not an original-coordinate Gram matrix.  Do not mistake
        # the scalar record for a vector mean; retain only entries with a
        # complete Gram here and let the case-level audit describe the scalar
        # mechanism separately.
        if not isinstance(parent_value, dict) or key not in parent_value:
            continue
        data = parent_value[key]
        if "gram" not in data:
            continue
        metric = fixed_gram_metrics(data["gram"], 16 if len(data["gram"]) == 32 else None)
        metric.update({"source": path, "component": key, "geometry": "EXACT"})
        mm_metrics.append(metric)
    by_group["mm_gemm_output_and_accumulation"].update({
        "mean_bias_status": "FIXED_SUITE_COMPONENT_MEAN_DESCRIPTIVE_CASE_LEVEL",
        "fixed_suite_components": mm_metrics,
        "interpretation": "Several named MM components have nonzero observed-set mean vectors or held-out projections; no cross-model population mean is claimed.",
    })

    for group, sources in fixed_sources.items():
        entries = []
        for path, stage in sources:
            data = read(path)
            if stage in ("semantic_total", "saved_state_reconstruction"):
                data = data["direction"][stage]
                metric = fixed_gram_metrics(data["gram"], 16)
                metric.update({"source": path, "component": stage, "geometry": "EXACT"})
            elif path.endswith("raw.json"):
                variants = data["stages"][stage]
                geometry = "EXACT" if "EXACT" in variants else sorted(variants)[0]
                metric = profile_metrics(variants[geometry]["profile"], geometry)
                metric["source"] = path
            else:
                metric = profile_metrics(data["profiles"][stage], "PERIODIC_LINEAR_SUMMARY")
                metric["source"] = path
            entries.append(metric)
        status = "FIXED_SUITE_MEAN_DESCRIPTIVE_NOT_POPULATION" if entries else "NO_VALID_MEAN_STATISTIC"
        by_group[group].update({
            "mean_bias_status": status,
            "fixed_suite_evidence": entries,
            "interpretation": "The retained Gram/profile describes the observed suite only; old population labels are not accepted by this audit.",
        })

    # The RMSNorm cast-order intervention has complete natural-model rows but
    # was not sampled from a declared iid population.  Keep its aligned write
    # evidence as a fixed-suite description rather than silently promoting the
    # normal intervals to a population guarantee.
    rmsnorm_path = ROOT / "results/property/root_cause_closure_v1/rmsnorm_cast_materialization_natural_v1.json"
    if rmsnorm_path.is_file() and "rmsnorm_cast_materialization" in by_group:
        rmsnorm = read(str(rmsnorm_path.relative_to(ROOT)))
        by_group["rmsnorm_cast_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [
                {
                    "model": row["model"],
                    "state_count": row["states"],
                    "stage": "PARAMETER_WRITE",
                    "aligned_mean": row["write_aligned_mean"],
                    "aligned_normal_95_interval": row["write_aligned_normal_95_interval"],
                    "write_effect_rms_mean_over_reference": row["write_effect_rms_mean_over_reference"],
                    "source": row["artifact"],
                    "scope": "DESCRIPTIVE_NATURAL_MODEL_FIXED_SUITE",
                }
                for row in rmsnorm["models"]
            ],
            "interpretation": (
                "Both real-model fixed suites show aligned parameter-write shrinkage for the declared "
                "cast-order reference. The state sets are not an iid population, so this remains a "
                "descriptive natural-model result rather than a population mean-bias certificate."
            ),
        })

    # The BERT LayerNorm probe is a real-checkpoint, fixed input-bank result,
    # not an iid population sample.  Keep its aligned write signal descriptive
    # and retain the explicit FP32 zero-difference control separately.
    bert_path = ROOT / "results/property/new_problem_group_search_v1/bert_tiny_layernorm_eager_natural_32_20260918.json"
    bert_fp32_path = ROOT / "results/property/new_problem_group_search_v1/bert_tiny_layernorm_fp32_reference_20260918.json"
    if bert_path.is_file() and "bert_layernorm_compiled_materialization" in by_group:
        bert = read(str(bert_path.relative_to(ROOT)))
        fp32 = read(str(bert_fp32_path.relative_to(ROOT))) if bert_fp32_path.is_file() else None
        by_group["bert_layernorm_compiled_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE",
            "fixed_suite_evidence": [{
                "model": bert["model"],
                "operator": bert["operator"],
                "state_count": bert["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": bert["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": bert["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": bert["summary"]["write_effect_rms_mean"],
                "source": str(bert_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_REAL_CHECKPOINT_FIXED_SUITE",
            }],
            "source_control": {
                "reference": fp32["reference"] if fp32 else None,
                "gradient_effect_rms_mean": fp32["summary"]["gradient_effect_rms_mean"] if fp32 else None,
                "write_effect_rms_mean": fp32["summary"]["write_effect_rms_mean"] if fp32 else None,
                "scope": "EXPLICIT_FP32_CONTROL_ON_SAME_DECLARED_BANK",
            },
            "interpretation": (
                "The compiled-versus-eager LayerNorm comparison has a fixed-suite aligned write "
                "signal. The explicit FP32 reference is a source control, not a population test; "
                "the result is not promoted to a natural-state mean-bias certificate."
            ),
        })

    # BERT embedding-sum materialization: a real-checkpoint fixed-suite
    # comparison of native-dtype word/position/token-type addition against an
    # otherwise identical FP32 three-term sum.  Keep it descriptive; the
    # document-derived windows are not an iid population sample.
    bert_embedding_path = ROOT / "results/property/new_problem_group_search_v1/bert_embedding_sum_materialization_natural_32_20260918.json"
    if bert_embedding_path.is_file() and "bert_embedding_sum_materialization" in by_group:
        bert_embedding = read(str(bert_embedding_path.relative_to(ROOT)))
        by_group["bert_embedding_sum_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": bert_embedding["model"],
                "operator": bert_embedding["operator"],
                "parameter": bert_embedding["parameter"],
                "state_count": bert_embedding["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": bert_embedding["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": bert_embedding["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": bert_embedding["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": bert_embedding["summary"].get("confirmation_projection_interval_normal_95"),
                "source": str(bert_embedding_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_REAL_CHECKPOINT_FIXED_SUITE",
            }],
            "interpretation": (
                "The BERT embedding block shows a stable negative aligned write effect when "
                "the three embedding terms are accumulated in native dtype rather than summed "
                "in FP32 and cast once. The held-out additive direction crosses zero, so this "
                "remains a scoped fixed-suite result rather than a population vector-mean claim."
            ),
        })

    # BERT pooler tanh: the expanded 32-window screen has a positive held-out
    # additive gradient direction on the embedding carrier.  The pooler-dense
    # carrier is retained as a same-boundary contrast and crosses zero.
    bert_tanh_path = ROOT / "results/property/new_problem_group_search_v1/bert_pooler_tanh_natural_cpu32_20260919.json"
    bert_tanh_dense_path = ROOT / "results/property/new_problem_group_search_v1/bert_pooler_tanh_dense_natural_cpu32_20260919.json"
    if bert_tanh_path.is_file() and bert_tanh_dense_path.is_file() and "bert_pooler_tanh_materialization" in by_group:
        bert_tanh = read(str(bert_tanh_path.relative_to(ROOT)))
        bert_tanh_dense = read(str(bert_tanh_dense_path.relative_to(ROOT)))
        by_group["bert_pooler_tanh_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ADDITIVE_DIRECTION_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": {
                "source": str(bert_tanh_path.relative_to(ROOT)),
                "dense_contrast_source": str(bert_tanh_dense_path.relative_to(ROOT)),
                "state_count": bert_tanh["state_count"],
                "stage": "PARAMETER_GRADIENT",
                "effect_rms_mean_over_reference": bert_tanh["summary"]["gradient_effect_rms_mean"],
                "heldout_additive_direction_interval": bert_tanh["summary"]["confirmation_projection_interval_normal_95"],
                "dense_contrast_additive_direction_interval": bert_tanh_dense["summary"]["confirmation_projection_interval_normal_95"],
                "scope": "DESCRIPTIVE_REAL_BERT_CHECKPOINT_FIXED_SUITE",
            },
            "interpretation": (
                "The native-dtype versus FP32 pooler-tanh boundary has a positive held-out additive "
                "gradient direction at the embedding carrier on the declared 32-state BERT-tiny suite. "
                "The same intervention at pooler.dense.weight crosses zero, so this is a carrier-specific "
                "fixed-suite directional result rather than a population or all-parameter mean-bias claim."
            ),
        })

    # Granite MoE attention residual addition: keep the isolated native-dtype
    # versus FP32-addition comparison as a descriptive fixed-suite result.
    granite_residual_path = ROOT / "results/property/new_problem_group_search_v1/granite_residual_addition_materialization_natural_32_20260918.json"
    if granite_residual_path.is_file() and "granite_residual_addition_materialization" in by_group:
        granite_residual = read(str(granite_residual_path.relative_to(ROOT)))
        by_group["granite_residual_addition_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": granite_residual["model"],
                "operator": granite_residual["operator"],
                "layer": granite_residual["layer"],
                "parameter": granite_residual["parameter"],
                "state_count": granite_residual["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": granite_residual["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": granite_residual["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": granite_residual["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": granite_residual["summary"].get("confirmation_projection_interval_normal_95"),
                "source": str(granite_residual_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GRANITE_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only the Granite attention residual addition precision produces stable "
                "negative aligned write scaling on the declared text bank. The held-out additive "
                "direction crosses zero, so this is not a population vector-mean certificate."
            ),
        })

    # Granite MoE gate-times-up product: a fixed-suite natural result with
    # routing and expert projections held constant.
    granite_gate_path = ROOT / "results/property/new_problem_group_search_v1/granite_moe_gate_product_materialization_natural_32_20260918.json"
    if granite_gate_path.is_file() and "granite_moe_gate_product_materialization" in by_group:
        granite_gate = read(str(granite_gate_path.relative_to(ROOT)))
        by_group["granite_moe_gate_product_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_AND_DIRECTIONAL_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": granite_gate["model"],
                "operator": granite_gate["operator"],
                "layer": granite_gate["layer"],
                "parameter": granite_gate["parameter"],
                "state_count": granite_gate["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": granite_gate["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": granite_gate["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": granite_gate["summary"]["write_effect_rms_mean"],
                "confirmation_projection_interval": granite_gate["summary"].get("confirmation_projection_interval_normal_95"),
                "source": str(granite_gate_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GRANITE_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "The isolated Granite MoE gate-times-up product has both stable negative aligned "
                "scaling and a positive held-out projection in this fixed real-text suite. Neither "
                "is promoted to a population guarantee without independent state sampling."
            ),
        })

    # Granite MoE expert-output times routing-gate product: another isolated
    # fixed-suite boundary, distinct from the gate-times-up product.
    granite_output_gate_path = ROOT / "results/property/new_problem_group_search_v1/granite_moe_output_gate_materialization_natural_32_20260918.json"
    if granite_output_gate_path.is_file() and "granite_moe_output_gate_materialization" in by_group:
        granite_output_gate = read(str(granite_output_gate_path.relative_to(ROOT)))
        by_group["granite_moe_output_gate_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": granite_output_gate["model"],
                "operator": granite_output_gate["operator"],
                "layer": granite_output_gate["layer"],
                "parameter": granite_output_gate["parameter"],
                "state_count": granite_output_gate["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": granite_output_gate["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": granite_output_gate["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": granite_output_gate["summary"]["write_effect_rms_mean"],
                "confirmation_projection_interval": granite_output_gate["summary"].get("confirmation_projection_interval_normal_95"),
                "source": str(granite_output_gate_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GRANITE_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only the expert-output times routing-gate product precision produces "
                "stable negative aligned write scaling on the declared Granite text bank. The "
                "held-out additive direction crosses zero, so this remains a fixed-suite result."
            ),
        })

    # Gemma-3 vision patch convolution: a same-input, one-variable FP32
    # accumulation reference on a real image/text bank.  This is a descriptive
    # fixed-suite aligned result; the held-out additive direction is kept
    # separate and is not promoted to a vector-mean population claim.
    conv_path = ROOT / "results/property/new_problem_group_search_v1/gemma3_conv_clean_natural_26_20260918.json"
    if conv_path.is_file() and "gemma3_vision_patch_convolution" in by_group:
        conv = read(str(conv_path.relative_to(ROOT)))
        by_group["gemma3_vision_patch_convolution"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": conv["model"],
                "operator": conv["operator"],
                "module": conv["module"],
                "state_count": conv["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": conv["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": conv["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": conv["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": conv["summary"]["projection_interval_normal_95"],
                "source": str(conv_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_REAL_IMAGE_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "The native BF16 patch convolution has a large, consistently negative aligned "
                "write scaling relative to the same-input FP32-accumulation reference on the "
                "declared real image/text bank. The additive direction interval crosses zero, "
                "so this is not a vector-mean bias certificate or a loss-quality claim."
            ),
        })

    # Qwen3-VL learned position interpolation: native-dtype weighted summation
    # versus FP32 weighted summation on the same real image/text loss path.
    position_path = ROOT / "results/property/new_problem_group_search_v1/qwen3vl_position_interpolation_training_16_20260918.json"
    if position_path.is_file() and "qwen3vl_position_interpolation" in by_group:
        position = read(str(position_path.relative_to(ROOT)))
        ratios = [float(row["aligned_ratio"]) for row in position["rows"]]
        mean = math.fsum(ratios) / len(ratios)
        sd = math.sqrt(math.fsum((value - mean) ** 2 for value in ratios) / (len(ratios) - 1))
        half_width = 2.145 * sd / math.sqrt(len(ratios))
        by_group["qwen3vl_position_interpolation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": position["model"],
                "target_parameter": position["target_parameter"],
                "state_count": position["image_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": mean,
                "aligned_normal_95_interval": [mean - half_width, mean + half_width],
                "write_effect_rms_mean_over_reference": position["fixed_suite_mean_write_rms"],
                "additive_projection_positive_count": position["confirmation_positive_count"],
                "additive_projection_negative_count": position["confirmation_negative_count"],
                "source": str(position_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_REAL_IMAGE_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "The native-dtype four-tap learned-position interpolation has a stable negative "
                "aligned write scaling relative to the same-input FP32 weighted-sum reference on "
                "the declared Qwen3-VL image/text bank. The held-out additive direction is mixed, "
                "so this is not a vector-mean population certificate or a long-run quality claim."
            ),
        })

    # DeepSeek/Qwen3 embedding backward accumulation: native repeated-token
    # accumulation versus a same-input FP32 index-add reference.  Keep this
    # as a descriptive real-checkpoint fixed-suite result; no population or
    # quality claim is inferred from it.
    embedding_path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_backward_24_20260918.json"
    if embedding_path.is_file() and "deepseek_embedding_backward_accumulation" in by_group:
        embedding = read(str(embedding_path.relative_to(ROOT)))
        by_group["deepseek_embedding_backward_accumulation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": embedding["model"],
                "operator": embedding["operator"],
                "state_count": embedding["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": embedding["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": embedding["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": embedding["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": embedding["summary"]["projection_interval_normal_95"],
                "source": str(embedding_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_REAL_CHECKPOINT_FIXED_SUITE",
            }],
            "interpretation": (
                "Native repeated-token embedding backward accumulation has a stable negative aligned "
                "write scaling relative to the same-input FP32 index-add reference on the declared bank. "
                "The forward loss is identical and the additive direction interval crosses zero; this "
                "is not a population or quality claim."
            ),
        })

    # Gemma-4 causal NLL: same logits, native cross-entropy versus explicit
    # FP32 shifted log-softmax/gather.  This is a real-checkpoint fixed-suite
    # description of the loss-evaluation boundary, not a population test.
    gemma4_nll_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_softcapped_nll_24_20260918.json"
    if gemma4_nll_path.is_file() and "gemma4_causal_nll_loss_evaluation" in by_group:
        nll = read(str(gemma4_nll_path.relative_to(ROOT)))
        by_group["gemma4_causal_nll_loss_evaluation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": nll["model"],
                "operator": nll["operator"],
                "state_count": nll["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": nll["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": nll["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": nll["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": nll["summary"]["projection_interval_normal_95"],
                "source": str(gemma4_nll_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "The native cross-entropy loss path has a stable negative aligned write scaling "
                "relative to explicit FP32 log-softmax/gather on identical logits in the declared "
                "Gemma-4 text bank. The additive direction interval crosses zero, so this is not "
                "a vector-mean, population or loss-quality certificate."
            ),
        })

    # Gemma-4 final-logit soft-cap: hidden state and lm_head are shared, and
    # only the final divide/tanh/multiply materialization changes.  The
    # expanded 26-state replay supports a descriptive fixed-suite aligned
    # endpoint, not a population or loss-quality conclusion.
    gemma4_softcap_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_final_logit_softcap_lm_head_confirmation_26_20260919.json"
    if gemma4_softcap_path.is_file() and "gemma4_final_logit_softcap_materialization" in by_group:
        softcap = read(str(gemma4_softcap_path.relative_to(ROOT)))
        by_group["gemma4_final_logit_softcap_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": softcap["model"],
                "operator": softcap["operator"],
                "state_count": softcap["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": softcap["summary"]["confirmation_aligned_mean"],
                "aligned_normal_95_interval": softcap["summary"]["confirmation_aligned_interval_normal_95"],
                "write_effect_rms_mean_over_reference": softcap["summary"]["write_effect_rms_mean"],
                "source": str(gemma4_softcap_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "The native BF16 final-logit soft-cap arithmetic has a stable negative aligned "
                "lm_head write scaling relative to the same-hidden-state FP32 reference on the "
                "declared Gemma-4 text bank. This is a fixed-suite source-boundary result, not a "
                "population, additive-vector or loss-quality certificate."
            ),
        })

    # Gemma-4 audio output projection: full audio-language loss on real speech.
    # This is a fixed-suite natural boundary result, not a population test.
    gemma4_audio_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_language_output_projection_natural_16_20260919.json"
    if gemma4_audio_path.is_file() and "gemma4_audio_output_projection_materialization" in by_group:
        audio = read(str(gemma4_audio_path.relative_to(ROOT)))
        summary = audio["summary"]
        by_group["gemma4_audio_output_projection_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE",
            "fixed_suite_evidence": [{
                "model": audio["model"],
                "operator_family": audio["operator_family"],
                "parameter": audio["parameter"],
                "state_count": summary["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": summary["confirmation_aligned_write_mean"],
                "aligned_normal_95_interval": summary["confirmation_aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": summary["write_effect_rms_mean"],
                "loss_interval": summary["loss_difference_interval_normal_95"],
                "source": str(gemma4_audio_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_AUDIO_LANGUAGE_FIXED_SUITE",
            }],
            "interpretation": (
                "The native BF16 audio output projection has a large fixed-suite write effect "
                "relative to the same-input FP32 reference under a full audio-language loss on "
                "real speech. The loss interval and held-out projection interval cross zero, so "
                "this is not a population mean-bias or long-run quality certificate."
            ),
        })

    # Gemma-4 audio attention logit soft-cap: a single-variable natural
    # boundary intervention inside layer 0.  The total write effect is large,
    # but the held-out aligned direction is heterogeneous; keep this as a
    # source-closed fixed-suite total-effect record rather than promoting it
    # to a population mean-bias claim.
    gemma4_audio_attention_softcap_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_attention_softcap_layer0_natural_16_20260920.json"
    if gemma4_audio_attention_softcap_path.is_file() and "gemma4_audio_attention_softcap_materialization" in by_group:
        softcap = read(str(gemma4_audio_attention_softcap_path.relative_to(ROOT)))
        summary = softcap["summary"]
        softcap_rows = softcap["rows"]
        split = int(summary["calibration_count"])
        confirmation = softcap_rows[split:]
        def interval(values):
            if len(values) < 2:
                return [float(values[0]), float(values[0])]
            mean = math.fsum(float(value) for value in values) / len(values)
            variance = math.fsum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
            half = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
            return [mean - half, mean + half]
        bf16_confirmation = [row["write_aligned_bf16"] for row in confirmation]
        fp64_confirmation = [row["write_aligned_fp64"] for row in confirmation]
        by_group["gemma4_audio_attention_softcap_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_TOTAL_EFFECT_SOURCE_CLOSED_DIRECTION_NOT_CONFIRMED",
            "fixed_suite_evidence": [{
                "model": softcap["model"],
                "operator_family": softcap["operator_family"],
                "layer": softcap["layer"],
                "parameter": softcap["target_parameter"],
                "state_count": summary["state_count"],
                "stage": "PARAMETER_WRITE",
                "bf16_write_effect_rms_mean": summary["bf16_write_effect_rms_mean"],
                "fp64_write_effect_rms_mean": summary["fp64_write_effect_rms_mean"],
                "bf16_full_suite_aligned_interval": summary["bf16_aligned_interval"],
                "fp64_full_suite_aligned_interval": summary["fp64_aligned_interval"],
                "bf16_confirmation_aligned_interval": interval(bf16_confirmation),
                "fp64_confirmation_aligned_interval": interval(fp64_confirmation),
                "bf16_confirmation_positive_count": sum(value > 0 for value in bf16_confirmation),
                "bf16_confirmation_negative_count": sum(value < 0 for value in bf16_confirmation),
                "fp64_confirmation_positive_count": sum(value > 0 for value in fp64_confirmation),
                "fp64_confirmation_negative_count": sum(value < 0 for value in fp64_confirmation),
                "source": str(gemma4_audio_attention_softcap_path.relative_to(ROOT)),
                "scope": "SOURCE_CLOSED_GEMMA4_REAL_AUDIO_LANGUAGE_FIXED_SUITE_TOTAL_EFFECT",
            }],
            "interpretation": (
                "Changing only the layer-0 audio-attention soft-cap arithmetic produces a large,"
                " nonzero q_proj gradient and first-step write effect on every state in the real"
                " audio-language bank. The full-suite aligned interval is negative, but the held-out"
                " confirmation aligned intervals cross zero; this supports a source-closed fixed-suite"
                " total-effect result, not a population directional-bias or quality certificate."
            ),
        })

    # Gemma-4 audio attention probability materialization: retain the native
    # FP32 softmax probabilities versus materializing them to BF16 immediately
    # before the value contraction.  This is a layer-0-scoped fixed-suite
    # aligned endpoint; the layer-1 repeat is retained as heterogeneity evidence.
    gemma4_audio_attention_softmax_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_natural_16_20260920.json"
    if gemma4_audio_attention_softmax_path.is_file() and "gemma4_audio_attention_softmax_probability_materialization" in by_group:
        softmax = read(str(gemma4_audio_attention_softmax_path.relative_to(ROOT)))
        layer1_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_attention_weights_fp32_materialization_layer1_natural_16_20260920.json"
        layer1 = read(str(layer1_path.relative_to(ROOT))) if layer1_path.is_file() else None
        by_group["gemma4_audio_attention_softmax_probability_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE",
            "fixed_suite_aligned_evidence": [{
                "model": softmax["model"],
                "operator_family": softmax["operator_family"],
                "layer": softmax["layer"],
                "parameter": softmax["target_parameter"],
                "state_count": softmax["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "write_effect_rms_mean": softmax["summary"]["write_effect_rms_mean"],
                "aligned_interval_normal_95": softmax["summary"]["write_aligned_interval_normal_95"],
                "loss_interval": softmax["summary"]["loss_difference_interval_normal_95"],
                "layer1_control": layer1["summary"] if layer1 is not None else None,
                "source": str(gemma4_audio_attention_softmax_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_AUDIO_LAYER0_FIXED_SUITE",
            }],
            "interpretation": (
                "Retaining native FP32 attention probabilities rather than materializing them "
                "to BF16 before value contraction gives a strictly negative layer-0 q_proj "
                "aligned-write interval on the declared real-speech bank. The layer-1 repeat "
                "crosses zero, so the result is layer-scoped and descriptive, not a population "
                "or audio-family-wide mean-bias certificate."
            ),
        })

    # Gemma-4 audio LightConv GLU product: the same real speech and full
    # audio-language loss, with only the gate-times-value product promoted to
    # FP32 before the original-dtype write-back.  Keep this as a fixed-suite
    # descriptive endpoint; the held-out additive projection interval crosses
    # zero, so this is not a population mean-bias certificate.
    gemma4_audio_glu_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_glu_layer0_natural_16_20260919.json"
    if gemma4_audio_glu_path.is_file() and "gemma4_audio_lightconv_glu_product_materialization" in by_group:
        glu = read(str(gemma4_audio_glu_path.relative_to(ROOT)))
        summary = glu["summary"]
        by_group["gemma4_audio_lightconv_glu_product_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE",
            "fixed_suite_evidence": [{
                "model": glu["model"],
                "operator_family": glu["operator_family"],
                "parameter": glu["parameter"],
                "state_count": summary["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": summary["confirmation_aligned_write_mean"],
                "aligned_normal_95_interval": summary["confirmation_aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": summary["write_effect_rms_mean"],
                "loss_interval": summary["loss_difference_interval_normal_95"],
                "source": str(gemma4_audio_glu_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_AUDIO_LANGUAGE_FIXED_SUITE",
            }],
            "interpretation": (
                "The native-dtype LightConv GLU product has a large fixed-suite write effect "
                "relative to the same-input FP32-product reference on real speech. The aligned "
                "write interval is entirely negative, while the held-out additive projection and "
                "loss intervals cross zero; this is not a population mean-bias or long-run quality "
                "certificate."
            ),
        })

    # Gemma-4 audio LightConv depthwise Conv1d: use the protocol's
    # ratio-of-sums aligned estimand rather than the mean of statewise ratios.
    gemma4_audio_depthwise_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_depthwise_conv_layer0_natural_16_20260920.json"
    if gemma4_audio_depthwise_path.is_file() and "gemma4_audio_lightconv_depthwise_conv_backward_accumulation" in by_group:
        depthwise = read(str(gemma4_audio_depthwise_path.relative_to(ROOT)))
        summary = depthwise["summary"]
        by_group["gemma4_audio_lightconv_depthwise_conv_backward_accumulation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE_RATIO_OF_SUMS",
            "fixed_suite_evidence": [{
                "model": depthwise["model"],
                "operator_family": depthwise["operator_family"],
                "parameter": depthwise["parameter"],
                "state_count": summary["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": summary["confirmation_aligned_write_ratio_of_sums"],
                "aligned_statewise_mean": summary["confirmation_aligned_write_mean"],
                "aligned_normal_95_interval": summary["confirmation_aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": summary["write_effect_rms_mean"],
                "loss_interval": summary["loss_difference_interval_normal_95"],
                "source": str(gemma4_audio_depthwise_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_AUDIO_LANGUAGE_FIXED_SUITE_RATIO_OF_SUMS",
            }],
            "interpretation": (
                "The native depthwise Conv1d backward boundary has a heterogeneous but nonzero "
                "fixed-suite write effect relative to the same-window explicit FP32 reference. "
                "The protocol ratio-of-sums aligned scaling is positive and remains positive when "
                "any one confirmation state is deleted, while the statewise interval and held-out "
                "additive projection cross zero. This is a fixed-suite aligned-bias description, "
                "not a population mean-bias or long-run quality certificate."
            ),
        })

    # Gemma-4 audio subsampling Conv2d layer-1: use the protocol's
    # ratio-of-sums aligned estimand on the rerun with raw numerators.
    gemma4_audio_subsample_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_raw_16_20260920.json"
    if gemma4_audio_subsample_path.is_file() and "gemma4_audio_subsampling_convolution_materialization" in by_group:
        subsample = read(str(gemma4_audio_subsample_path.relative_to(ROOT)))
        summary = subsample["summary"]
        by_group["gemma4_audio_subsampling_convolution_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_AUDIO_LANGUAGE_RATIO_OF_SUMS",
            "fixed_suite_evidence": [{
                "model": subsample["model"],
                "operator_family": subsample["operator_family"],
                "parameter": subsample["parameter"],
                "state_count": summary["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": summary["confirmation_aligned_write_ratio_of_sums"],
                "aligned_statewise_mean": summary["confirmation_aligned_write_mean"],
                "aligned_normal_95_interval": summary["confirmation_aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": summary["write_effect_rms_mean"],
                "loss_interval": summary["loss_difference_interval_normal_95"],
                "source": str(gemma4_audio_subsample_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_AUDIO_LANGUAGE_FIXED_SUITE_RATIO_OF_SUMS",
            }],
            "interpretation": (
                "The native layer-1 audio subsampling Conv2d has a large fixed-suite write effect "
                "relative to the same-input FP32-accumulate/BF16-write reference. The protocol "
                "ratio-of-sums aligned scaling is negative and robust to deleting any one confirmation "
                "state, while the statewise interval, held-out additive projection and loss interval "
                "cross zero; this is not a population mean-bias or long-run quality certificate."
            ),
        })

    # DeBERTa disentangled relative attention: two layers of the same
    # checkpoint provide a cross-layer fixed-suite source-boundary check.
    deberta_paths = [
        ROOT / "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_layer0_seed0_32_20260919.json",
        ROOT / "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_layer5_32_20260919.json",
    ]
    if all(path.is_file() for path in deberta_paths) and "deberta_disentangled_relative_attention_materialization" in by_group:
        deberta = [read(str(path.relative_to(ROOT))) for path in deberta_paths]
        by_group["deberta_disentangled_relative_attention_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": item["model"],
                "layer": item["layer"],
                "operator": item["operator"],
                "parameter": item["target"],
                "state_count": item["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": item["summary"]["write_aligned_mean"],
                "aligned_normal_95_interval": item["summary"]["write_aligned_interval_normal_95"],
                "write_effect_rms_mean_over_reference": item["summary"]["write_effect_rms_mean"],
                "loss_interval": item["summary"]["loss_difference_interval_normal_95"],
                "source": str(path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_DEBERTA_REAL_TEXT_FIXED_SUITE",
            } for item, path in zip(deberta, deberta_paths)],
            "interpretation": (
                "Native c2p/p2c disentangled relative-attention score materialization has a stable "
                "negative aligned write effect at both tested layers. The auxiliary MLM head is newly "
                "initialized in this local checkpoint, so no pretrained MLM quality claim is made."
            ),
        })

    # Bloom ALiBi: two attention layers provide the analogous cross-layer
    # check for a different positional-attention family.
    bloom_paths = [
        ROOT / "results/property/new_problem_group_search_v1/bloom_alibi_attention_materialization_natural_16_20260919.json",
        ROOT / "results/property/new_problem_group_search_v1/bloom_alibi_attention_materialization_natural_layer11_16_20260919.json",
    ]
    if all(path.is_file() for path in bloom_paths) and "bloom_alibi_attention_materialization" in by_group:
        bloom = [read(str(path.relative_to(ROOT))) for path in bloom_paths]
        by_group["bloom_alibi_attention_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": item["model"],
                "layer": item["layer"],
                "operator": item["operator"],
                "parameter": item["target"],
                "state_count": item["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": item["summary"]["write_aligned_mean"],
                "aligned_normal_95_interval": item["summary"]["write_aligned_interval_normal_95"],
                "write_effect_rms_mean_over_reference": item["summary"]["write_effect_rms_mean"],
                "loss_interval": item["summary"]["loss_difference_interval_normal_95"],
                "source": str(path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_BLOOM_REAL_TEXT_FIXED_SUITE",
            } for item, path in zip(bloom, bloom_paths)],
            "interpretation": (
                "Native ALiBi score materialization has a stable negative aligned write effect at both "
                "tested layers. This is a fixed-suite positional-attention boundary result, not a "
                "population or long-run loss-quality certificate."
            ),
        })

    # Gemma-4 RMSNorm row-reduction order: same-input FP32 reduction order
    # intervention.  Keep this as a descriptive fixed-suite aligned result;
    # it is distinct from the existing cast-before-weight multiplication
    # group and does not establish a released kernel's internal order.
    gemma4_row_reduction_path = ROOT / "results/property/new_problem_group_search_v1/gemma4_rms_row_reduction_natural_26_20260919.json"
    if gemma4_row_reduction_path.is_file() and "gemma4_rms_row_reduction_order" in by_group:
        reduction = read(str(gemma4_row_reduction_path.relative_to(ROOT)))
        by_group["gemma4_rms_row_reduction_order"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": reduction["model"],
                "operator": "Gemma-4 RMSNorm row square-sum reduction",
                "parameter": reduction["target"],
                "state_count": reduction["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": reduction["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": reduction["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": reduction["summary"]["write_effect_rms_mean"],
                "loss_interval": reduction["summary"]["loss_difference_interval_normal_95"],
                "source": str(gemma4_row_reduction_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_GEMMA4_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "The native and reversed FP32 row square-sum orders have a stable negative "
                "aligned write scaling on the declared Gemma-4 bank. The loss interval crosses "
                "zero, and the result is not a population or released-kernel internal-order claim."
            ),
        })

    # Qwen3 eager versus SDPA is a path-preserving natural boundary result.
    # Keep it as a fixed-suite descriptive aligned effect; the backend's
    # lower-level arithmetic is not yet decomposed and no population claim is
    # implied by the normal interval.
    qwen3_attention_path = ROOT / "results/property/new_problem_group_search_v1/qwen3_attention_backend_isolated_layer13_qproj_16_20260918.json"
    if qwen3_attention_path.is_file() and "qwen3_attention_sdpa_eager_backend" in by_group:
        attention = read(str(qwen3_attention_path.relative_to(ROOT)))
        by_group["qwen3_attention_sdpa_eager_backend"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": attention["model"],
                "operator": attention["operator"],
                "state_count": attention["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": sum(row["write_aligned"] for row in attention["rows"]) / len(attention["rows"]),
                "aligned_normal_95_interval": attention["summary"]["write_aligned_interval_normal_95"],
                "write_effect_rms_mean_over_reference": attention["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": attention["summary"]["heldout_write_projection_interval_normal_95"],
                "source": str(qwen3_attention_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_QWEN3_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Replacing the eager attention interface with the SDPA interface changes the aligned "
                "parameter write on the declared Qwen3 bank. This is a boundary-level descriptive result; "
                "the lower-level backend arithmetic and any population or loss consequence remain open."
            ),
        })

    # Mamba softplus materialisation: same sequential recurrence, only the
    # precision used for delta softplus changes.  Keep this as a descriptive
    # real-checkpoint aligned result; the frozen banks are not a population.
    mamba_softplus_path = ROOT / "results/property/new_problem_group_search_v1/mamba_softplus_materialization_natural_32_20260918.json"
    if mamba_softplus_path.is_file() and "mamba_softplus_materialization" in by_group:
        softplus = read(str(mamba_softplus_path.relative_to(ROOT)))
        by_group["mamba_softplus_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": softplus["model"],
                "operator": softplus["operator"],
                "parameter": softplus["parameter"],
                "state_count": softplus["summary"]["state_count"] if "state_count" in softplus["summary"] else softplus["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": softplus["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": softplus["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": softplus["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": softplus["summary"]["confirmation_projection_interval_normal_95"],
                "source": str(mamba_softplus_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_MAMBA_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only delta softplus precision produces stable negative aligned write "
                "scaling on the declared Mamba text bank. The held-out vector-direction interval "
                "crosses zero, so this is not a full additive mean-bias or population certificate."
            ),
        })

    # Mamba discrete transition materialisation: keep this separate from the
    # softplus, convolution, and state-output probes because the only changed
    # boundary is the exponent argument used to form exp(A * delta).
    mamba_transition_path = ROOT / "results/property/new_problem_group_search_v1/mamba_discrete_transition_materialization_natural_cpu16_20260919.json"
    if mamba_transition_path.is_file() and "mamba_discrete_transition_materialization" in by_group:
        transition = read(str(mamba_transition_path.relative_to(ROOT)))
        by_group["mamba_discrete_transition_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": transition["model"],
                "operator": transition["operator"],
                "parameter": transition["parameter"],
                "state_count": transition["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": transition["summary"]["aligned_write_mean_confirmation"],
                "aligned_normal_95_interval": transition["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": transition["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": transition["summary"]["heldout_projection_interval_normal_95"],
                "source": str(mamba_transition_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_MAMBA_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only the exponent-argument materialisation in the discrete transition "
                "produces stable negative aligned write scaling on the declared Mamba text bank. "
                "The held-out vector-direction interval crosses zero, so this is not a full "
                "additive mean-bias or population certificate."
            ),
        })

    # Mamba causal depthwise convolution: a same-input FP32 accumulation
    # reference with the original native-dtype activation.  This is a new
    # natural operator family, although it shares the broad accumulation-
    # precision mechanism with the Gemma patch-convolution case.
    mamba_conv_path = ROOT / "results/property/new_problem_group_search_v1/mamba_causal_conv_materialization_natural_32_20260918.json"
    if mamba_conv_path.is_file() and "mamba_causal_conv_accumulation" in by_group:
        conv = read(str(mamba_conv_path.relative_to(ROOT)))
        by_group["mamba_causal_conv_accumulation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_AND_DIRECTIONAL_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": conv["model"],
                "operator": conv["operator"],
                "parameter": conv["parameter"],
                "state_count": conv["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": conv["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": conv["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": conv["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": conv["summary"]["confirmation_projection_interval_normal_95"],
                "source": str(mamba_conv_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_MAMBA_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only causal convolution accumulation precision produces both a negative "
                "aligned write effect and a negative held-out calibration-direction projection on "
                "the declared 32-state Mamba text bank. This supports a scoped natural directional "
                "and aligned result, not a population or loss-quality certificate."
            ),
        })

    mamba_state_output_path = ROOT / "results/property/new_problem_group_search_v1/mamba_state_output_contraction_natural_32_20260918.json"
    if mamba_state_output_path.is_file() and "mamba_state_output_contraction_materialization" in by_group:
        state_output = read(str(mamba_state_output_path.relative_to(ROOT)))
        by_group["mamba_state_output_contraction_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": state_output["model"],
                "operator": state_output["operator"],
                "parameter": state_output["parameter"],
                "state_count": state_output["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": state_output["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": state_output["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": state_output["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": state_output["summary"]["confirmation_projection_interval_normal_95"],
                "source": str(mamba_state_output_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_MAMBA_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only state-to-C output contraction precision produces stable negative "
                "aligned write scaling on the declared Mamba text bank. The held-out vector-direction "
                "interval crosses zero, so this is not a full additive mean-bias or population certificate."
            ),
        })

    mamba_d_skip_path = ROOT / "results/property/new_problem_group_search_v1/mamba_d_skip_materialization_natural_32_20260918.json"
    if mamba_d_skip_path.is_file() and "mamba_d_skip_materialization" in by_group:
        d_skip = read(str(mamba_d_skip_path.relative_to(ROOT)))
        by_group["mamba_d_skip_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": d_skip["model"],
                "operator": d_skip["operator"],
                "parameter": d_skip["parameter"],
                "state_count": d_skip["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": d_skip["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": d_skip["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": d_skip["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": d_skip["summary"]["confirmation_projection_interval_normal_95"],
                "source": str(mamba_d_skip_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_MAMBA_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only the residual D skip-product precision produces stable negative "
                "aligned write scaling on the declared Mamba text bank. The held-out vector-direction "
                "interval crosses zero, so this is not a full additive mean-bias or population certificate."
            ),
        })

    mamba_z_gate_path = ROOT / "results/property/new_problem_group_search_v1/mamba_z_gate_materialization_natural_32_20260918.json"
    if mamba_z_gate_path.is_file() and "mamba_z_gate_materialization" in by_group:
        z_gate = read(str(mamba_z_gate_path.relative_to(ROOT)))
        by_group["mamba_z_gate_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": z_gate["model"],
                "operator": z_gate["operator"],
                "layer": z_gate["layer"],
                "parameter": z_gate["parameter"],
                "state_count": z_gate["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": z_gate["summary"]["aligned_write_mean"],
                "aligned_normal_95_interval": z_gate["summary"]["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": z_gate["summary"]["write_effect_rms_mean"],
                "additive_projection_interval": z_gate["summary"].get("confirmation_projection_interval_normal_95"),
                "source": str(mamba_z_gate_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_MAMBA_REAL_TEXT_FIXED_SUITE",
            }],
            "interpretation": (
                "Changing only the Mamba scan-output z-gate product precision produces stable "
                "negative aligned write scaling on the declared text bank. The held-out additive "
                "direction crosses zero, so this remains a fixed-suite aligned result."
            ),
        })

    mamba_scan_cut_path = ROOT / "results/property/new_problem_group_search_v1/mamba_scan_source_cut_layer3_16_20260920.json"
    if mamba_scan_cut_path.is_file() and "mamba_fused_selective_scan_reassociation" in by_group:
        scan_cut = read(str(mamba_scan_cut_path.relative_to(ROOT)))
        fused_summary = scan_cut["summary"]
        by_group["mamba_fused_selective_scan_reassociation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": [{
                "model": scan_cut["model"],
                "operator": "actual mamba_ssm fused selective-scan",
                "layer": scan_cut["layer"],
                "parameter": scan_cut["target_parameter"],
                "state_count": scan_cut["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": fused_summary["fused_aligned_mean"],
                "aligned_normal_95_interval": fused_summary["fused_aligned_interval_normal_95"],
                "write_effect_rms_mean_over_reference": sum(
                    row["fused_write_effect_rms_over_reference"] for row in scan_cut["rows"]
                ) / len(scan_cut["rows"]),
                "additive_projection_interval": fused_summary["fused_frozen_direction_interval_normal_95"],
                "source_cut_aligned_mean": fused_summary["cut_aligned_mean"],
                "source_cut_aligned_interval": fused_summary["cut_aligned_interval_normal_95"],
                "source": str(mamba_scan_cut_path.relative_to(ROOT)),
                "scope": "DESCRIPTIVE_MAMBA_REAL_TEXT_FIXED_SUITE_SELECTIVE_SCAN_CUT",
            }],
            "interpretation": (
                "The actual single-layer fused selective-scan boundary has a stable negative "
                "aligned write effect, and replacing only that scan with an explicit recurrence "
                "reproduces the signed aligned component. The source-cut effect vector is not "
                "identical to the fused vector, so this is a scoped aligned result rather than "
                "a complete-vector, vector-mean or population certificate."
            ),
        })

    # A later, predeclared with-replacement run provides an actual signed
    # projected-mean test for the Granite order variant. Keep the historical
    # fixed-suite evidence, but do not let it hide the newer population result.
    granite_population_path = ROOT / "results/property/granite_expert_order_population_v1/result.json"
    if granite_population_path.is_file():
        population = read(str(granite_population_path.relative_to(ROOT)))
        write = population["parameter_write"]
        gradient = population.get("gradient")

        def granite_stage_test(stage: str, values: dict[str, Any]) -> dict[str, Any]:
            """Normalize a stage record without changing its estimand."""
            reference_rms = values.get("reference_energy_rms")
            return {
                "source": str(granite_population_path.relative_to(ROOT)),
                "estimand": "MEAN_FIXED_CALIBRATION_DIRECTION_PROJECTION",
                "stage": stage,
                "independent_unit_count": population["confirmation_count"],
                "mean": values["confirmation_projection_mean"],
                "two_sided_95_interval": values["confirmation_projection_two_sided_95"],
                "one_sided_95_lower_bound": values["confirmation_projection_one_sided_95_lower"],
                "two_sided_95_nonzero_supported": bool(
                    values["confirmation_projection_two_sided_95"][0] > 0.0
                    or values["confirmation_projection_two_sided_95"][1] < 0.0
                ),
                "positive_mean_supported_one_sided": bool(
                    values["confirmation_projection_one_sided_95_lower"] > 0.0
                ),
                "positive_count": values["confirmation_positive_count"],
                "negative_count": values["confirmation_negative_count"],
                "zero_count": values["confirmation_zero_count"],
                "cross_state_u_statistic": values.get("confirmation_cross_state_u_statistic"),
                "scope": values["scope"],
                "implies_vector_mean_nonzero_if_assumptions_hold": bool(
                    values.get("population_vector_mean_claim", False)
                ),
                "confirmation_reference_rms": reference_rms,
                "normalized_mean_projection": (
                    values["confirmation_projection_mean"] / reference_rms
                    if reference_rms and reference_rms > 0 else None
                ),
                "normalized_one_sided_95_lower_bound": (
                    values["confirmation_projection_one_sided_95_lower"] / reference_rms
                    if reference_rms and reference_rms > 0 else None
                ),
                "assumptions": [
                    "with-replacement independent draws from the declared empirical bank",
                    "calibration direction fixed before confirmation",
                    "finite variance and Student-t approximation for the signed projection",
                    "zero-moment one-step AdamW response and declared carrier only",
                ],
            }

        stage_tests = {"parameter_write": granite_stage_test("PARAMETER_WRITE", write)}
        if gradient is not None:
            stage_tests["gradient"] = granite_stage_test("PARAMETER_GRADIENT", gradient)
    by_group["granite_moe_expert_contribution_order"].update({
        "mean_bias_status": "SUPPORTED_PROJECTED_MEAN_UNDER_DECLARED_IID_EMPIRICAL_BANK",
            "scoped_vector_mean_nonzero_supported": True,
            "population_test": {
                "source": str(granite_population_path.relative_to(ROOT)),
                "estimand": "MEAN_FIXED_CALIBRATION_DIRECTION_PROJECTION",
                "stage": "PARAMETER_WRITE",
                "independent_unit_count": population["confirmation_count"],
                "mean": write["confirmation_projection_mean"],
                "two_sided_95_interval": write["confirmation_projection_two_sided_95"],
                "one_sided_95_lower_bound": write["confirmation_projection_one_sided_95_lower"],
                "two_sided_95_nonzero_supported": bool(
                    write["confirmation_projection_two_sided_95"][0] > 0.0
                    or write["confirmation_projection_two_sided_95"][1] < 0.0
                ),
                "positive_mean_supported_one_sided": bool(
                    write["confirmation_projection_one_sided_95_lower"] > 0.0
                ),
                "positive_count": write["confirmation_positive_count"],
                "negative_count": write["confirmation_negative_count"],
                "cross_state_u_statistic": write["confirmation_cross_state_u_statistic"],
                "scope": write["scope"],
                "implies_vector_mean_nonzero_if_assumptions_hold": True,
                "confirmation_reference_write_rms": write.get("reference_energy_rms"),
                "normalized_mean_projection": (
                    write["confirmation_projection_mean"] / write["reference_energy_rms"]
                    if write.get("reference_energy_rms", 0.0) > 0 else None
                ),
                "normalized_one_sided_95_lower_bound": (
                    write["confirmation_projection_one_sided_95_lower"] / write["reference_energy_rms"]
                    if write.get("reference_energy_rms", 0.0) > 0 else None
                ),
                "assumptions": [
                    "with-replacement independent draws from the declared empirical bank",
                    "calibration direction fixed before confirmation",
                    "finite variance and Student-t approximation for the signed projection",
                    "zero-moment one-step AdamW response and declared carrier only",
                ],
            },
            "population_stage_tests": stage_tests,
            "interpretation": "Under the declared iid-with-replacement empirical bank and frozen calibration direction, the signed parameter-write projection mean has a positive one-sided bound; this implies a nonzero vector mean only for that scoped population.",
        })

    if gradient is not None:
        by_group["granite_moe_expert_contribution_order"]["interpretation"] = (
            "The fixed-precision expert-order change has a two-sided positive projected mean "
            "at the gradient stage under the declared empirical-bank protocol. The parameter-write "
            "stage is positive only under a one-sided bound; both results are conditional on the "
            "declared carrier, cold-start one-step AdamW response, and empirical bank."
        )

    # OLMoE router accumulation is a separate natural fixed-suite boundary:
    # it changes accumulation dtype, not expert order.  The probe has a
    # descriptive aligned-write interval but no iid population contract.
    olmoe_path = ROOT / "results/property/new_problem_group_search_v1/olmoe_router_accum_natural_16_20260919.json"
    if olmoe_path.is_file() and "olmoe_router_expert_accumulation" in by_group:
        olmoe = read(str(olmoe_path.relative_to(ROOT)))
        write_summary = olmoe["summaries"]["write"]
        by_group["olmoe_router_expert_accumulation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": {
                "source": str(olmoe_path.relative_to(ROOT)),
                "state_count": olmoe["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": write_summary["confirmation_aligned_mean"],
                "aligned_normal_95_interval": write_summary["confirmation_aligned_interval_normal_95"],
                "write_effect_rms_over_reference": write_summary["effect_rms_over_reference"],
                "routing_and_weights_unchanged": all(
                    row["routing_same"] and row["routing_weights_same"] for row in olmoe["rows"]
                ),
                "scope": "DESCRIPTIVE_NATURAL_OLMOE_FIXED_SUITE",
            },
            "interpretation": (
                "The OLMoE router-combine intervention has a negative aligned-write interval on the "
                "declared real-model suite, while its held-out calibration-direction projection crosses "
                "zero. This supports a scoped aligned update bias, not a population vector-mean claim."
            ),
        })

    # Granite router-score precision is a separate fixed-selection boundary:
    # unlike the ordinary full-route comparison, the native top-k indices are
    # held fixed, so the observed effect is score/softmax-weight materialization
    # rather than a changed expert set.  The real text windows are a declared
    # fixed checkpoint suite, not an iid population sample.
    granite_router_path = ROOT / "results/property/new_problem_group_search_v1/granite_router_projection_materialization_frozen_32_20260919.json"
    if granite_router_path.is_file() and "granite_moe_router_score_materialization" in by_group:
        granite_router = read(str(granite_router_path.relative_to(ROOT)))
        values = [
            float(value)
            for value in granite_router["summary"]["confirmation_aligned_write_values"]
        ]
        n = len(values)
        mean = math.fsum(values) / n
        variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1)
        sd = math.sqrt(variance)
        from scipy.stats import t
        half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        by_group["granite_moe_router_score_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_evidence": {
                "source": str(granite_router_path.relative_to(ROOT)),
                "state_count": granite_router["summary"]["state_count"],
                "calibration_count": granite_router["summary"]["calibration_count"],
                "confirmation_count": granite_router["summary"]["confirmation_count"],
                "stage": "PARAMETER_WRITE",
                "selected_set_changed_states": granite_router["summary"]["selected_set_changed_states"],
                "aligned_mean": mean,
                "aligned_normal_95_interval": [mean - half, mean + half],
                "positive_count": sum(value > 0.0 for value in values),
                "negative_count": sum(value < 0.0 for value in values),
                "write_effect_rms_relative_mean": granite_router["summary"]["mean_router_write_rms_relative"],
                "scope": "DESCRIPTIVE_NATURAL_GRANITE_FIXED_TOPK_SUITE",
            },
            "interpretation": (
                "With the native top-k indices frozen, the Granite router-score precision intervention "
                "has a negative aligned-write effect on the declared real text suite. This is a fixed-"
                "checkpoint descriptive aligned result, not an iid population vector-mean certificate."
            ),
        })
    # Complete original-coordinate Grams retained for the softmax and MM
    # source decompositions can also be analysed as finite empirical banks.
    # This is a scoped offline population result, not a natural-training claim.
    empirical_gram_path = ROOT / "results/property/root_cause_closure_v1/empirical_bank_projected_mean_v1.json"
    if empirical_gram_path.is_file():
        empirical = read(str(empirical_gram_path.relative_to(ROOT)))
        for group in ("softmax_saved_state_backward", "mm_gemm_output_and_accumulation"):
            tests = [row for row in empirical["results"] if row["problem_group"] == group]
            if not tests:
                continue
            by_group[group]["empirical_bank_population_tests"] = tests
            supported = [row for row in tests if row["status"] == "PROJECTED_MEAN_SUPPORTED"]
            if group == "softmax_saved_state_backward":
                primary = next((row for row in supported if row["component"] == "semantic_total"), None)
                if primary is not None:
                    by_group[group]["mean_bias_status"] = "SUPPORTED_PROJECTED_MEAN_UNDER_DECLARED_CASE_EMPIRICAL_BANK"
                    by_group[group]["scoped_vector_mean_nonzero_supported"] = True
                    by_group[group]["interpretation"] = (
                        "The semantic-total complete-coordinate Gram supports a positive signed projected "
                        "mean under iid draws from the declared finite case bank; the local saved-state "
                        "source remains closed, but this does not establish a natural-state population or loss consequence."
                    )
            else:
                totals = [row for row in supported if row["component"] == "total"]
                if totals:
                    by_group[group]["mean_bias_status"] = "SUPPORTED_PROJECTED_MEAN_UNDER_DECLARED_CASE_EMPIRICAL_BANK"
                    by_group[group]["scoped_vector_mean_nonzero_supported"] = True
                    by_group[group]["interpretation"] = (
                        "Each retained MM case has a positive signed projected mean for its complete "
                        "total-effect Gram under iid draws from its declared finite case bank; this "
                        "does not merge the case-specific arithmetic sources into a universal MM root."
                    )

    liger_population_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length64_population_mean_v1.json"
    if liger_population_path.is_file():
        population = read(str(liger_population_path.relative_to(ROOT)))
        confirmation = population["confirmation"]
        population_test = {
                "source": str(liger_population_path.relative_to(ROOT)),
                "estimand": "MEAN_FIXED_CALIBRATION_DIRECTION_PROJECTION",
                "stage": "PARAMETER_WRITE",
                "independent_unit_count": population["confirmation_count"],
                "mean": confirmation["mean_projection"],
                "two_sided_95_interval": confirmation["two_sided_95_interval"],
            "one_sided_95_lower_bound": confirmation["one_sided_95_lower_bound"],
            "two_sided_95_nonzero_supported": bool(
                confirmation["two_sided_95_interval"][0] > 0.0
                or confirmation["two_sided_95_interval"][1] < 0.0
            ),
            "positive_mean_supported_one_sided": bool(
                confirmation["one_sided_95_lower_bound"] > 0.0
            ),
                "positive_count": confirmation["positive_count"],
                "negative_count": confirmation["negative_count"],
                "scope": confirmation["scope"],
                "implies_vector_mean_nonzero_if_assumptions_hold": confirmation["implies_vector_mean_nonzero_if_one_sided_bound_positive"],
                "assumptions": [
                    "with-replacement independent draws from the declared empirical bank",
                    "calibration direction fixed before confirmation",
                    "finite variance and Student-t approximation for the signed projection",
                    "zero-moment one-step torch AdamW response and lm_head.weight carrier only",
                ],
        }
        confirmation_rows = population.get("confirmation_rows", [])
        if confirmation_rows:
            reference_rms = math.sqrt(
                math.fsum(float(row["repair_energy"]) for row in confirmation_rows)
                / len(confirmation_rows)
            )
            population_test["confirmation_reference_write_rms"] = reference_rms
            population_test["normalized_mean_projection"] = (
                confirmation["mean_projection"] / reference_rms if reference_rms > 0 else None
            )
            population_test["normalized_one_sided_95_lower_bound"] = (
                confirmation["one_sided_95_lower_bound"] / reference_rms if reference_rms > 0 else None
            )
        liger_row = by_group["liger_fused_linear_ce_dw_accumulation"]
        liger_row.update({
            "mean_bias_status": "SUPPORTED_PROJECTED_MEAN_UNDER_DECLARED_IID_EMPIRICAL_BANK",
            "scoped_vector_mean_nonzero_supported": True,
            "population_test": population_test,
            "interpretation": "Under the declared iid-with-replacement empirical bank and frozen original-coordinate calibration direction, the signed parameter-write projection mean has a positive one-sided bound; this implies a nonzero vector mean only for that scoped population.",
        })
        length256_path = ROOT / "results/property/liger_fp32_chunk_order_v1/length256_population_mean_v1.json"
        if length256_path.is_file():
            length256 = read(str(length256_path.relative_to(ROOT)))
            c256 = length256["confirmation"]
            liger_row["additional_population_tests"] = {
                "length256": {
                    "source": str(length256_path.relative_to(ROOT)),
                    "confirmation_count": length256["confirmation_count"],
                    "mean": c256["mean_projection"],
                    "two_sided_95_interval": c256["two_sided_95_interval"],
                    "one_sided_95_lower_bound": c256["one_sided_95_lower_bound"],
                    "positive_count": c256["positive_count"],
                    "negative_count": c256["negative_count"],
                    "scope": c256["scope"],
                    "implies_vector_mean_nonzero_if_assumptions_hold": c256["implies_vector_mean_nonzero_if_one_sided_bound_positive"],
                }
            }

    jsd_path = ROOT / "results/property/case_causal_audit_v1/liger_jsd_natural_training_boundary.json"
    if jsd_path.is_file() and "liger_fused_linear_jsd_distillation" in by_group:
        jsd = read(str(jsd_path.relative_to(ROOT)))
        by_group["liger_fused_linear_jsd_distillation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_TEXT_BOUNDARY",
            "scoped_vector_mean_nonzero_supported": False,
            "fixed_suite_aligned_evidence": {
                "source": str(jsd_path.relative_to(ROOT)),
                "state_count": jsd["state_count"],
                "calibration_count": jsd["calibration_count"],
                "confirmation_count": jsd["confirmation_count"],
                "write_effect_rms_mean": jsd["write_effect_rms_mean"],
                "write_effect_relative_rms_mean": jsd["write_effect_relative_rms_mean"],
                "aligned_confirmation_mean": jsd["aligned_confirmation_mean"],
                "aligned_confirmation_interval_95_approx": jsd["aligned_confirmation_interval_95_approx"],
                "aligned_confirmation_positive": jsd["aligned_confirmation_positive"],
                "aligned_confirmation_negative": jsd["aligned_confirmation_negative"],
                "scope": "FIXED_REAL_TEXT_DISTILLATION_BOUNDARY_ONLY",
            },
            "interpretation": (
                "The real-text student/teacher boundary has a descriptive positive aligned "
                "write interval, but this is not an iid population certificate and does not "
                "establish a vector mean or long-run quality effect."
            ),
        })

    addmm_path = ROOT / "results/property/case_causal_audit_v1/bert_fused_addmm_bias_materialization_boundary.json"
    if addmm_path.is_file() and "bert_fused_addmm_bias_materialization" in by_group:
        addmm = read(str(addmm_path.relative_to(ROOT)))
        summary = addmm["source_control"]
        by_group["bert_fused_addmm_bias_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CPU_BOUNDARY",
            "scoped_vector_mean_nonzero_supported": False,
            "fixed_suite_aligned_evidence": {
                "source": str(addmm_path.relative_to(ROOT)),
                "state_count": addmm["state_count"],
                "calibration_count": addmm["calibration_count"],
                "confirmation_count": addmm["confirmation_count"],
                "write_effect_rms_mean": addmm["write_effect_rms_mean"],
                "aligned_confirmation_mean": addmm["aligned_write_mean"],
                "aligned_confirmation_interval_95_approx": addmm["aligned_write_interval_normal_95"],
                "aligned_confirmation_positive": addmm["aligned_write_positive"],
                "aligned_confirmation_negative": addmm["aligned_write_negative"],
                "scope": "FIXED_REAL_CPU_TEXT_BOUNDARY_ONLY",
            },
            "interpretation": (
                "The BERT-tiny CPU fused-addmm boundary has a descriptive negative aligned write "
                "interval, with factorial controls isolating the fused-linear path; this is not an "
                "iid population certificate or a vector-mean claim."
            ),
        })

    softmax_path = ROOT / "results/property/case_causal_audit_v1/bert_attention_softmax_materialization_boundary.json"
    if softmax_path.is_file() and "bert_attention_softmax_materialization" in by_group:
        softmax = read(str(softmax_path.relative_to(ROOT)))
        summary = softmax["summary"]
        by_group["bert_attention_softmax_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CUDA_BOUNDARY",
            "scoped_vector_mean_nonzero_supported": False,
            "fixed_suite_aligned_evidence": {
                "source": str(softmax_path.relative_to(ROOT)),
                "state_count": softmax["state_count"],
                "calibration_count": softmax["calibration_count"],
                "confirmation_count": softmax["confirmation_count"],
                "write_effect_rms_mean": summary["write_effect_rms_mean"],
                "aligned_confirmation_mean": summary["aligned_write_mean"],
                "aligned_confirmation_interval_95_approx": summary["aligned_write_interval_normal_95"],
                "aligned_confirmation_positive": summary["aligned_write_positive"],
                "aligned_confirmation_negative": summary["aligned_write_negative"],
                "scope": "FIXED_REAL_CUDA_TEXT_BOUNDARY_ONLY",
            },
            "interpretation": (
                "The BERT-tiny CUDA attention-softmax boundary has a descriptive negative aligned "
                "write interval, but this is not an iid population certificate or a vector-mean claim."
            ),
        })

    score_path = ROOT / "results/property/case_causal_audit_v1/bert_attention_score_materialization_boundary.json"
    if score_path.is_file() and "bert_attention_score_materialization" in by_group:
        score = read(str(score_path.relative_to(ROOT)))
        summary = score["summary"]
        by_group["bert_attention_score_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CUDA_FP16_BOUNDARY",
            "scoped_vector_mean_nonzero_supported": False,
            "fixed_suite_aligned_evidence": {
                "source": str(score_path.relative_to(ROOT)),
                "state_count": score["state_count"],
                "calibration_count": score["calibration_count"],
                "confirmation_count": score["confirmation_count"],
                "write_effect_rms_mean": summary["write_effect_rms_mean"],
                "aligned_confirmation_mean": summary["aligned_write_mean_confirmation"],
                "aligned_confirmation_interval_95_approx": summary["aligned_write_interval_normal_95"],
                "aligned_confirmation_positive": summary["aligned_positive_count"],
                "aligned_confirmation_negative": summary["aligned_negative_count"],
                "scope": "FIXED_REAL_CUDA_FP16_TEXT_BOUNDARY_ONLY",
            },
            "interpretation": (
                "The BERT-tiny CUDA FP16 attention-score boundary has a descriptive negative "
                "aligned write interval, but this is not an iid population certificate or a "
                "vector-mean claim."
            ),
        })

    value_path = ROOT / "results/property/case_causal_audit_v1/bert_attention_value_materialization_boundary.json"
    if value_path.is_file() and "bert_attention_value_materialization" in by_group:
        value = read(str(value_path.relative_to(ROOT)))
        summary = value["summary"]
        by_group["bert_attention_value_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CUDA_FP16_BOUNDARY",
            "scoped_vector_mean_nonzero_supported": False,
            "fixed_suite_aligned_evidence": {
                "source": str(value_path.relative_to(ROOT)),
                "state_count": value["state_count"],
                "calibration_count": value["calibration_count"],
                "confirmation_count": value["confirmation_count"],
                "write_effect_rms_mean": summary["write_effect_rms_mean"],
                "aligned_confirmation_mean": summary["aligned_write_mean_confirmation"],
                "aligned_confirmation_interval_95_approx": summary["aligned_write_interval_normal_95"],
                "aligned_confirmation_positive": summary["aligned_positive_count"],
                "aligned_confirmation_negative": summary["aligned_negative_count"],
                "scope": "FIXED_REAL_CUDA_FP16_TEXT_BOUNDARY_ONLY",
            },
            "interpretation": (
                "The BERT-tiny CUDA FP16 attention-value boundary has a descriptive negative "
                "aligned write interval, but this is not an iid population certificate or a "
                "vector-mean claim."
            ),
        })

    # Exact original-coordinate GELU rows support an aligned descriptive
    # statistic, but no vector mean because the vectors were not retained.
    gelu_path = "results/property/numerical_coverage_v1/gemma_gelu_capture_v2/raw/gelu_backward_952_out_ptr0.json"
    gelu = read(gelu_path)
    gelu_entries = {}
    for stage, stage_rows in gelu["original_coordinate_statistics"].items():
        aligned = [
            float(row["effect_repair_inner_product"]) / float(row["repair_energy"])
            for row in stage_rows if float(row.get("repair_energy", 0.0)) > 0
        ]
        gelu_entries[stage] = {
            "state_count": len(aligned),
            "aligned_ratio_mean_fixed_suite": math.fsum(aligned) / len(aligned),
            "positive_count": sum(x > 0 for x in aligned),
            "negative_count": sum(x < 0 for x in aligned),
            "scope": "DESCRIPTIVE_FIXED_SUITE_ONLY",
        }
    by_group["gemma_gelu_backward_evaluation"].update({
        "mean_bias_status": "FIXED_SUITE_ALIGNED_DESCRIPTIVE_VECTOR_MEAN_UNAVAILABLE",
        "fixed_suite_aligned_evidence": gelu_entries,
        "interpretation": "Original-coordinate energy and aligned summaries exist, but the vector mean cannot be reconstructed from scalar rows.",
    })

    # GPT-Neo native GELU is a separate implementation boundary from the
    # unresolved generated-Triton Gemma GELU case.  The retained 32-state
    # record supports a fixed-suite aligned description only; its loss interval
    # crosses zero and no population mean is inferred.
    gptneo_gelu_path = ROOT / "results/property/root_cause_closure_v1/gptneo_gelu_natural_32_20260920.json"
    if gptneo_gelu_path.is_file() and "gptneo_gelu_native_fp32_materialization" in by_group:
        gptneo = read(str(gptneo_gelu_path.relative_to(ROOT)))
        summary = gptneo["summary"]
        by_group["gptneo_gelu_native_fp32_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "source": str(gptneo_gelu_path.relative_to(ROOT)),
                "model": gptneo["model"],
                "operator": gptneo["operator"],
                "layer": gptneo["layer"],
                "state_count": summary["state_count"],
                "stage": "PARAMETER_WRITE",
                "aligned_mean": summary["aligned_write_mean"],
                "aligned_normal_95_interval": summary["aligned_write_interval_normal_95"],
                "write_effect_rms_mean_over_reference": summary["write_effect_rms_mean"],
                "gradient_effect_rms_mean_over_reference": summary["gradient_effect_rms_mean"],
                "loss_interval": summary["loss_difference_interval_normal_95"],
                "source_native_formula_bf16_relative_l2_mean": summary[
                    "source_native_formula_bf16_relative_l2_mean"
                ],
                "source_native_vs_fp32_formula_relative_l2_mean": summary[
                    "source_native_vs_fp32_formula_relative_l2_mean"
                ],
                "scope": "DESCRIPTIVE_GPTNEO_REAL_TEXT_FIXED_SUITE",
            },
            "interpretation": (
                "The native GPT-Neo GELU evaluation has a stable negative aligned write effect "
                "relative to the same-input FP32 tanh-GELU reference on the declared text bank. "
                "The loss interval crosses zero, so this is not a population mean-bias or long-run "
                "quality certificate. It is a separate native implementation boundary from the "
                "unresolved generated-Triton Gemma GELU case."
            ),
        })

    # RWKV time-mix decay materialisation: the same recurrent boundary is
    # replayed with only the time_decay exponential promoted to FP32.  This is
    # a fixed real-text suite description; it is not an iid population test.
    rwkv_path = ROOT / "results/property/new_problem_group_search_v1/rwkv_time_mix_natural_16_20260920.json"
    if rwkv_path.is_file() and "rwkv_time_decay_materialization" in by_group:
        rwkv = read(str(rwkv_path.relative_to(ROOT)))
        decay = rwkv["summary"]["modes"]["decay_fp32"]
        value = rwkv["summary"]["modes"]["value_fp32"]
        by_group["rwkv_time_decay_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "source": str(rwkv_path.relative_to(ROOT)),
                "model": rwkv["model"],
                "operator_family": rwkv["operator_family"],
                "target_parameter": rwkv["target_parameter"],
                "state_count": rwkv["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "decay_fp32": decay,
                "value_fp32_control": value,
                "scope": "DESCRIPTIVE_RWKV_REAL_TEXT_FIXED_SUITE",
            },
            "interpretation": (
                "The RWKV time_decay-only FP32 intervention has a negative aligned-write "
                "interval on all 16 declared real-text states, while the value-only control "
                "interval crosses zero. This is a fixed-suite source-boundary result, not an "
                "iid population or long-run loss-quality certificate."
            ),
        })

    # RWKV receptance sigmoid: two independent real-text banks support the
    # same fixed-suite negative aligned write direction after changing only
    # sigmoid evaluation to FP32.  This remains a scoped descriptive result.
    rwkv_sigmoid_paths = [
        ROOT / "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_natural_32_20260920.json",
        ROOT / "results/property/new_problem_group_search_v1/rwkv_receptance_sigmoid_method_32_20260920.json",
    ]
    if all(path.is_file() for path in rwkv_sigmoid_paths) and "rwkv_receptance_sigmoid_materialization" in by_group:
        records = [read(str(path.relative_to(ROOT))) for path in rwkv_sigmoid_paths]
        by_group["rwkv_receptance_sigmoid_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "sources": [str(path.relative_to(ROOT)) for path in rwkv_sigmoid_paths],
                "model": records[0]["model"],
                "operator_family": records[0]["operator_family"],
                "target_parameter": records[0]["target_parameter"],
                "state_count_per_bank": [record["summary"]["state_count"] for record in records],
                "stage": "PARAMETER_WRITE",
                "banks": [record["summary"] for record in records],
                "scope": "DESCRIPTIVE_RWKV_REAL_TEXT_FIXED_SUITES",
            },
            "interpretation": (
                "Changing only RWKV receptance sigmoid evaluation to FP32 gives a strictly negative "
                "confirmation aligned-write interval on both declared real-text banks. This supports "
                "a scoped fixed-suite aligned source result, not a population vector mean or a long-run "
                "loss-quality certificate."
            ),
        })

    # Final embedding-gradient materialization is a distinct, same-cotangent
    # source intervention from repeated-token index-add accumulation.  Keep
    # its tiny but signed fixed-suite endpoint visible without upgrading it to
    # a population or practical-quality claim.
    embedding_cast_path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_gradient_cast_materialization_natural_16_20260920.json"
    if embedding_cast_path.is_file() and "deepseek_embedding_gradient_materialization" in by_group:
        embedding_cast = read(str(embedding_cast_path.relative_to(ROOT)))
        summary = embedding_cast["summary"]
        by_group["deepseek_embedding_gradient_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "source": str(embedding_cast_path.relative_to(ROOT)),
                "model": embedding_cast["model"],
                "operator": embedding_cast["operator"],
                "state_count": summary["state_count"],
                "stage": "PARAMETER_WRITE",
                "write_effect_rms_mean": summary["write_effect_rms_mean"],
                "aligned_mean": summary["aligned_write_mean"],
                "aligned_interval_normal_95": summary["aligned_write_interval_normal_95"],
                "projection_interval_normal_95": summary["projection_interval_normal_95"],
                "scope": "DESCRIPTIVE_DEEPSEEK_REAL_TEXT_FIXED_SUITE_MICRO_EFFECT",
            },
            "interpretation": (
                "The same captured embedding cotangent, with repeated-index accumulation held "
                "fixed, shows a tiny negative aligned write effect when the final gradient is "
                "materialized to BF16. The source is isolated, but the effect is about 3e-8 RMS "
                "and is not a practical quality result."
            ),
        })

    # A generated fused embedding/NLL/normalization boundary is retained as a
    # separate source-closed case when the BF16-partial variant reproduces a
    # signed write effect while the FP64 reduction control is near identity.
    fused_embedding_path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_v4_20260920.json"
    if fused_embedding_path.is_file() and "deepseek_fused_embedding_nll_partial_materialization" in by_group:
        fused_embedding = read(str(fused_embedding_path.relative_to(ROOT)))
        variants = fused_embedding["summary"]["variants"]
        by_group["deepseek_fused_embedding_nll_partial_materialization"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "source": str(fused_embedding_path.relative_to(ROOT)),
                "model": fused_embedding["model"],
                "operator": fused_embedding["operator"],
                "state_count": fused_embedding["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "bf16_partial": variants["bf16_partial"],
                "fp64_reduction_control": variants["fp64_reduction"],
                "scope": "DESCRIPTIVE_DEEPSEEK_GENERATED_FUSED_BOUNDARY_FIXED_SUITE",
            },
            "interpretation": (
                "The BF16-partial intervention has a strictly negative aligned embedding-write "
                "interval, while the FP64 reduction control is near identity and crosses zero. "
                "This is a fixed-suite source-boundary result, not a population or long-run quality claim."
            ),
        })

    # The same fused boundary also has a distinct same-precision reduction-order
    # intervention.  Keep it separate from BF16 partial materialization while
    # retaining its micro-scale fixed-suite scope.
    fused_order_path = ROOT / "results/property/new_problem_group_search_v1/deepseek_embedding_fused_boundary_probe_16_reverse_v5_20260920.json"
    if fused_order_path.is_file() and "deepseek_fused_embedding_nll_reduction_order" in by_group:
        fused_order = read(str(fused_order_path.relative_to(ROOT)))
        variants = fused_order["summary"]["variants"]
        by_group["deepseek_fused_embedding_nll_reduction_order"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "source": str(fused_order_path.relative_to(ROOT)),
                "model": fused_order["model"],
                "operator": fused_order["operator"],
                "state_count": fused_order["summary"]["state_count"],
                "stage": "PARAMETER_WRITE",
                "fp32_reverse_order": variants["fp32_reverse_order"],
                "fp64_reduction_control": variants["fp64_reduction"],
                "scope": "DESCRIPTIVE_DEEPSEEK_GENERATED_FUSED_BOUNDARY_FIXED_SUITE_MICRO_EFFECT",
            },
            "interpretation": (
                "Reversing only the FP32 reduction element order gives a tiny positive aligned "
                "write interval, while the FP64 control crosses zero. This is a source-closed "
                "fixed-suite result and not a practical quality or population claim."
            ),
        })

    # The larger BERT-tiny document-bank rerun supersedes the earlier 32-window
    # development screen for this boundary.  The held-out additive direction
    # remains inconclusive, but the fixed-suite aligned-write interval is
    # strictly positive, so retain it as scoped descriptive evidence rather
    # than upgrading it to a population mean claim.
    bert_nll_path = ROOT / "results/property/new_problem_group_search_v1/bert_tiny_nll_natural_method_128_20260920.json"
    if bert_nll_path.is_file() and "bert_nll_loss_evaluation" in by_group:
        bert_nll = read(str(bert_nll_path.relative_to(ROOT)))
        summary = bert_nll["summary"]
        by_group["bert_nll_loss_evaluation"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "source": str(bert_nll_path.relative_to(ROOT)),
                "model": bert_nll["model"],
                "operator": bert_nll["operator"],
                "state_count": summary["state_count"],
                "calibration_count": summary["calibration_count"],
                "confirmation_count": summary["confirmation_count"],
                "write_effect_rms_mean": summary["write_effect_rms_mean"],
                "aligned_write_mean": summary["aligned_write_mean"],
                "aligned_write_interval_normal_95": summary["aligned_write_interval_normal_95"],
                "heldout_projection_interval_normal_95": summary["projection_interval_normal_95"],
                "scope": "DESCRIPTIVE_FIXED_NATURAL_DOCUMENT_BANK_ONLY",
                "population_mean_bias_decision": "NOT_ASSESSED",
            },
            "interpretation": (
                "The 128-window BERT document-bank rerun has a strictly positive aligned "
                "decoder-write interval, while its held-out additive direction and loss "
                "intervals cross zero. This is scoped fixed-suite NLL boundary evidence, "
                "not an iid population or quality claim."
            ),
        })

    # These groups have trajectory/checkpoint projections but no independent
    # one-step vector sample suitable for a mean test.
    trajectory_only = {
        "silu_backward_evaluation": "Cumulative trajectory projections are positive at the retained checkpoints, but are not independent one-step units.",
        "attention_state_to_q_projection_region": "Checkpoint/state projection summaries change sign across checkpoints; no stable mean endpoint is declared.",
        "fused_rope_position_scaling": "No retained vector sample separates the arithmetic source from optimizer-state factors.",
    }
    for group, interpretation in trajectory_only.items():
        by_group[group].update({
            "mean_bias_status": "TRAJECTORY_OR_SOURCE_EVIDENCE_NOT_A_MEAN_TEST",
            "interpretation": interpretation,
        })
    # A natural Qwen3 standard-RoPE probe supplies a fixed-suite aligned
    # description for the broader RoPE family.  It is deliberately not a
    # population certificate and does not split the existing RoPE problem
    # group into a duplicate model-specific row.
    qwen3_rope_path = ROOT / "results/property/new_problem_group_search_v1/qwen3_rope_layer13_kproj_16_20260918.json"
    if qwen3_rope_path.is_file():
        qwen3_rope = read(str(qwen3_rope_path.relative_to(ROOT)))
        summary = qwen3_rope["summary"]
        by_group["fused_rope_position_scaling"].update({
            "mean_bias_status": "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE_REAL_CHECKPOINT",
            "fixed_suite_aligned_evidence": {
                "source": str(qwen3_rope_path.relative_to(ROOT)),
                "model": qwen3_rope["model"],
                "layer": qwen3_rope["layer"],
                "parameter": qwen3_rope["parameter"],
                "state_count": summary["state_count"],
                "write_effect_rms_mean": summary["write_effect_rms_mean"],
                "write_aligned_interval_normal_95": summary["write_aligned_interval_normal_95"],
                "heldout_write_projection_interval_normal_95": summary["heldout_write_projection_interval_normal_95"],
                "heldout_write_projection_positive": summary["heldout_write_projection_positive"],
                "heldout_write_projection_negative": summary["heldout_write_projection_negative"],
                "scope": "DESCRIPTIVE_FIXED_NATURAL_BANK_ONLY; SAME_ROPE_FAMILY_NOT_NEW_GROUP",
                "population_mean_bias_decision": "NOT_ASSESSED",
            },
            "interpretation": (
                "The Qwen3 native rotary helper versus explicit FP32 same-formula reference "
                "has a negative aligned write interval on the declared 16-state bank and a "
                "negative held-out projection in all eight confirmation states. This is fixed-"
                "suite family evidence, not a population or universal RoPE claim."
            ),
        })
    # SiLU has one post-discovery full-coordinate recapture.  Its finite-suite
    # mean magnitude is useful descriptive evidence, but the held-out
    # calibration direction does not reproduce on confirmation states.  Keep
    # both facts explicit: a nonzero observed-set mean is not an iid
    # population proof.
    silu_mean_path = ROOT / "results/property/numerical_coverage_v1/deepseek128_silu_full_mean_replay_retry1/full_coordinate_means.json"
    if silu_mean_path.is_file():
        silu_mean = read(str(silu_mean_path.relative_to(ROOT)))
        write_stage = silu_mean["stages"].get("PARAMETER_WRITE", {})
        by_group["silu_backward_evaluation"]["fixed_suite_mean_vector_descriptive"] = {
            "source": str(silu_mean_path.relative_to(ROOT)),
            "stage": "PARAMETER_WRITE",
            "confirmation_mean_relative_magnitude": write_stage.get("confirmation_mean_relative_magnitude"),
            "confirmation_residual_mean_relative_magnitude": write_stage.get("confirmation_residual_mean_relative_magnitude"),
            "normalized_heldout_effect_projections": write_stage.get("normalized_heldout_effect_projections"),
            "scope": "DESCRIPTIVE_FIXED_SUITE_ONLY",
            "population_mean_bias_decision": "NOT_ASSESSED",
        }
        by_group["silu_backward_evaluation"]["interpretation"] = (
            "A full-coordinate recapture shows a nonzero finite-suite mean magnitude, but its "
            "calibration direction does not reproduce on the confirmation states; no iid "
            "population mean-bias claim is made."
        )
    # A later natural-state probe samples the retained 4096-state bank with
    # replacement and measures the native/reference parameter-write aligned
    # projection directly. This is a scoped empirical-bank mean test, not an
    # unrestricted natural-training population claim.
    silu_probe_path = ROOT / "results/property/root_cause_closure_v1/silu_mean_probe_empirical_bank_128_20260921.json"
    if silu_probe_path.is_file():
        probe = mean_of_probe_aligned_values(
            str(silu_probe_path.relative_to(ROOT)), "torch_reference"
        )
        supported = bool(probe["two_sided_95_nonzero_supported"])
        by_group["silu_backward_evaluation"].update({
            "mean_bias_status": (
                "SUPPORTED_ALIGNED_MEAN_UNDER_DECLARED_IID_EMPIRICAL_BANK"
                if supported else "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE"
            ),
            "population_test": probe,
            "scoped_aligned_mean_supported": supported,
            "interpretation": (
                "A 128-draw with-replacement probe from the declared DeepSeek trajectory bank "
                "supports a negative native-minus-reference aligned parameter-write mean. This "
                "is a scoped empirical-bank result; it does not generalize to an unrestricted "
                "natural training population or establish a loss consequence."
            ),
        })
    # Gemma GELU uses the same declared-bank, with-replacement protocol as the
    # SiLU probe.  Keep this as a scoped empirical-bank estimand; it does not
    # upgrade the existing source closure into an unrestricted training-population
    # claim.
    gelu_probe_path = ROOT / "results/property/root_cause_closure_v1/gelu_mean_probe_empirical_bank_128_20260922.json"
    if gelu_probe_path.is_file():
        probe = mean_of_probe_aligned_values(
            str(gelu_probe_path.relative_to(ROOT)),
            "torch_reference",
            scope="DECLARED_IID_WITH_REPLACEMENT_FROM_GEMMA4_TRAJECTORY_BANK",
        )
        supported = bool(probe["two_sided_95_nonzero_supported"])
        by_group["gemma_gelu_backward_evaluation"].update({
            "mean_bias_status": (
                "SUPPORTED_ALIGNED_MEAN_UNDER_DECLARED_IID_EMPIRICAL_BANK"
                if supported else "FIXED_SUITE_ALIGNED_MEAN_DESCRIPTIVE"
            ),
            "population_test": probe,
            "scoped_aligned_mean_supported": supported,
            "interpretation": (
                "A 128-draw with-replacement probe from the declared Gemma-4 trajectory bank "
                "supports a negative native-minus-reference aligned parameter-write mean. This "
                "is a scoped empirical-bank result; it does not generalize to an unrestricted "
                "natural training population or establish a loss consequence."
            ),
        })
    by_group["gemma_rms_feature_reduction_order"].update({
        "mean_bias_status": "NEGATIVE_CONTROL_NO_CONFIRMED_FIXED_SUITE_MEAN_DIRECTION",
        "interpretation": "The tested endpoint order is a completed negative control; it does not prove all RMS implementations are unbiased.",
    })
    by_group["granite_router_topk_selection"].update({
        "mean_bias_status": "IDENTITY_ON_OBSERVED_SUITE_NO_MEAN_EFFECT",
        "interpretation": "All retained endpoint differences are exactly zero under the declared tie-order comparison.",
    })

    # Ensure every active group has an explicit status; this is the audit's
    # central invariant and prevents future cases from silently disappearing.
    missing = [row["problem_group"] for row in rows if "mean_bias_status" not in row]
    if missing:
        raise RuntimeError(f"missing mean-bias status for {missing}")
    benchmark_means = benchmark_fixed_suite_mean_evidence()
    benchmark_population = benchmark_empirical_bank_mean_tests(benchmark_means)
    # A broader retrospective audit covers every retained complete
    # original-coordinate endpoint in the large coverage shards.  It is kept
    # separate from the scientific problem groups: endpoint-level evidence
    # is useful for coverage, but does not by itself establish a root cause or
    # a natural training-population claim.
    endpoint_audit_path = ROOT / "results/property/root_cause_closure_v1/full_coordinate_endpoint_mean_bias_v1.json"
    endpoint_audit = read(str(endpoint_audit_path.relative_to(ROOT))) if endpoint_audit_path.is_file() else None
    return {
        "schema": "kernel-analyzer-all-case-mean-bias-audit-v1",
        "status": "ALL_ACTIVE_GROUPS_ATTEMPTED_WITH_SCOPED_MEAN_EVIDENCE",
        "new_gpu_measurements": False,
        "estimand_warning": "Only AdamW8bit, Liger, Granite, softmax, MM and the newly sampled SiLU row have declared iid projected-mean tests; all other rows are fixed-suite, trajectory, source, or identity evidence. The broader complete-coordinate endpoint audit is retrospective coverage-bank evidence and has no family-wise error control.",
        "rows": rows,
        "benchmark_fixed_suite_mean_evidence": benchmark_means,
        "benchmark_empirical_bank_mean_tests": benchmark_population,
        "complete_coordinate_endpoint_audit": endpoint_audit,
        "benchmark_empirical_bank_multiplicity": {
            "scope": "RETROSPECTIVE_ROW_LEVEL_ONLY",
            "familywise_error_controlled": False,
            "note": "The per-row directional intervals are useful conditional diagnostics; the reported count of supported rows is not a familywise discovery rate because these benchmark rows were selected from retained artifacts after the original measurements.",
        },
        "summary": {
            # ``rows`` includes the retained negative controls.  Keep the
            # audited total and the scientific active count separate so the
            # root-cause ledger's 25 active groups are not confused with its
            # Total rows include retained negative controls as well as active groups.
            "audited_problem_groups": len(rows),
            "active_problem_groups": sum(
                row.get("case_role") == "ACTIVE_BIAS_CASE" for row in rows
            ),
            "population_mean_supported": sum(row["mean_bias_status"].startswith("SUPPORTED_") for row in rows),
            "scoped_vector_mean_nonzero_supported": sum(
                bool(row.get("scoped_vector_mean_nonzero_supported", False)) for row in rows
            ),
            "scoped_aligned_mean_supported": sum(
                row["mean_bias_status"].startswith("SUPPORTED_ALIGNED_MEAN_")
                for row in rows
            ),
            # The following fields are deliberately disjoint.  Earlier
            # versions counted substring matches, so a negative control could
            # be counted both as fixed-suite evidence and as a non-test.
            "population_mean_not_established": sum(
                not row["mean_bias_status"].startswith("SUPPORTED_") for row in rows
            ),
            "remaining_trajectory_or_source_only": sum(
                row["mean_bias_status"].startswith("TRAJECTORY_") for row in rows
            ),
            "remaining_fixed_suite_or_identity": sum(
                row["mean_bias_status"].startswith(("FIXED_SUITE_", "NEGATIVE_", "IDENTITY_"))
                for row in rows
            ),
            # Compatibility fields retained for readers of older reports;
            # unlike the fields above, these are not intended to partition
            # the rows.
            "fixed_suite_descriptive": sum(row["mean_bias_status"].startswith("FIXED_SUITE_") for row in rows),
            "not_mean_test": sum(not row["mean_bias_status"].startswith("SUPPORTED_") for row in rows),
            "negative_or_identity": sum(row["mean_bias_status"].startswith(("NEGATIVE", "IDENTITY")) for row in rows),
            "benchmark_exact_mean_vector_rows": len(benchmark_means),
            "benchmark_exact_mean_vector_nonzero_rows": sum(
                item["mean_vector_nonzero_on_observed_suite"] for item in benchmark_means
            ),
            "benchmark_empirical_bank_projected_mean_supported": sum(
                item["status"] == "PROJECTED_MEAN_SUPPORTED" for item in benchmark_population
            ),
            "complete_coordinate_endpoint_rows": (
                endpoint_audit.get("summary", {}).get("complete_coordinate_endpoint_rows", 0)
                if endpoint_audit else 0
            ),
            "complete_coordinate_endpoint_projected_mean_supported": (
                endpoint_audit.get("summary", {}).get("projected_mean_supported", 0)
                if endpoint_audit else 0
            ),
        },
        "population_source_units": population_units,
    }


def main() -> None:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"json": str(OUT), "rows": len(report["rows"]), "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
