#!/usr/bin/env python3
"""Build a deduplicated family-depth report from checked machine artifacts.

The report deliberately separates catalogue coverage, update evidence,
mathematical-source evidence, modification validation, and training outcome.
Repeated models, shapes, layers, or checkpoints do not create new groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SOURCES = {
    "adamw_measurement": "results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_v2/analysis.json",
    "adamw_mechanism": "results/property/numerical_coverage_v1/torchao_adamw8bit_block_mechanism_v1/summary.json",
    "adamw_training": "results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/verification.json",
    "adamw_modified_training": "results/property/numerical_coverage_v1/mamba_adamw8bit_hybrid_training_confirmation_v1/verification_v2.json",
    "adamw_residual": "results/property/numerical_coverage_v1/adamw8bit_block_residual_development_v1.json",
    "adamw_compensation_probe": "results/property/numerical_coverage_v1/adamw8bit_error_compensation_probe_v1/verification.json",
    "adamw_compensation_training": "results/property/numerical_coverage_v1/adamw8bit_error_compensation_training_v1/verification.json",
    "rotary": "results/property/numerical_coverage_v1/ministral_fused_rotary_optimizer_condition_summary_v1.json",
    "rotary_family": "results/property/numerical_coverage_v1/ministral_fused_rotary_family_evidence_v1.json",
    "attention": "results/property/numerical_coverage_v1/qwen_flash_sdpa_attention_v1/analysis.json",
    "liger_identity": "results/property/training_numerical_analysis_v2/language_accumulation_identity_retry/result.json",
    "liger_training_1024": "results/property/training_numerical_analysis_v2/language_training_confirmation_iid/summary.json",
    "liger_training_10000": "results/property/single_point_collapse_v2/full_10000_summary.json",
    "liger_order": "results/property/liger_fp32_chunk_order_v1/summary.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(key: str) -> dict:
    return json.loads((ROOT / SOURCES[key]).read_text())


def build(family_report: dict) -> dict:
    if family_report.get("schema") != "operator-family-report-v2":
        raise ValueError("Unsupported family report")
    catalogue = []
    for row in family_report["families"]:
        stages = row["support_stage_counts"]
        kinds = row.get("valid_measurement_implementation_kind_counts", {})
        valid = stages.get("VALID_MEASUREMENT_COMPLETED", 0)
        triton = sum(count for kind, count in kinds.items() if "TRITON" in kind)
        catalogue.append({
            "family_id": row["family_id"],
            "label": row["label"],
            "classified_positions": row["classified_positions"],
            "valid_measurement_positions": valid,
            "valid_triton_positions": triton,
            "valid_other_or_undeclared_positions": valid - triton,
            "additional_measurement_artifacts": len(row.get("additional_measurement_evidence", [])),
            "confirmed_problem_count": None,
        })

    adamw_measurement = _read("adamw_measurement")
    adamw_mechanism = _read("adamw_mechanism")
    adamw_training = _read("adamw_training")
    adamw_modified = _read("adamw_modified_training")
    adamw_residual = _read("adamw_residual")
    adamw_compensation_probe = _read("adamw_compensation_probe")
    adamw_compensation_training = _read("adamw_compensation_training")
    rotary = _read("rotary")
    rotary_family = _read("rotary_family")["records"][0]
    attention = _read("attention")
    liger_identity = _read("liger_identity")
    liger_training = _read("liger_training_1024")
    liger_10000 = _read("liger_training_10000")
    liger_order = _read("liger_order")

    priorities = [
        {
            "problem_group": "ADAMW8BIT_BLOCKWISE_MOMENT_QUANTIZATION",
            "reporting_family": "OPTIMIZER_UPDATE",
            "actual_implementation": "TORCH_COMPILE_GENERATED_TRITON",
            "fixed_suite_update_evidence": {
                "decision": adamw_measurement["equivalence_decision"],
                "parameter_write_total_rms": adamw_measurement["bias_analysis"]["fixed_suite_total_rms"],
            },
            "mathematical_source": {
                "status": adamw_mechanism["prediction_result"],
                "statement": adamw_mechanism["frozen_prediction"]["primary"],
            },
            "modification_validation": {
                "block64_training": adamw_training["modified_variant"]["decision"],
                "fp32_first_moment_training": adamw_modified["primary"]["decision"],
                "simple_block_residual_corrections": "REJECTED_IN_DEVELOPMENT",
                "default_write_rms_in_cpu_replay": adamw_residual["aggregate"]["parameter_write_relative_rms_default"],
                "recurrence_compensation_write_prediction":
                    adamw_compensation_probe["recomputed"]["prediction_result"],
                "recurrence_compensation_default_write_rms":
                    adamw_compensation_probe["recomputed"]["mean_write_rms"]["default_block256"],
                "recurrence_compensation_modified_write_rms":
                    adamw_compensation_probe["recomputed"]["mean_write_rms"]["compensated_block256"],
                "recurrence_compensation_training":
                    adamw_compensation_training["recomputed"]["decision"],
            },
            "training_outcome": {
                "decision": adamw_training["primary"]["decision"],
                "candidate_minus_reference_loss_mean": adamw_training["primary"]["mean"],
                "interval_95": adamw_training["primary"]["interval_95"],
                "collapse": adamw_training["collapse_decision"],
                "default_minus_recurrence_compensated_loss_mean":
                    adamw_compensation_training["recomputed"]["mean"],
                "default_minus_recurrence_compensated_interval_95":
                    adamw_compensation_training["recomputed"]["interval_95"],
            },
            "chain_status": (
                "RECURRENCE_SOURCE_UPDATE_AND_TRAINING_IMPROVEMENT_CONFIRMED; "
                "SINGLE_DECLARED_PROTOCOL_ONLY"
            ),
        },
        {
            "problem_group": "FUSED_ROTARY_POSITION_TRANSFORM",
            "reporting_family": "ROTARY",
            "actual_implementation": rotary_family["backend"],
            "fixed_suite_update_evidence": {
                "high_position_total_rms": rotary_family["high_position_result"]["parameter_write_total_rms"],
                "low_position_total_rms": rotary_family["low_position_result"]["parameter_write_total_rms"],
            },
            "mathematical_source": {
                "status": "NOT_UNIQUELY_IDENTIFIED",
                "optimizer_state_observation": rotary["prediction_status"],
                "warm_to_reset_write_rms_ratio": rotary["comparisons"]["reset_to_warm_parameter_write_rms_ratio"],
            },
            "modification_validation": "NOT_RUN",
            "training_outcome": "NOT_MEASURED",
            "chain_status": "STRONG_FIXED_SUITE_PHENOMENON; SOURCE_AND_TRAINING_CHAIN_OPEN",
        },
        {
            "problem_group": "FUSED_LINEAR_CE_WEIGHT_GRADIENT_ACCUMULATION",
            "reporting_family": "CROSS_ENTROPY",
            "actual_implementation": "MIXED_LIGER_TRAINING_COMPUTATION",
            "fixed_suite_update_evidence": {
                "same_chunk_products": liger_identity["checks"]["same_chunk_products"],
                "telescoping_residual_exact": liger_identity["checks"]["telescoping_residual_exact"],
                "same_fp32_different_order_direction_repeated": liger_order["source_prediction"]["direction_repeated"],
            },
            "mathematical_source": {
                "status": "ROUNDING_ACCUMULATION_IDENTITY_VERIFIED; NONZERO_EXPECTATION_NOT_PROVED",
            },
            "modification_validation": "ACCUMULATION_AND_ORDER_VARIANTS_MEASURED; NO_GENERAL_TRAINING_FIX",
            "training_outcome": {
                "step_1024_mean_loss_gap": liger_training["mean_loss_gap_descriptive_only"],
                "step_1024_positive_pairs": liger_training["positive_pairs"],
                "step_1024_pair_count": liger_training["total_pairs"],
                "step_10000_collapse": (
                    "COLLAPSE_OBSERVED" if liger_10000["collapse"] else "NO_COLLAPSE_OBSERVED"
                ),
            },
            "chain_status": "SOURCE_IDENTITY_AND_TRAJECTORY_SPLIT; PERSISTENT_DEGRADATION_NOT_SUPPORTED",
        },
        {
            "problem_group": "FUSED_CAUSAL_ATTENTION_BACKEND_SUBSTITUTION",
            "reporting_family": "FUSED_ATTENTION",
            "actual_implementation": "PYTORCH_FLASH_CUDA_VS_MATH",
            "fixed_suite_update_evidence": {
                "decision": attention["equivalence_decision"],
                "parameter_write_total_rms": attention["bias_analysis"]["fixed_suite_total_rms"],
            },
            "mathematical_source": "NOT_IDENTIFIED",
            "modification_validation": "NOT_RUN",
            "training_outcome": "NOT_MEASURED",
            "chain_status": "STRONG_FIXED_SUITE_COMPARATOR; NOT_A_TRITON_MECHANISM_CHAIN",
        },
    ]
    return {
        "schema": "operator-problem-group-depth-report-v2",
        "scope": (
            "Deduplicated reporting groups. Catalogue coverage, update evidence, "
            "mathematical source, modification validation, and training outcome are separate."
        ),
        "catalogue_family_count": len(catalogue),
        "valid_position_count": family_report["support_stage_counts"].get(
            "VALID_MEASUREMENT_COMPLETED", 0
        ),
        "valid_position_count_is_not_problem_count": True,
        "catalogue": catalogue,
        "priority_problem_groups": priorities,
        "source_sha256": {
            "family_report": family_report.get("input_sha256", {}),
            **{key: _sha(ROOT / path) for key, path in SOURCES.items()},
            "builder": _sha(Path(__file__)),
        },
    }


def _markdown(report: dict) -> str:
    lines = [
        "# 算子族与重点问题组证据深度", "",
        f"当前清单有 {report['valid_position_count']} 个已核验位置、"
        f"{report['catalogue_family_count']} 个目录家族。位置数不是独立问题数。", "",
        "## 目录家族", "",
        "| 家族 | 已归类位置 | 已核验 | 其中明确 Triton | 其他或未声明 | 清单外专项证据 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["catalogue"]:
        lines.append(
            f"| {row['label']} | {row['classified_positions']} | "
            f"{row['valid_measurement_positions']} | {row['valid_triton_positions']} | "
            f"{row['valid_other_or_undeclared_positions']} | "
            f"{row['additional_measurement_artifacts']} |"
        )
    lines += ["", "## 当前应深入的去重问题组", "",
              "| 问题组 | 实际实现 | 当前闭合程度 |", "|---|---|---|"]
    for row in report["priority_problem_groups"]:
        lines.append(
            f"| {row['problem_group']} | {row['actual_implementation']} | {row['chain_status']} |"
        )
    lines += ["", (
        "该表不从已核验位置推断 bias，也不把 update 差异、数学来源、修改成功和训练后果"
        "合并成一个 PASS。模型、层号、shape、checkpoint 和训练阶段变化不会自动增加问题组。"
    ), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-report", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.json, args.markdown):
        if output.exists() or not output.resolve().is_relative_to(Path("/data1/tzh")):
            parser.error("Choose new outputs under /data1/tzh")
    report = build(json.loads(args.family_report.read_text()))
    report["source_sha256"]["family_report_artifact"] = _sha(args.family_report)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.markdown.write_text(_markdown(report))
    print(json.dumps({
        "families": report["catalogue_family_count"],
        "valid_positions": report["valid_position_count"],
        "priority_problem_groups": len(report["priority_problem_groups"]),
    }))


if __name__ == "__main__":
    main()
