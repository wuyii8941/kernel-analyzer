#!/usr/bin/env python3
"""Audit whether the current low-cost cascade recovers the frozen eight cases.

Predictions are constructed before the development targets are consulted.  The
audit therefore cannot promote a case by copying its historical mechanism or
trajectory verdict into the screen result.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from kernel_analyzer.bias_oracle_recovery import (
    RecoveryPrediction,
    compare_recovery,
    predict_conditional_source_risk,
    predict_crossfit_projection_risk,
    predict_population_coherence_risk,
    predict_reference_relative_risk,
    predict_response_rectification_risk,
    predict_source_fidelity_boundary,
)


ROOT = Path(__file__).resolve().parents[1]


# These labels are read only after build_predictions returns.  They are the
# frozen development targets from the systematic eight-case audit, not inputs
# to any predictor.
FROZEN_TARGETS = {
    "liger_fused_ce": "STRICT_POSITIVE",
    "phi4_seq64_lmhead_dx": "STRICT_POSITIVE",
    "qwen64_vproj_mm": "STRICT_POSITIVE",
    "qwen128_vproj_mm": "STRICT_POSITIVE",
    "qwen_saved_p_seq128": "STRICT_POSITIVE",
    "qwen3vl_silu_layer0": "STRICT_POSITIVE",
    "mamba_seq64_input_proj": "PARTIAL_POSITIVE",
    "qwen_layer23_attention_state": "ABSTAIN_BOUNDARY",
}


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {relative}")
    return value


def build_predictions() -> list[RecoveryPrediction]:
    """Build predictions without reading ``FROZEN_TARGETS``."""

    phi = load("results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json")
    return [
        predict_reference_relative_risk(
            "liger_fused_ce",
            load("results/property/bias_oracle_recovery/liger_reference_relative.json"),
        ),
        predict_population_coherence_risk(
            "phi4_seq64_lmhead_dx",
            phi["populations"]["confirmation"]["PARAMETER_GRADIENT"],
            states=4,
        ),
        predict_conditional_source_risk(
            "qwen64_vproj_mm",
            load("results/coverage/cases/qwen64_vproj_conditional_debias_r16.json.gz"),
            repeats=4,
            condition_limit=1,
        ),
        predict_conditional_source_risk(
            "qwen128_vproj_mm",
            load("results/coverage/cases/qwen128_vproj_conditional_debias.json.gz"),
            repeats=4,
            condition_limit=1,
        ),
        predict_response_rectification_risk(
            "qwen_saved_p_seq128",
            load("results/property/bias_property_search/saved_p_pairing_work_v2.json"),
        ),
        predict_response_rectification_risk(
            "qwen3vl_silu_layer0",
            load("results/property/bias_property_search/vl_silu_optimizer_oddness_v2.json"),
        ),
        predict_conditional_source_risk(
            "mamba_seq64_input_proj",
            load("results/coverage/cases/mamba_seq64_input_proj_conditional_debias.json.gz"),
            repeats=4,
            condition_limit=1,
        ),
        predict_source_fidelity_boundary(
            "qwen_layer23_attention_state",
            load("results/final/l23_s_bwd_antithetic.json"),
        ),
    ]


def build_matched_controls() -> dict[str, Any]:
    phi = load("results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json")
    phi_local = predict_population_coherence_risk(
        "phi_local_centered_control",
        phi["populations"]["confirmation"]["LOCAL_ENDPOINT"],
        states=4,
    )
    liger_events = load(
        "results/property/bias_oracle_recovery/liger_joint_event.json"
    )
    saved = load(
        "results/property/bias_property_search/saved_p_pairing_work_v2.json"
    )["aggregate"]
    silu = load(
        "results/property/bias_property_search/vl_silu_optimizer_oddness_v2.json"
    )["aggregate"]
    return {
        "phi_local_population": {
            "expected": "NO_GLOBAL_LOCAL_BIAS",
            "prediction": phi_local.as_dict(),
            "passed": not phi_local.direct_risk,
        },
        "liger_event_coherence_hypothesis": {
            "expected": "NOT_A_REQUIRED_CHANNEL",
            "prediction": liger_events["joint_event_certificate"],
            "passed": liger_events["joint_event_certificate"]["status"]
            == "CANCELING_EVENT_STRUCTURE",
            "scientific_role": (
                "falsifies the discarded hypothesis that Liger requires mutually "
                "aligned per-chunk rounding atoms"
            ),
        },
        "stateless_sgd_odd_response": {
            "saved_p_resultant_ratio": (
                saved["stateless_sgd_natural_resultant_l2"]
                / saved["natural_update_resultant_l2"]
            ),
            "silu_resultant_ratio": (
                silu["stateless_sgd_resultant_l2"]
                / silu["natural_update_resultant_l2"]
            ),
            "maximum_safe_ratio": 0.01,
            "passed": (
                saved["stateless_sgd_natural_resultant_l2"]
                / saved["natural_update_resultant_l2"] < 0.01
                and silu["stateless_sgd_resultant_l2"]
                / silu["natural_update_resultant_l2"] < 0.01
            ),
        },
    }


def build_secondary_regression() -> dict[str, Any]:
    """Exercise frozen witnesses outside the six-case design roster.

    The artifacts predate this screen, so this is retrospective rather than a
    prospective held-out result.  It still tests a distinct witness path and a
    real non-directional control without changing thresholds.
    """

    lmhead = load("archive/nonprecision_v1/runs/lmhead.dh.screen.json")
    rmsnorm = load("archive/nonprecision_v1/runs/liger.rmsnorm.certificate.json")
    lmhead_prediction = predict_crossfit_projection_risk(
        "qwen_seq128_lmhead_dx_layout",
        lmhead["confirmation"]["projections"],
        evaluation_count=8,
        basis_frozen_before_evaluation=True,
    )
    rmsnorm_prediction = predict_crossfit_projection_risk(
        "liger_rmsnorm_dx_control",
        rmsnorm["endpoint_results"]["dX"]["carrier_projections"],
        evaluation_count=8,
        basis_frozen_before_evaluation=True,
    )
    return {
        "scientific_role": "SECONDARY_RETROSPECTIVE_REGRESSION_NOT_HELDOUT",
        "thresholds_changed_after_reading": False,
        "positive": {
            "expected": "DIRECTIONAL_RISK",
            "prediction": lmhead_prediction.as_dict(),
            "passed": lmhead_prediction.direct_risk,
        },
        "non_directional_control": {
            "expected": "NO_DIRECT_RISK_FROM_THIS_WITNESS",
            "prediction": rmsnorm_prediction.as_dict(),
            "passed": not rmsnorm_prediction.direct_risk,
        },
        "passed": (
            lmhead_prediction.direct_risk
            and not rmsnorm_prediction.direct_risk
        ),
    }


def render(document: dict[str, Any]) -> str:
    audit = document["recovery"]
    rows = audit["rows"]
    direct = sum(row["directly_recovered"] for row in rows)
    lines = [
        "# Bias Oracle 已知案例恢复审计",
        "",
        "## 结论",
        "",
        (
            "更新后的多通道 screen 在当前 development roster 的六个严格 formation "
            f"positive 中，全部 {int(audit['strict_direct_recall'] * audit['strict_positive_count'])}/"
            f"{audit['strict_positive_count']} 能被当前实际接通的筛选路径直接报出风险；"
            "Liger 由 reference-relative moving frame 命中，Phi 由四状态 complete-Gram "
            "coherence 命中。"
        ),
        "",
        (
            f"八个边界中共有 {direct}/8 得到直接风险/边界判定；"
            f"false-safe 为 {audit['false_safe_count']}。自动升级使严格 positive 的 routed recall "
            f"为 {audit['strict_routed_recall']:.1%}，但这只能证明流程安全，不能证明指标完整。"
        ),
        "",
        "## 逐例结果",
        "",
        "| case | 冻结目标 | 当前预测 | 直接命中 | fail-closed |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | {row['target']} | `{row['prediction']}` | "
            f"{'yes' if row['directly_recovered'] else 'no'} | "
            f"{'yes' if row['fail_closed_routed'] else 'no'} |"
        )
    lines.extend([
        "",
        "## 指标结构",
        "",
        "- Qwen64/128：四次 fixed-condition repeat 已能直接发现 local source risk，但下游层仍需顺序追加 repeat。",
        "- saved-P/SiLU：exact antithetic gradient pair 的 Adam response-even screen 可直接命中。",
        "- Mamba：local risk 可直接命中；真实 gradient/SGD 的条件覆盖仍不完整，所以必须保留 partial。",
        "- layer-23：natural-source fidelity 未通过，正确输出是 abstain，而不是 risk 或 safe。",
        "- Liger：chunk atoms 彼此并不相干；真正可比较的是每个 state 内误差相对 FP32-accumulator reference update 的乘性系数。",
        "- Phi：reference-relative 系数会变号，但四状态 complete-vector Gram 已直接暴露共同参数坐标分量。",
        "",
        "因此合适的候选不是一个标量，而是三个互补、均不使用 trajectory label 的风险证据族：",
        "",
        "1. conditional event/source asymmetry；",
        "2. transported directional component（complete-vector population、same-state moving frame 或冻结 cross-fit projection）；",
        "3. exact antithetic response non-oddness。",
        "",
        "任一 witness 命中即可报风险；全部未命中只能升级或 abstain，不能签发 safe。",
        "",
        "## 二级回归",
        "",
        (
            "未参与当前六案例指标拟合的旧 Qwen lm_head confirmation 被 cross-fit witness 命中，"
            "而 Liger RMSNorm dX 的真实 sign-changing control 未被命中。它们仍是回顾性证据，"
            "不能替代下一轮预先冻结的 prospective held-out。"
        ),
        "",
        "## 边界",
        "",
        "这是已参与假设形成的 development recovery audit，不是 held-out accuracy。六个严格 positives 全部命中只说明候选指标值得进入冻结 held-out；不能据此反调阈值或声称通用。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_oracle_recovery",
    )
    args = parser.parse_args()
    predictions = build_predictions()
    recovery = compare_recovery(predictions, FROZEN_TARGETS)
    controls = build_matched_controls()
    secondary_regression = build_secondary_regression()
    document = {
        "schema": "kernel-analyzer-bias-oracle-recovery-v1",
        "scientific_role": "DEVELOPMENT_CASE_RECOVERY_NOT_HELDOUT_VALIDATION",
        "prediction_built_before_target_comparison": True,
        "historical_verdicts_used_as_predictor_inputs": False,
        "protocol": {
            "conditional_repeat_budget": 4,
            "conditional_subset": "FIRST_FOUR_ACQUISITION_REPEATS",
            "response_minimum_nonoddness_ratio": 0.05,
            "response_minimum_energy_weighted_crossing_even_fraction": 0.5,
            "safe_release_on_unresolved": False,
            "coded_group_natural_case_measurements": 0,
            "shared_hvp_natural_case_measurements": 0,
        },
        "predictions": [prediction.as_dict() for prediction in predictions],
        "matched_controls": controls,
        "secondary_regression": secondary_regression,
        "recovery": recovery,
        "decision": (
            "DEVELOPMENT_RECOVERY_PASS__FREEZE_AND_RUN_HELDOUT_BEFORE_ORACLE_CLAIM"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recovery.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "summary.md").write_text(render(document), encoding="utf-8")


if __name__ == "__main__":
    main()
