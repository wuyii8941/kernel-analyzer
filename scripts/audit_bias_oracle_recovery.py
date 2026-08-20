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
    missing_screen_prediction,
    predict_conditional_source_risk,
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

    return [
        missing_screen_prediction("liger_fused_ce", missing="EVENT_MOMENT"),
        missing_screen_prediction(
            "phi4_seq64_lmhead_dx", missing="TRANSPORT_JOINT_MOMENT"
        ),
        predict_conditional_source_risk(
            "qwen64_vproj_mm",
            load("results/coverage/cases/qwen64_vproj_conditional_debias_r16.json.gz"),
            repeats=4,
        ),
        predict_conditional_source_risk(
            "qwen128_vproj_mm",
            load("results/coverage/cases/qwen128_vproj_conditional_debias.json.gz"),
            repeats=4,
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
        ),
        predict_source_fidelity_boundary(
            "qwen_layer23_attention_state",
            load("results/final/l23_s_bwd_antithetic.json"),
        ),
    ]


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
            "当前低成本路径不是一个已经完成的通用 oracle。六个严格 formation "
            f"positive 中，只有 {int(audit['strict_direct_recall'] * audit['strict_positive_count'])}/"
            f"{audit['strict_positive_count']} 能被当前实际接通的筛选路径直接报出风险；"
            "Liger 与 Phi 只能 fail-closed 升级，不能算作指标直接命中。"
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
        "## 指标缺口",
        "",
        "- Qwen64/128：四次 fixed-condition repeat 已能直接发现 local source risk，但下游层仍需顺序追加 repeat。",
        "- saved-P/SiLU：exact antithetic gradient pair 的 Adam response-even screen 可直接命中。",
        "- Mamba：local risk 可直接命中；真实 gradient/SGD 的条件覆盖仍不完整，所以必须保留 partial。",
        "- layer-23：natural-source fidelity 未通过，正确输出是 abstain，而不是 risk 或 safe。",
        "- Liger：缺的是由声明 schedule/operands 自动生成的 event-antithetic moment，旧 24/24 机制结果不能回填为新预测。",
        "- Phi：缺的是联合残差－transport moment；只计算 `J E[epsilon]` 会漏掉 `E[J_e epsilon_e]` 中的 covariance 通道。",
        "",
        "因此需要重新思考的是低成本指标的**输入与覆盖**，不是推翻条件反对称分解。下一版 screen 至少必须显式估计：",
        "",
        "`E[J_e epsilon_e | c] = E[J_e|c] E[epsilon_e|c] + Cov(J_e, epsilon_e | c)`。",
        "",
        "若仍只有 source mean 与局部 HVP，它会系统漏掉 Phi 型 pairing bias；若不能从 schedule 自动构造 event orbit，也会漏掉 Liger。",
        "",
        "## 边界",
        "",
        "这是已参与假设形成的 development recovery audit，不是 held-out accuracy。coded-group 与 shared-HVP 目前只有 synthetic/semantic-cut feasibility，尚未在这八个自然案例上运行，因此没有计作任何 case 的直接恢复。",
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
        "recovery": recovery,
        "decision": (
            "CURRENT_DIRECT_INDICATOR_INCOMPLETE__ADD_EVENT_AND_TRANSPORT_JOINT_MOMENTS"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recovery.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "summary.md").write_text(render(document), encoding="utf-8")


if __name__ == "__main__":
    main()
