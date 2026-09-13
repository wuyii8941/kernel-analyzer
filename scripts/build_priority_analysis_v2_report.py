#!/usr/bin/env python3
"""Build concise tables and documentation for completed priority analyses 1 and 2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); document = args.document.resolve()
    if document.exists() or not document.is_relative_to(ROOT):
        parser.error("choose a new document inside kernel-analyzer")
    moment = load(root / "moment_reconstruction/result.json")
    selective = load(root / "selective_parameter_compensation/result.json")
    units = [load(path) for path in sorted((root / "selective_parameter_compensation/units").glob("*.json"))]
    method = load(root / "method_incremental_value.json")
    verification = load(root / "verification.json")
    if verification["status"] != "VERIFIED":
        raise ValueError("priority analysis has not passed independent verification")

    selective_csv = root / "selective_parameter_compensation/units.csv"
    with selective_csv.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "unit", "key_energy_fraction", "off_rms", "all_rms",
            "key_only_rms", "rest_only_rms", "complement_checks",
        ])
        writer.writeheader()
        for row in units:
            writer.writerow({
                "unit": row["unit"],
                "key_energy_fraction": row["off_effect_energy_fraction_in_key"],
                "off_rms": row["write_rms"]["OFF"],
                "all_rms": row["write_rms"]["ALL"],
                "key_only_rms": row["write_rms"]["KEY_ONLY"],
                "rest_only_rms": row["write_rms"]["REST_ONLY"],
                "complement_checks": row["all_complement_checks"],
            })

    method_csv = root / "method_incremental_value.csv"
    with method_csv.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "scenario", "local_rms", "update_rms", "update_mean",
            "one_state_rms", "full_decision", "confirmed_structure",
        ])
        writer.writeheader()
        for row in method["controlled_scenarios"]:
            methods = row["methods"]
            writer.writerow({
                "scenario": row["name"], "local_rms": methods["LOCAL_RMS"]["value"],
                "update_rms": methods["UPDATE_RMS"]["value"],
                "update_mean": methods["UPDATE_MEAN_ONLY"]["value"],
                "one_state_rms": methods["ONE_CONFIRMATION_STATE"]["value"],
                "full_decision": methods["FULL_ANALYSIS"]["decision"],
                "confirmed_structure": "+".join(methods["FULL_ANALYSIS"]["confirmed_structure"]),
            })

    off = moment["summary"]["OFF"]
    on = moment["summary"]["ON"]
    key = selective["key_fraction_of_off_effect_energy"]
    lines = [
        "# 优先分析1和2：机制定位与方法新增信息\n\n",
        "本页由保存结果自动生成。结论限于各协议声明的数据与训练条件。\n\n",
        "## 结论\n\n",
        "1. AdamW8bit 保存的量化残差可以逐步重构后续 moment 差；在固定梯度中，",
        "一阶 moment 对写入差异的作用大于二阶 moment。\n",
        "2. 预先指定的 embedding 与最后一层 mixer output projection 在8条新 history 中",
        "稳定占据绝大多数未补偿误差。只补偿这两组参数能去掉大部分写入误差，",
        "但全参数补偿仍明显更接近 FP32。\n",
        "3. 对固定集合的二元等价判断，完整方法与全空间 update RMS 相同；新增价值来自",
        "定位差异在哪个训练阶段形成、区分差异结构，以及指导修改。\n",
        "4. 单步 update RMS 降低不是 loss 改善的充分条件；历史 block64 和 FP32一阶",
        "moment 修改是阴性对照，跨步残差补偿是已确认正例。\n\n",
        "## 1. Moment递推重构\n\n",
        "在32个固定梯度记录中，显式保留浮点求值余项后，moment误差重构的最大相对残差为",
        f" `{verification['checks']['maximum_moment_relative_reconstruction_residual']:.9g}`。",
        "这验证代数传播路径，不证明残差具有非零总体均值。\n\n",
        "| 条件 | 写入总差异 | 仅一阶moment | 仅二阶moment | 两者共同余项 |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for label, row in (("关闭补偿", off), ("开启补偿", on)):
        value = row["mean_write_component_relative_rms"]
        lines.append(
            f"| {label} | {value['total_relative_rms']:.6%} | "
            f"{value['first_moment_only_relative_rms']:.6%} | "
            f"{value['second_moment_only_relative_rms']:.6%} | "
            f"{value['joint_nonlinear_remainder_relative_rms']:.6%} |\n"
        )
    lines.extend([
        "\n这里的一阶／二阶结果来自同一当前步的数值替换；它们因AdamW分母的非线性不能简单相加。\n\n",
        "## 2. 参数组互补验证\n\n",
        f"8条 history 中，关键参数占未补偿误差能量的范围为 `{key['minimum']:.4%}`–",
        f"`{key['maximum']:.4%}`，平均 `{key['mean']:.4%}`。\n\n",
        "| 补偿范围 | 平均写入RMS（相对FP32） |\n",
        "|---|---:|\n",
    ])
    labels = {"OFF": "不补偿", "ALL": "全部参数", "KEY_ONLY": "仅关键参数",
              "REST_ONLY": "仅其余参数"}
    for mode in ("OFF", "KEY_ONLY", "REST_ONLY", "ALL"):
        lines.append(f"| {labels[mode]} | {selective['mean_write_rms'][mode]:.6%} |\n")
    lines.extend([
        "\n8/8条history中，仅关键参数都比仅其余参数更接近FP32；8/8中全补偿仍最好。",
        "所有互补路径检查逐位成立。这是optimizer响应定位，不是局部补偿的loss确认。\n\n",
        "## 3. 方法对照\n\n",
        "9个共享的高维受控场景通过同一生产统计函数。下表是针对预先构造的固定集合",
        "update范围判断，不是自然kernel检出率。\n\n",
        "| 方法 | 正确场景数/9 | 它缺少的信息 |\n",
        "|---|---:|---|\n",
    ])
    descriptions = {
        "LOCAL_RMS": "1/9", "UPDATE_RMS": "9/9", "UPDATE_MEAN_ONLY": "7/9",
        "ONE_CONFIRMATION_STATE": "8/9", "FULL_ANALYSIS": "9/9",
    }
    missing = {
        "LOCAL_RMS": "不知道差异经过backward和optimizer后的结果",
        "UPDATE_RMS": "不知道固定方向、随状态缩放或剩余方向",
        "UPDATE_MEAN_ONLY": "会漏掉零均值高能量和随正常update旋转的缩放",
        "ONE_CONFIRMATION_STATE": "会漏掉只在其他状态出现的差异",
        "FULL_ANALYSIS": "仍不能由短程profile直接预测loss",
    }
    for mode in descriptions:
        lines.append(f"| {mode} | {descriptions[mode]} | {missing[mode]} |\n")
    lines.extend([
        "\n完整方法在固定集合等价判断上没有超过update RMS，因为update RMS正是其强制",
        "全空间范围。它增加的是诊断信息，不能写成更高的二元检测准确率。\n\n",
        "## 4. 真实修改的回顾性对照\n\n",
        "| 修改 | 单步写入差异 | 训练判断 |\n",
        "|---|---|---|\n",
    ])
    for row in method["empirical_modification_evidence"]:
        write = row["write_evidence"]
        lines.append(
            f"| {row['modification']} | {write['default_rms']:.6g} → "
            f"{write['modified_rms']:.6g} | {row['training_decision']} |\n"
        )
    lines.extend([
        "\n三种修改并非使用同一批训练数据，不能作随机化排名。它们共同支持一个较窄结论：",
        "降低单步写入RMS不足以预测loss，跨步状态中的误差信息为修改选择提供了额外依据。\n\n",
        "## 仍未建立\n\n",
        "- 关键参数局部补偿能复现全补偿的训练loss改善；\n",
        "- 平均bias而非误差波动是改善的唯一原因；\n",
        "- 相对RENDER或TTrace的检测准确率优势；\n",
        "- 跨模型、checkpoint或数据分布的机制外推。\n",
        "机器结果与核验位于 `results/property/result_analysis_v2/`。\n",
    ])
    document.write_text("".join(lines))
    print(json.dumps({"status": "COMPLETE", "document": str(document),
                      "csv": [str(selective_csv), str(method_csv)]}))


if __name__ == "__main__":
    main()
