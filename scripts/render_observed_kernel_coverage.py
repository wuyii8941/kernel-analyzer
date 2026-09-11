#!/usr/bin/env python3
"""Render the exhaustive observed-kernel catalogue summary as one table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_analyzer.operator_taxonomy import FAMILY_LABELS


def render(summary: dict) -> str:
    support = summary["distinct_family_support_status_counts"]
    implementation = summary["distinct_family_implementation_kind_counts"]
    lines = [
        "# 全部已观测 kernel 覆盖清单",
        "",
        "这张表按保存的实际训练执行记录自动生成。位置数、kernel 调用数、已支持测量数和已确认数值问题是不同量；本表不把自动分类当作参考语义、bias 或根因证明。",
        "",
        f"- 保存目录：{summary['release_count']}；不同任务清单：{summary['distinct_release_task_package_count']}；重复目录：{summary['duplicate_release_directory_count']}。",
        f"- 含目录副本的输出记录：{summary['position_count']}；不同任务位置：{summary['distinct_release_qualified_position_count']}；实际 kernel 调用记录：{summary['kernel_invocation_count']}。",
        f"- 去重后的有效测量位置：{summary['distinct_position_support_status_counts'].get('VALID_MEASUREMENT_COMPLETED', 0)}；已具备参考和训练绑定、尚待测量：{summary['distinct_position_support_status_counts'].get('READY_FOR_MEASUREMENT', 0)}。",
        "",
        "| 计算家族 | 输出位置 | Triton | 其他实现 | 已识别待补参考/绑定 | 有参考待绑定 | 可直接测量 | 有效测量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILY_LABELS:
        count = summary["distinct_position_family_counts"].get(family, 0)
        if count == 0:
            continue
        stages = support.get(family, {})
        kinds = implementation.get(family, {})
        triton = kinds.get("TRITON", 0)
        lines.append(
            f"| {FAMILY_LABELS[family]} (`{family}`) | {count} | {triton} | {count-triton} | "
            f"{stages.get('IDENTIFIED', 0)} | {stages.get('REFERENCE_AVAILABLE', 0)} | "
            f"{stages.get('READY_FOR_MEASUREMENT', 0)} | {stages.get('VALID_MEASUREMENT_COMPLETED', 0)} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- `已识别`只表示实际执行位置和保守计算分类存在，不表示已有可信 reference。",
        "- `可直接测量`表示已有 reference 与目标参数绑定；其中 AOT 端点重放覆盖闭合区域，不能自动归因为单个 kernel。",
        "- `有效测量`只表示采集和统一复算通过，不表示发现 bias，更不表示已有训练后果。",
        "- 新运行队列对相同任务包去重，并优先不同计算家族和不同结构签名，不按历史数值大小挑选。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    summary = json.loads(args.summary.read_text())
    if summary.get("schema") != "observed-kernel-catalog-summary-v1":
        raise ValueError("Unexpected catalogue summary schema")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
