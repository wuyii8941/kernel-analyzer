#!/usr/bin/env python3
"""Summarize family-first campaigns from task status and analysis artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def failure_reason(run: Path, execution: str) -> str | None:
    """Extract a short machine-readable reason without reinterpreting failure as data.

    The frozen capture writer intentionally stores only the return code and a
    terminal status.  Keep the original log as the source of truth, but make
    the family summary useful for scheduling repairs by surfacing the final
    exception line when one is present.
    """
    if execution not in {"EXECUTION_FAILED", "INCOMPLETE_ATTEMPT"}:
        return None
    log = run / "capture.log"
    if not log.exists():
        return "CAPTURE_LOG_MISSING"
    lines = [line.strip() for line in log.read_text(errors="replace").splitlines() if line.strip()]
    markers = ("ValueError:", "RuntimeError:", "ModuleNotFoundError:",
               "ImportError:", "AssertionError:", "NotImplementedError:",
               "torch._dynamo.exc.Unsupported:")
    for line in reversed(lines):
        if line.startswith(markers) or any(marker in line for marker in markers):
            for marker in markers:
                if marker in line:
                    start = line.index(marker)
                    return line[start:start + 500]
            return line[:500]
    return lines[-1][-500:] if lines else "CAPTURE_LOG_EMPTY"


def summarize(manifest: dict) -> dict:
    rows = []
    for campaign in manifest["campaigns"]:
        output = Path(campaign["campaign_output"])
        specialized = campaign.get("adapter") not in {None, "GENERIC_COVERAGE"}
        run_id = hashlib.sha256(campaign["task_id"].encode()).hexdigest()[:20]
        run = output / "runs" / run_id
        completion_path = output / "completion_verification.json"
        if specialized and completion_path.exists():
            completion = json.loads(completion_path.read_text())
            records = completion.get("records", [])
            record = next((r for r in records if r.get("task_id") == campaign["task_id"]), {})
            execution = "VALID" if record.get("status") == "VERIFIED" or record.get("status") == "RECORDED_MEASUREMENT_CHECKED" else record.get("status", "INCOMPLETE")
            analysis = record.get("analysis") or {}
            run = output
        else:
            status_path = run / "status.json"
            if status_path.exists():
                execution = json.loads(status_path.read_text()).get("status", "UNDECLARED")
            elif run.exists():
                execution = "INCOMPLETE_ATTEMPT"
            else:
                execution = "NOT_STARTED"
        if not specialized:
            analysis_path = run / "analysis.json"
            analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
        bias = analysis.get("bias_analysis", {})
        rows.append({
            "operator_family": campaign["operator_family"],
            "case_id": campaign["case"]["case_id"],
            "task_id": campaign["task_id"],
            "implementation_kind": campaign.get(
                "implementation_kind", "NOT_RECORDED_IN_V1_MANIFEST"
            ),
            "reference_method": campaign["case"]["reference_method"],
            "reference_scope": (
                "CLOSED_REGION_NOT_SINGLE_KERNEL_SOURCE"
                if campaign["case"]["reference_method"] == "AOT_REPLAY"
                else "COMMON_OPERAND_EXTERNAL_RECOMPUTE"
            ),
            "execution_status": execution,
            "failure_reason": failure_reason(run, execution),
            "measurement_status": analysis.get("measurement_status", "NOT_ASSESSED"),
            "equivalence_decision": analysis.get("equivalence_decision", "NOT_ASSESSED"),
            "fixed_suite_parameter_write_total_rms": bias.get("fixed_suite_total_rms"),
            "fixed_suite_parameter_write_aligned_ratio": bias.get("fixed_suite_aligned_ratio_of_sums"),
            "population_guarantee": False,
            "bias_or_root_cause_confirmed_by_this_row": False,
            "training_outcome_measured": False,
        })
    return {
        "schema": "family-first-campaign-summary-v1",
        "selection_uses_numerical_outcomes": False,
        "campaign_count": len(rows),
        "execution_status_counts": dict(Counter(row["execution_status"] for row in rows)),
        "valid_measurement_count": sum(row["measurement_status"] == "VALID" for row in rows),
        "scope": (
            "BOUNDED_FAMILY_INTERFACE_VALIDATION; NOT_A_BIAS_COUNT, ROOT_CAUSE_COUNT, "
            "OR TRAINING_OUTCOME STUDY"
        ),
        "rows": rows,
    }


def markdown(result: dict) -> str:
    lines = [
        "# 不同计算家族的首轮自动测量",
        "",
        "任务在读取数值结果前按计算家族选择。AOT reference 的结果只表示闭合区域替换作用，不自动证明单个 kernel 的 bias 或根因。",
        "",
        "| 计算家族 | reference | 执行 | 测量 | 参数写入 RMS | 固定集合判断 | 失败原因 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in result["rows"]:
        rms = row["fixed_suite_parameter_write_total_rms"]
        lines.append(
            f"| `{row['operator_family']}` | `{row['reference_method']}` | `{row['execution_status']}` | "
            f"`{row['measurement_status']}` | {('N/A' if rms is None else format(rms, '.8g'))} | "
            f"`{row['equivalence_decision']}` | {row['failure_reason'] or 'N/A'} |"
        )
    lines += [
        "",
        "有效测量、系统性 bias、数学根因和训练后果必须分别确认；本表只自动汇总前者。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_json, args.output_md):
        if path.exists() or not path.resolve().is_relative_to(Path("/data1/tzh")):
            parser.error("Choose new outputs under /data1/tzh")
    manifest = json.loads(args.manifest.read_text())
    result = summarize(manifest)
    result["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "campaign_count": result["campaign_count"],
        "execution_status_counts": result["execution_status_counts"],
        "valid_measurement_count": result["valid_measurement_count"],
    }))


if __name__ == "__main__":
    main()
