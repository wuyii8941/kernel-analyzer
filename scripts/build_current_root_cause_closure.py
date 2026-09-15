#!/usr/bin/env python3
"""Build the current, conservative root-cause closure ledger.

The ledger is deliberately problem-oriented: model positions and repeated
conditions are evidence for a problem group, not additional root causes.  All
numeric fields for the newly incorporated GELU and selection controls are
recomputed from the retained raw records.  This report does not promote a
nonzero norm to a nonzero mean bias, and it does not turn an unresolved source
into a negative result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json"
OUT_MD = ROOT / "docs/root_cause_closure_current.md"


def read(path: str | Path) -> Any:
    return json.loads((ROOT / path if not Path(path).is_absolute() else Path(path)).read_text())


def confirmation_rows(raw: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    ids = set(map(str, raw.get("confirmation_state_ids", [])))
    return [row for sid, row in zip(raw["state_ids"], raw["original_coordinate_statistics"][stage])
            if str(sid) in ids]


def rms(raw: dict[str, Any], stage: str) -> float:
    rows = confirmation_rows(raw, stage)
    effect = sum(float(row["effect_energy"]) for row in rows)
    repair = sum(float(row["repair_energy"]) for row in rows)
    if repair <= 0:
        return 0.0 if effect == 0 else math.inf
    return math.sqrt(effect / repair)


def aligned(raw: dict[str, Any], stage: str) -> float:
    rows = confirmation_rows(raw, stage)
    effect = sum(float(row.get("effect_repair_inner_product", 0.0)) for row in rows)
    repair = sum(float(row["repair_energy"]) for row in rows)
    return effect / repair if repair else 0.0


def gelu_evidence() -> dict[str, Any]:
    base = "results/property/numerical_coverage_v1"
    dirs = {
        "natural_reference": "gemma_gelu_capture_v2",
        "explicit_exponential_tanh": "gemma_gelu_tanh_exp_confirmation_v3",
        "native_tanh": "gemma_gelu_tanh_native_confirmation_v3",
        "fused_multiply_add": "gemma_gelu_fma_confirmation_v1",
    }
    records: dict[str, Any] = {}
    for label, directory in dirs.items():
        raw = read(f"{base}/{directory}/raw/gelu_backward_952_out_ptr0.json")
        records[label] = {
            "state_count": len(raw["state_ids"]),
            "confirmation_count": len(raw["confirmation_state_ids"]),
            "local_rms": rms(raw, "LOCAL"),
            "gradient_rms": rms(raw, "PARAMETER_GRADIENT"),
            "update_rms": rms(raw, "ADAMW_UPDATE"),
            "write_rms": rms(raw, "PARAMETER_WRITE"),
            "update_aligned_ratio": aligned(raw, "ADAMW_UPDATE"),
            "reference_variant": raw["reference_comparison_scope"]["reference_variant"],
            "same_local_operands": raw["reference_comparison_scope"]["same_local_operands"],
        }
    profile_fields = ("local_rms", "gradient_rms", "update_rms", "write_rms", "update_aligned_ratio")
    records["native_tanh_vs_fma_exact_in_recorded_profile"] = all(
        records["native_tanh"][field] == records["fused_multiply_add"][field]
        for field in profile_fields
    )
    return {
        "case_id": "gemma_gelu_backward_952",
        "source": "generated Triton tanh-GELU backward product; same operands and one declared trainable parameter",
        "status": "SOURCE_CHOICE_RESPONSE_CONFIRMED_NATURAL_MEAN_BIAS_NOT_CONFIRMED",
        "evidence": records,
        "interpretation": (
            "The explicit exponential tanh choice changes the measured gradient/update profile, "
            "whereas native tanh and its fused multiply-add spelling agree on the disjoint confirmation. "
            "This identifies a source-choice-sensitive response, not a persistent natural bias or a unique "
            "hardware root for every GELU position."
        ),
    }


def liger_order_confirmation_evidence() -> dict[str, Any]:
    result = {}
    for length in (64, 256):
        data = read(f"results/property/liger_fp32_chunk_order_v1/length{length}_confirmation.json")
        update = data["profiles"]["ADAMW_UPDATE"]["population_inference"]
        gradient = data["profiles"]["PARAMETER_GRADIENT"]["population_inference"]
        result[str(length)] = {
            "state_count": len(data["state_ids"]),
            "confirmation_positive_count": data["source_prediction"]["confirmation_positive_count"],
            "confirmation_count": data["source_prediction"]["confirmation_count"],
            "direction_repeated": data["source_prediction"]["direction_repeated"],
            "gradient_branches_confirmed": {
                name: bool(branch["raw_confirmed"])
                for name, branch in gradient["branches"].items()
            },
            "update_branches_confirmed": {
                name: bool(branch["raw_confirmed"])
                for name, branch in update["branches"].items()
            },
            "claim_boundary": data["claim_boundary"],
        }
    training = read("results/property/liger_fp32_chunk_order_v1/fp32_order_training_1024.json")
    result["training_1024"] = {
        "validation_loss_difference": training["validation_loss_difference"],
        "claim_boundary": training["claim_boundary"],
    }
    return result


def silu_factorial_evidence() -> dict[str, Any]:
    raw = read(
        "results/property/numerical_coverage_v1/silu_factorial_explicit_source_run3/raw/"
        "mapped_backward_667_in_out_ptr0-silu-common-input.json"
    )
    result = {
        "case_id": raw["case_id"],
        "source_task_id": raw["source_task_id"],
        "state_count": len(raw["state_ids"]),
        "confirmation_count": len(raw["confirmation_state_ids"]),
        "same_local_operands": raw["reference_comparison_scope"]["same_local_operands"],
        "source_attribution": raw["reference_comparison_scope"]["single_kernel_source_attribution"],
        "stages": {},
    }
    for stage in ("LOCAL", "PARAMETER_GRADIENT", "ADAMW_MOMENT1_WRITE", "ADAMW_MOMENT2_WRITE", "ADAMW_UPDATE", "PARAMETER_WRITE", "NEXT_STEP_COMMON_GRADIENT_UPDATE"):
        profiles = []
        for value in raw["stages"][stage].values():
            suite = value["profile"]["suite"]
            profiles.append({
                "total_effect_rms": suite["total_effect_rms"],
                "mean_effect_over_repair_rms": suite["mean_effect_over_repair_rms"],
                "repair_aligned_effect": suite["repair_aligned_effect"],
                "residual_direction_heldout_effect": suite.get("residual_direction_heldout_effect"),
            })
        result["stages"][stage] = {
            "total_effect_rms_range": [min(x["total_effect_rms"] for x in profiles), max(x["total_effect_rms"] for x in profiles)],
            "mean_effect_over_repair_rms_range": [min(x["mean_effect_over_repair_rms"] for x in profiles), max(x["mean_effect_over_repair_rms"] for x in profiles)],
            "repair_aligned_effect_range": [min(x["repair_aligned_effect"] for x in profiles), max(x["repair_aligned_effect"] for x in profiles)],
        }
    return result


def rms_order_evidence() -> dict[str, Any]:
    data = read("results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/trajectory32_isolated.json")
    cases = {}
    for task, row in data["cases"].items():
        cases[task] = {
            "native_endpoint_rms": row.get("variants", {}).get("FP32_NATIVE", {}).get("endpoint", {}).get("total_effect_rms"),
            "reverse_endpoint_rms": row.get("variants", {}).get("FP32_REVERSE_FEATURE_ORDER", {}).get("endpoint", {}).get("total_effect_rms"),
            "direct_endpoint_rms": row.get("source_intervention", {}).get("endpoint_total_rms_absolute_difference"),
            "direct_gradient_rms": row.get("source_intervention", {}).get("reference_parameter_gradient_difference", {}).get("total_effect_rms"),
            "direction_status": row.get("direction_status"),
        }
    return {
        "case_id": "gemma_rms_feature_reduction_order",
        "source": "FP32 feature-reduction order in a real compiled Gemma graph",
        "status": "SOURCE_CONTROL_NEGATIVE_NO_ROOT_CAUSE_FOUND",
        "state_count": data["state_count"],
        "evidence": cases,
        "interpretation": (
            "Isolating the two endpoints shows one tiny changed endpoint without a confirmed direction "
            "and one exact identity. The earlier combined replacement was an upstream region effect, "
            "so no RMS endpoint is promoted as a root cause."
        ),
    }


def selection_evidence() -> dict[str, Any]:
    data = read("results/property/numerical_coverage_v1/granite_selection_suite24_v1/summary.json")
    records = data["records"]
    return {
        "case_id": "granite_router_topk_sort",
        "source": "ATen Top-k/sort selection path with deterministic tie-order variant",
        "status": "NO_DIFFERENCE_OBSERVED_SELECTION_VARIANT_NOT_A_BIAS_CASE",
        "state_count": len(records),
        "all_scores_equal": all(row.get("same_scores") for row in records),
        "all_gradients_zero_difference": all(
            row["original_coordinate_statistics"]["PARAMETER_GRADIENT"]["effect_energy"] == 0
            for row in records
        ),
        "all_writes_zero_difference": all(
            row["original_coordinate_statistics"]["PARAMETER_WRITE"]["effect_energy"] == 0
            for row in records
        ),
        "selected_set_unchanged": all(row.get("selected_expert_set_equal", True) for row in records),
        "interpretation": (
            "Index order can change for equal scores while the selected expert set, values, gradients, "
            "writes and loss remain equal in the fixed suite. This is a useful semantic negative control, "
            "not a discovered bias."
        ),
    }


def granite_expert_evidence() -> dict[str, Any]:
    data = read(
        "results/property/training_numerical_analysis_v2/"
        "granite_expert_order_confirmation_v2/recomputed.json"
    )
    bias = data["bias_analysis"]
    return {
        "case_id": data["case_id"],
        "contrast_id": data["contrast_id"],
        "state_count": len(bias["confirmation_state_ids"]),
        "fixed_suite_total_rms": bias["fixed_suite_total_rms"],
        "fixed_suite_aligned_ratio_of_sums": bias["fixed_suite_aligned_ratio_of_sums"],
        "equivalence_decision": data["equivalence_decision"],
        "training_outcome": data["training_outcome"],
    }


def gemma_square_sum_binding_evidence() -> dict[str, Any]:
    failure = read(
        "results/property/training_numerical_analysis_v2/recovery/"
        "gemma_bound_square_sum/live_source_failure.json"
    )
    plan = read(
        "results/property/training_numerical_analysis_v2/recovery/"
        "gemma_bound_square_sum/case_plan.json"
    )
    case = plan["cases"][0]
    return {
        "case_id": case["case_id"],
        "task_id": case["task_id"],
        "status": failure["status"],
        "measurements_accepted": failure["measurements_accepted"],
        "error": failure["error"],
        "scope": failure["scope"],
    }


def main() -> None:
    prior = read("results/property/case_causal_audit_v1/scientific_case_closure.json")
    rows = []
    coverage_collections = []
    for row in prior["rows"]:
        if row["problem_group"] in {"common_input_silu_and_rms_backward_families", "reference_graph_regions"}:
            coverage_collections.append({
                "collection": row["problem_group"],
                "status": row["closure"],
                "scope": row["implementation_boundary"],
                "interpretation": row["remaining_limit"],
                "evidence": row["evidence"],
            })
            continue
        current = {
            "problem_group": row["problem_group"],
            "closure": row["closure"],
            "numerical_source": row["numerical_source"],
            "bias_formation": row["bias_formation"],
            "training_outcome": row["training_outcome"],
            "what_is_proven": row["remaining_limit"],
            "next_needed_observation": row["next_root_cause_test"],
            "evidence": row["evidence"],
        }
        if row["problem_group"] == "silu_backward_evaluation":
            current["closure"] = "SOURCE_CHOICE_RESPONSE_CLOSED_NATURAL_BIAS_OPEN"
            current["numerical_source"] = (
                "finite-precision source choice in the sigmoid/SiLU derivative expression "
                "at a generated Triton gate-gradient endpoint"
            )
            current["bias_formation"] = (
                "a same-operand explicit exponential source variant changes the measured "
                "gradient, moment and update/write profiles; natural mean asymmetry remains open"
            )
            current["what_is_proven"] = (
                "one AST-checked gate-gradient endpoint has a locally isolated source-choice "
                "response across all recorded stages; this is not a universal SiLU root"
            )
            current["next_needed_observation"] = (
                "only needed for a stronger natural-bias claim: separate sigmoid approximation, "
                "expression order and final cast on an independent state bank"
            )
            current["evidence"] = list(row["evidence"]) + [
                "results/property/numerical_coverage_v1/silu_factorial_source_manifest_v2.json",
                "results/property/numerical_coverage_v1/silu_factorial_explicit_source_run3/raw/mapped_backward_667_in_out_ptr0-silu-common-input.json",
            ]
            current["derived"] = silu_factorial_evidence()
        if row["problem_group"] == "liger_fused_linear_ce_dw_accumulation":
            current["what_is_proven"] = (
                "the local addition-order source and its direction are reproduced in disjoint "
                "length-64 and length-256 confirmation banks; the retained training run does not "
                "establish a material loss consequence"
            )
            current["next_needed_observation"] = (
                "only needed for a stronger claim: preserve original-coordinate writes or run a "
                "predeclared longer paired loss comparison"
            )
            current["evidence"] = list(row["evidence"]) + [
                "results/property/liger_fp32_chunk_order_v1/length64_confirmation.json",
                "results/property/liger_fp32_chunk_order_v1/length256_confirmation.json",
                "results/property/liger_fp32_chunk_order_v1/fp32_order_training_1024.json",
            ]
            current["derived"] = liger_order_confirmation_evidence()
        rows.append(current)
    rows.extend([
        {
            "problem_group": "gemma_gelu_backward_evaluation",
            "closure": "SOURCE_CHOICE_RESPONSE_CLOSED_NATURAL_BIAS_OPEN",
            "numerical_source": "tanh evaluation and expression spelling in a generated Triton tanh-GELU backward product",
            "bias_formation": "same-input source variants change gradient/update response on the selected confirmation case; native tanh and fused multiply-add agree",
            "training_outcome": "not measured as an independent full training result",
            "what_is_proven": "a source-choice-sensitive response for one bound position; not a stable natural mean bias and not a universal GELU root",
            "next_needed_observation": "only needed if a natural GELU bias claim is desired: predeclared independent source distribution and training endpoint",
            "evidence": ["results/property/numerical_coverage_v1/gemma_gelu_capture_v2", "results/property/numerical_coverage_v1/gemma_gelu_tanh_exp_confirmation_v3", "results/property/numerical_coverage_v1/gemma_gelu_tanh_native_confirmation_v3", "results/property/numerical_coverage_v1/gemma_gelu_fma_confirmation_v1"],
            "derived": gelu_evidence(),
        },
        {
            "problem_group": "gemma_rms_feature_reduction_order",
            "closure": "NEGATIVE_SOURCE_CONTROL_NO_ROOT_CAUSE",
            "numerical_source": "FP32 feature-reduction order",
            "bias_formation": "isolated order change is tiny or exact identity and has no confirmed direction",
            "training_outcome": "not measured",
            "what_is_proven": "the tested order change is not sufficient to explain the earlier combined endpoint effect",
            "next_needed_observation": "none for this negative control; a different RMS source would require a new predeclared intervention",
            "evidence": ["results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/trajectory32_isolated.json"],
            "derived": rms_order_evidence(),
        },
        {
            "problem_group": "granite_router_topk_selection",
            "closure": "NEGATIVE_CONTROL_FIXED_SUITE_IDENTITY",
            "numerical_source": "selection tie-order variation with unchanged selected set",
            "bias_formation": "no local, gradient, write or loss difference observed in 24 fixed states",
            "training_outcome": "no difference observed; no population or quality claim",
            "what_is_proven": "the tested legal tie-order variation did not create a measurable training difference under the declared deterministic protocol",
            "next_needed_observation": "none unless a different selection semantic (changed selected set, NaN, or non-tie scores) is explicitly studied",
            "evidence": ["results/property/numerical_coverage_v1/granite_selection_suite24_v1/summary.json"],
            "derived": selection_evidence(),
        },
        {
            "problem_group": "granite_moe_expert_contribution_order",
            "closure": "SOURCE_CLOSED_FIXED_SUITE_NATURAL_BIAS_OPEN",
            "numerical_source": "FP32 expert-contribution accumulation order",
            "bias_formation": "reversing the legal expert accumulation order creates a small nonzero write difference in the declared fixed suite; the source choice is isolated, while a population mean bias is not assessed",
            "training_outcome": "not measured",
            "what_is_proven": "the new model/MoE family reuses the common update analysis and has a fixed-suite RMS of about 0.002281%; this is not a unique root or a population bias claim",
            "next_needed_observation": "if promoted beyond a coverage confirmation, isolate expert accumulation from routing and run an independent state-bank confirmation",
            "evidence": [
                "results/property/training_numerical_analysis_v2/granite_expert_order_confirmation_v2/recomputed.json",
                "docs/training_numerical_analysis_v2.md",
            ],
            "derived": granite_expert_evidence(),
        },
        {
            "problem_group": "gemma_bound_square_sum",
            "closure": "EXECUTION_SOURCE_MISMATCH_UNRESOLVED",
            "numerical_source": "intended bound square-sum endpoint was not the actual executed kernel source",
            "bias_formation": "not assessable because the live source formula differed and no replacement measurement was accepted",
            "training_outcome": "not measured",
            "what_is_proven": "the attempted recapture failed closed at source binding; this is an unresolved measurement, not a negative bias result",
            "next_needed_observation": "bind the actual executed endpoint and re-run the declared square-sum comparison before making any root-cause claim",
            "evidence": [
                "results/property/training_numerical_analysis_v2/recovery/gemma_bound_square_sum/live_source_failure.json",
                "results/property/training_numerical_analysis_v2/recovery/gemma_bound_square_sum/case_plan.json",
            ],
            "derived": gemma_square_sum_binding_evidence(),
        },
    ])
    payload = {
        "schema": "kernel-analyzer-current-root-cause-closure-v1",
        "status": "CURRENT_EVIDENCE_LEDGER_WITH_EXPLICIT_OPEN_BRANCHES",
        "counting_rule": "One row is one deduplicated scientific problem group; model, layer, state and implementation conditions are evidence within a group.",
        "closure_scale": {
            "END_TO_END": "source, propagation, targeted intervention and a declared training outcome are connected",
            "SOURCE_CLOSED": "the local arithmetic/source choice is isolated, while natural bias or quality consequence remains open",
            "PARTIAL": "a transport or semantic-region contributor is isolated, but the whole region has additional sources",
            "NEGATIVE_CONTROL": "the tested variant did not produce a difference; this does not prove every variant is safe",
            "OPEN": "existing measurements show a difference or response, but competing source explanations remain",
        },
        "summary": {
            "problem_group_count": len(rows),
            "end_to_end_count": sum(r["closure"].startswith("END_TO_END") for r in rows),
            "source_or_local_closed_count": sum("CLOSED" in r["closure"] for r in rows),
            "negative_control_count": sum(r["closure"].startswith("NEGATIVE_CONTROL") or r["closure"].startswith("NEGATIVE_SOURCE") for r in rows),
            "groups_with_open_branch": sum("OPEN" in r["closure"] for r in rows),
        },
        "coverage_collections": coverage_collections,
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    lines = [
        "# 当前根因闭环账本",
        "",
        "本表按科学问题组去重。它把‘已经知道差异来自哪里’、‘已经证明形成了 bias’和‘已经看到训练后果’分开记录。模型位置、层号和训练状态是同一问题组内的证据，不重复计数。",
        "",
        f"当前共 {len(rows)} 个科学问题组；其中 {payload['summary']['end_to_end_count']} 个连到声明的 material loss improvement，{payload['summary']['source_or_local_closed_count']} 个至少完成了局部来源闭环，{payload['summary']['negative_control_count']} 个是阴性控制；另保留 {len(coverage_collections)} 个测量覆盖集合。其余开放分支明确列出，不用‘有差异’代替根因。",
        "",
        "| 问题组 | 当前根因结论 | 训练后果 | 还缺什么 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['problem_group']}` | {row['closure']}；{row['numerical_source']} | {row['training_outcome']} | {row['next_needed_observation']} |")
    lines += [
        "",
        "## 本轮新增的可核对结论",
        "",
        "Liger 的同精度加法顺序来源已经有互斥的长度 64 和 256 确认：确认半区分别为 14/16 和 11/16 沿预先声明方向，参数梯度与零矩 AdamW 首步分支也有记录支持。它仍使用摘要更新量，1024 步配对 loss 差异回到零，因此根因来源闭合，但训练质量后果没有闭合。",
        "",
        "SiLU 的 source-factorial run3 在同一调用前输入和同一 gate-gradient 输出上，显式指数求值相对候选改变了 local、gradient、moment、update 和 write profile；实际执行位置经过函数 AST 与调用前输入核对。该结果闭合一个局部求值来源，但不代表整个 SiLU 家族或自然输入总体存在平均 bias。",
        "",
        "Gemma GELU 的同一位置、相同输入和独立确认状态上，显式指数重构会改变 gradient/update profile，而 native tanh 与融合乘加表达式在记录的 profile 中一致。这把 GELU 的来源缩小到求值选择的响应差异，但没有证明自然输入总体存在稳定 mean bias。",
        "",
        "Gemma RMS 的 endpoint-by-endpoint 归约顺序干预说明，早期同时替换造成的较大差异来自上游传播，不能归因给第二个 RMS endpoint。Granite Top-k 的 24 状态结果则是完整阴性控制：合法的相等分数索引排列没有改变选中集合、值、梯度、写入或 loss。",
        "",
        "## 使用边界",
        "",
        "所有固定集合结果只覆盖各自声明的状态、参数和实现边界。‘源已闭合’不自动表示自然总体 bias 已闭合；‘训练轨迹不相同’不自动表示质量持续恶化。后续若要把开放分支升级，必须新增能区分表中竞争解释的观测，而不是从已有聚合量反推。",
        "",
        "机器结果：`results/property/case_causal_audit_v1/root_cause_closure_current.json`。",
        "",
        "## 覆盖集合（不计入科学问题组）",
        "",
        "原审计中的 99 个同输入 SiLU/RMS 位置和 31 个 reference-graph 区域保留为覆盖集合。它们证明统一流程可以运行，但没有逐项完成根因隔离，因此不计入上表的独立问题组数量。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT_JSON), "groups": len(rows), "end_to_end": payload["summary"]["end_to_end_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
