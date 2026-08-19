#!/usr/bin/env python3
"""Build one evidence-bounded mechanism audit for all eight trajectory cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from kernel_analyzer.systematic_bias_audit import (
    first_conditional_bias_stage,
    validate_audit,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_systematic"


def read_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {relative}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def trajectory_index() -> dict[str, dict[str, Any]]:
    source = read_json("results/property/bias_formation_v22/trajectory_reclassification.json")
    return {str(row["case_id"]): row for row in source["cases"]}


def old_formation(relative: str) -> dict[str, str]:
    source = read_json(relative)
    confirmation = source["populations"]["confirmation"]
    return {
        "local": confirmation["LOCAL_ENDPOINT"]["status"].replace(
            "UNRESOLVED_INSUFFICIENT_STATES", "UNRESOLVED"
        ),
        "gradient": confirmation["PARAMETER_GRADIENT"]["status"].replace(
            "UNRESOLVED_INSUFFICIENT_STATES", "UNRESOLVED"
        ),
        "update": confirmation["EFFECTIVE_UPDATE"]["status"].replace(
            "UNRESOLVED_INSUFFICIENT_STATES", "UNRESOLVED"
        ),
    }


def trajectory(row: dict[str, Any]) -> dict[str, Any]:
    certificate = row["trajectory_certificate"]
    return {
        "status": certificate["status"],
        "steps": certificate["step_count"],
        "initial_drift_norm": certificate.get("initial_drift_norm"),
        "final_drift_norm": certificate.get("final_drift_norm"),
        "fixed_global_direction_passed_old_gate": row.get("old_fixed_direction_gate"),
        "formation_label": False,
        "artifact": row["artifact"],
    }


def symmetric_consequence(relative: str) -> dict[str, Any]:
    source = read_json(relative)
    evaluation = source["evaluation"]
    return {
        "status": source["status"],
        "evaluation_steps": evaluation["evaluation_steps"],
        "local_accumulation_l2": evaluation["local_accumulation_l2"],
        "feedback_accumulation_l2": evaluation["feedback_accumulation_l2"],
        "max_recurrence_relative_residual": evaluation["max_recurrence_relative_residual"],
        "stable_fixed_carrier": source["gates"].get("stable_calibration_carrier"),
        "signed_persistence": evaluation.get("signed_persistence"),
        "source": relative,
    }


def cases() -> list[dict[str, Any]]:
    trajectories = trajectory_index()
    not_measured = {"local": "NOT_MEASURED", "gradient": "NOT_MEASURED", "update": "NOT_MEASURED"}

    result = [
        {
            "case_id": "liger_fused_ce",
            "model": "Qwen3-1.7B + Liger fused linear CE",
            "semantic_unit": "Z=HW^T; G=(softmax(Z)-onehot)/N; dH=GW; dW=G^T H",
            "forward_backward": {"status": "CLOSED", "scope": "full fused loss F+B region"},
            "physical_source": (
                "64 two-token chunk contributions are added sequentially to a BF16 dW "
                "accumulator; chunk geometry changes the rounding schedule"
            ),
            "formation": {
                "conditional": dict(not_measured),
                "global": old_formation("results/property/bias_formation/formation/liger_fused_ce_t128.json"),
                "label_source": "OPEN_LOOP_GLOBAL_V21",
                "interpretation": "calibration was directional, disjoint confirmation unresolved; this is not a conditional null",
            },
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY", "P6_SEMANTIC_ORBIT_CENTERING"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_SOURCE_MECHANISM",
                "intervention": {
                    "description": "promote only dW accumulation to FP32 while preserving loss, dH, and all untied gradients",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "removed_candidate_added_error_fraction": 0.957,
                },
                "why_directional": (
                    "finite-precision sequential accumulation is conditionally asymmetric under the "
                    "declared chunk schedule, so E[epsilon|chunk geometry] need not vanish"
                ),
                "claim_boundary": "case-specific source mechanism; no universal P1 property and no M7",
            },
            "trajectory": {
                **trajectory(trajectories["liger_fused_ce"]),
                "local_feedback": symmetric_consequence(
                    "results/property/seup_mainline/liger_seup.json"
                ),
            },
            "next_decisive_test": "variance-matched stratum-mean removal if P1 is to become a general property",
            "evidence": [
                "archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json",
                "archive/nonprecision_v1/runs/liger.fused_ce.certificate.json",
                "archive/nonprecision_v1/runs/liger.fused_ce.chunk.certificate.json",
                "results/trajectory/liger_trajectory.json",
                "results/property/seup_mainline/liger_seup.json",
            ],
        },
        {
            "case_id": "phi4_seq64_lmhead_dx",
            "model": "Phi-4-mini",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at lm_head input VJP",
            "forward_backward": {"status": "CLOSED", "scope": "one exact backward MM invocation and both VJP edges"},
            "physical_source": "same-BF16-operand MM kernel arithmetic; final output rounding is noncoherent",
            "formation": {
                "conditional": dict(not_measured),
                "global": old_formation("results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json"),
                "label_source": "OPEN_LOOP_GLOBAL_V21",
                "interpretation": "the strongest existing population result is LOCAL_CENTERED -> GRADIENT_BIASED -> UPDATE_BIASED",
            },
            "mechanism": {
                "candidate_properties": ["P2_SOURCE_TRANSPORT_ALIGNMENT"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_TRANSPORT_MECHANISM",
                "intervention": {
                    "description": "permute residual/row transport pairing while preserving the local residual multiset and norm",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "natural_gradient": "BIASED",
                    "shuffled_gradient": "CENTERED",
                },
                "why_directional": (
                    "the local mean is small, but the real backward pairing makes Cov(T,epsilon|c) nonzero"
                ),
                "claim_boundary": "empirical composite transport mechanism; analytic transport reconstruction remains incomplete",
            },
            "trajectory": {
                **trajectory(trajectories["phi4_seq64_lmhead_dx"]),
                "local_feedback": symmetric_consequence(
                    "results/property/seup_mainline/phi_seup.json"
                ),
            },
            "next_decisive_test": "close the remaining analytic VJP factors before naming one physical transport factor",
            "evidence": [
                "results/coverage/cases/phi4_seq64_lmhead_dx.json",
                "results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json",
                "results/property/bias_formation/interventions/phi4_mm_transport_pairing.json",
                "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json",
            ],
        },
        {
            "case_id": "qwen64_vproj_mm",
            "model": "Qwen3-1.7B",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj",
            "forward_backward": {"status": "CLOSED", "scope": "one exact forward MM with actual AOT backward edges"},
            "physical_source": "precision contrast is directional; full split into kernel, output rounding, and inherited operands is incomplete",
            "formation": {"conditional": dict(not_measured), "global": dict(not_measured), "label_source": "NOT_MEASURED"},
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY"],
                "verdict": "PARTIAL_SOURCE_MECHANISM",
                "intervention": {
                    "description": "same-input FP32 MM accumulation followed by the original BF16 ABI",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "full_observed_source_repaired": False,
                },
                "why_directional": "the isolated accumulation residual changes the real dW path, but its relation to the complete precision residual is not closed",
                "claim_boundary": "trajectory-local partial source; no complete P1 attribution",
            },
            "trajectory": trajectory(trajectories["qwen64_vproj_mm"]),
            "next_decisive_test": "complete the three-way local source decomposition, then capture within-condition formation",
            "evidence": [
                "results/coverage/cases/qwen64_vproj.json",
                "results/coverage/cases/qwen64_vproj_repair_pilot.json",
                "results/coverage/cases/qwen64_vproj_trajectory.json",
            ],
        },
        {
            "case_id": "qwen128_vproj_mm",
            "model": "Qwen3-1.7B",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at layer-0 v_proj",
            "forward_backward": {"status": "CLOSED", "scope": "one exact forward MM with actual AOT backward edges"},
            "physical_source": "global local decomposition identifies deterministic FP32-to-BF16 output rounding, not MM kernel arithmetic",
            "formation": {"conditional": dict(not_measured), "global": dict(not_measured), "label_source": "NOT_MEASURED"},
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY", "P6_SEMANTIC_ORBIT_CENTERING"],
                "verdict": "UNRESOLVED_CONTRAST_MISMATCH",
                "intervention": {
                    "description": "existing trajectory promotes MM accumulation but retains BF16 output rounding",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                },
                "trajectory_repairs_declared_local_source": False,
                "why_directional": (
                    "the source decomposition and trajectory manipulate different contrasts; "
                    "they cannot yet be composed into one causal explanation"
                ),
                "claim_boundary": "do not attribute the trajectory to output rounding until an output-rounding repair is run",
            },
            "trajectory": trajectory(trajectories["qwen128_vproj_mm"]),
            "next_decisive_test": "run an exact output-rounding intervention with sham at the same F+B boundary",
            "evidence": [
                "results/coverage/cases/qwen128_vproj.json",
                "results/coverage/cases/qwen128_vproj_precision_decomposition.json",
                "results/coverage/cases/qwen128_vproj_repair_pilot.json",
                "results/coverage/cases/qwen128_vproj_trajectory.json",
            ],
        },
        {
            "case_id": "qwen_saved_p_seq128",
            "model": "Qwen3-1.7B",
            "semantic_unit": "p=softmax(a); da=p*(q-<p,q>) at layer-27 attention",
            "forward_backward": {"status": "CLOSED", "scope": "softmax forward, saved/reconstructed P, dS, and actual q/k VJPs"},
            "physical_source": "backward reconstructs P from BF16 logits plus FP32 max/sum instead of consuming true-forward FP32 P",
            "formation": {
                "conditional": dict(not_measured),
                "global": old_formation("results/property/bias_formation/formation/qwen_saved_p_seq128.json"),
                "label_source": "OPEN_LOOP_GLOBAL_V21",
                "interpretation": "global centered does not imply conditional or trajectory variance-only",
            },
            "mechanism": {
                "candidate_properties": ["P3_FORWARD_BACKWARD_NUMERICAL_CONSISTENCY", "P2_SOURCE_TRANSPORT_ALIGNMENT"],
                "verdict": "SUPPORTED_CASE_SPECIFIC_CONTRACT_MECHANISM",
                "intervention": {
                    "description": "replace reconstructed P by the exact true-forward P only at dS, retain BF16 dS ABI",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "forward_loss_unchanged": True,
                },
                "why_directional": (
                    "the implementation violates a forward/saved/backward representation contract; "
                    "its effect is trajectory-conditioned even though unrelated-state directions cancel"
                ),
                "claim_boundary": (
                    "validated case-specific contract difference; conditional formation stage is "
                    "unresolved, while symmetric recurrence shows local and feedback accumulation "
                    "of comparable norm without a stable fixed carrier"
                ),
            },
            "trajectory": {
                **trajectory(trajectories["qwen_saved_p_seq128"]),
                "local_feedback": symmetric_consequence(
                    "results/property/seup_mainline/qwen_softmax_seup.json"
                ),
            },
            "next_decisive_test": (
                "measure conditional local/gradient/update traces; symmetric recurrence is already "
                "closed and shows comparable local and feedback accumulation"
            ),
            "evidence": [
                "results/coverage/cases/qwen128_softmax_fb.json",
                "results/coverage/cases/qwen128_softmax_fb_formal.json",
                "results/property/bias_formation/formation/qwen_saved_p_seq128.json",
                "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json",
            ],
        },
        {
            "case_id": "qwen3vl_silu_layer0",
            "model": "Qwen3-VL-2B",
            "semantic_unit": "y=x*sigmoid(x); dx=q*sigmoid(x)*(1+x*(1-sigmoid(x)))",
            "forward_backward": {"status": "CLOSED", "scope": "same forward and one exact layer-0 SiLU backward invocation"},
            "physical_source": "AOT graph-dtype elementary backward arithmetic differs from native aten.silu_backward arithmetic",
            "formation": {"conditional": dict(not_measured), "global": dict(not_measured), "label_source": "NOT_MEASURED"},
            "mechanism": {
                "candidate_properties": ["P4_NONLINEAR_RECTIFICATION", "P3_FORWARD_BACKWARD_NUMERICAL_CONSISTENCY"],
                "verdict": "CAUSAL_IMPLEMENTATION_DIFFERENCE_FORMATION_UNRESOLVED",
                "intervention": {
                    "description": "swap only the target backward between decomposed and native implementations; forward is identical",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                },
                "why_directional": (
                    "the backward implementation causes real update differences, but no sign-symmetric "
                    "epsilon intervention or conditional formation trace identifies rectification"
                ),
                "claim_boundary": "complete causal F+B difference and trajectory; P3/P4 formation mechanism unresolved",
            },
            "trajectory": trajectory(trajectories["qwen3vl_silu_layer0"]),
            "next_decisive_test": "capture repeated within-condition traces and run a norm/support-matched +/-epsilon nonlinear control",
            "evidence": [
                "results/round2/vl_silu_cause.json",
                "results/round2/vl_silu_cause_fp32.json",
                "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
            ],
        },
        {
            "case_id": "mamba_seq64_input_proj",
            "model": "Mamba-130M",
            "semantic_unit": "Y=XW^T; dX=QW; dW=Q^T X at layer-0 in_proj",
            "forward_backward": {"status": "CLOSED", "scope": "one exact recurrent input-projection MM and actual VJP edges"},
            "physical_source": "both same-operand MM kernel arithmetic and deterministic output rounding are directional",
            "formation": {"conditional": dict(not_measured), "global": dict(not_measured), "label_source": "NOT_MEASURED"},
            "mechanism": {
                "candidate_properties": ["P1_CONDITIONAL_SOURCE_ASYMMETRY"],
                "verdict": "PARTIAL_SOURCE_MECHANISM",
                "intervention": {
                    "description": "promote only MM accumulation to FP32, then restore the BF16 ABI",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                    "full_observed_source_repaired": False,
                },
                "why_directional": (
                    "at least two additive local source terms are directional; the trajectory closes "
                    "the kernel-accumulation arm but not the output-rounding arm"
                ),
                "claim_boundary": "cross-architecture partial source mechanism; total observed error is not single-source",
            },
            "trajectory": trajectory(trajectories["mamba_seq64_input_proj"]),
            "next_decisive_test": "run separate kernel-only and output-rounding-only conditional interventions",
            "evidence": [
                "results/coverage/cases/mamba_seq64_input_proj.json",
                "results/coverage/cases/mamba_seq64_input_proj_precision_decomposition.json",
                "results/coverage/cases/mamba_seq64_input_proj_repair_pilot.json",
                "results/coverage/cases/mamba_seq64_input_proj_trajectory.json",
            ],
        },
        {
            "case_id": "qwen_layer23_attention_state",
            "model": "Qwen3-1.7B",
            "semantic_unit": "S_bwd=alpha*J_softmax(P)^T(DV^T); Gq=S_bwd*K; dWq=Gq^T H",
            "forward_backward": {"status": "CLOSED", "scope": "layer-23 q_proj attention-state semantic region and exact tile carrier"},
            "physical_source": "attention-backward state S_bwd is causal; upstream contributors overlap and include delayed key materialization",
            "formation": {"conditional": dict(not_measured), "global": dict(not_measured), "label_source": "SEMANTIC_REGION_CAUSAL_EVIDENCE"},
            "mechanism": {
                "candidate_properties": ["P2_SOURCE_TRANSPORT_ALIGNMENT", "P3_FORWARD_BACKWARD_NUMERICAL_CONSISTENCY", "P6_SEMANTIC_ORBIT_CENTERING"],
                "verdict": "SUPPORTED_SEMANTIC_REGION_TRANSPORT_CONTRACT_MECHANISM",
                "intervention": {
                    "description": "restore S_bwd at bmm_76; K-only repair is insufficient; joint S/K repair closes the direction",
                    "causal_effect": True,
                    "matched_sham_exact": True,
                },
                "why_directional": (
                    "the changed attention state is transported through Gq=S_bwd*K into a fixed q_proj tile; "
                    "S_bwd restoration removes that carrier"
                ),
                "claim_boundary": "validated semantic-region mechanism, not a uniquely identified kernel instruction",
            },
            "trajectory": trajectory(trajectories["qwen_layer23_attention_state"]),
            "next_decisive_test": "capture conditional layer traces if a first-bias-stage claim is required; do not force single-kernel attribution",
            "evidence": [
                "results/coverage/cases/l23_qproj_attention_state_region.json",
                "results/final/l23_attention_live_weight.json",
                "results/property/bias_formation_final/qwen_l23_attention_mechanism.json",
            ],
        },
    ]
    validate_audit(result)
    for case in result:
        case["formation"]["first_conditional_bias_stage"] = first_conditional_bias_stage(case)
    return result


def report(case: dict[str, Any]) -> str:
    cond = case["formation"]["conditional"]
    glob = case["formation"]["global"]
    mechanism = case["mechanism"]
    evidence = "\n".join(f"- `{path}`" for path in case["evidence"])
    local_feedback = case["trajectory"].get("local_feedback")
    consequence = ""
    if local_feedback:
        consequence = (
            "\n\n对称四反事实 recurrence 已测：local accumulation L2 "
            f"`{local_feedback['local_accumulation_l2']}`，feedback accumulation L2 "
            f"`{local_feedback['feedback_accumulation_l2']}`，最大相对闭合残差 "
            f"`{local_feedback['max_recurrence_relative_residual']}`。"
        )
    return f"""# {case['case_id']}

## 数学单位

模型：{case['model']}

F+B：{case['semantic_unit']}

闭合范围：{case['forward_backward']['scope']}（{case['forward_backward']['status']}）。

## 统一 bias 分解

使用 `E[Δg|c] = E[T|c]E[ε|c] + Cov(T,ε|c) + E[R(ε)|c]`。本例的物理差异是：{case['physical_source']}。

条件化 formation（local / gradient / update）：`{cond['local']} / {cond['gradient']} / {cond['update']}`。

旧跨无关状态结果（local / gradient / update）：`{glob['local']} / {glob['gradient']} / {glob['update']}`。它只描述 global/state-invariant bias，不替代 conditional bias。

## 机制判定

判定：`{mechanism['verdict']}`。

原因：{mechanism['why_directional']}。

干预：{mechanism['intervention']['description']}。

边界：{mechanism['claim_boundary']}。

## 轨迹后果

`{case['trajectory']['status']}`，{case['trajectory']['steps']} steps，drift norm `{case['trajectory']['initial_drift_norm']}` → `{case['trajectory']['final_drift_norm']}`。轨迹不提供 formation 标签。{consequence}

## 下一项决定性实验

{case['next_decisive_test']}。

## 证据

{evidence}
"""


def main() -> None:
    rows = cases()
    for case in rows:
        for relative in case["evidence"]:
            if not (ROOT / relative).is_file():
                raise FileNotFoundError(f"{case['case_id']}: {relative}")

    protocol = {
        "schema": "kernel-analyzer-systematic-bias-audit-protocol-v1",
        "status": "FROZEN_EIGHT_CASE_EVIDENCE_AUDIT",
        "question": "Why does each causally paired F+B implementation difference become directional training bias?",
        "case_denominator": 8,
        "mechanism_family_clusters": 7,
        "equations": {
            "formation": "E[delta_g|c] = E[T|c]E[epsilon|c] + Cov(T,epsilon|c) + E[R(epsilon)|c]",
            "optimizer": "delta_u = Opt(g+delta_g,z)-Opt(g,z)",
            "trajectory": "D_(t+1) = D_t + L_t + B_t + recurrence_residual_t",
        },
        "rules": [
            "Analyze one complete forward plus its actual backward as the unit.",
            "Use conditional, trajectory, and global bias as separate claims.",
            "A global centered result is not a conditional null.",
            "A trajectory is consequence evidence and never labels formation.",
            "A supported mechanism requires a causal intervention and exact matched sham.",
            "Do not join evidence produced by different repaired contrasts.",
            "SEUP describes persistence only after formation evidence exists.",
        ],
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "case_audit.json", {"schema": "kernel-analyzer-systematic-bias-audit-v1", "cases": rows})

    matrix_rows = []
    for case in rows:
        cond = case["formation"]["conditional"]
        glob = case["formation"]["global"]
        matrix_rows.append({
            "case_id": case["case_id"],
            "model": case["model"],
            "fb": case["forward_backward"]["status"],
            "conditional_local": cond["local"],
            "conditional_gradient": cond["gradient"],
            "conditional_update": cond["update"],
            "global_local": glob["local"],
            "global_gradient": glob["gradient"],
            "global_update": glob["update"],
            "mechanism_verdict": case["mechanism"]["verdict"],
            "trajectory": case["trajectory"]["status"],
            "next_decisive_test": case["next_decisive_test"],
        })
    matrix_path = OUT / "case_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = matrix_path.with_name(".case_matrix.csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix_rows)
    temporary.replace(matrix_path)

    reports = OUT / "case_reports"
    reports.mkdir(parents=True, exist_ok=True)
    for case in rows:
        (reports / f"{case['case_id']}.md").write_text(report(case), encoding="utf-8")

    counts: dict[str, int] = {}
    for case in rows:
        verdict = case["mechanism"]["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
    gap_order = [
        "qwen128_vproj_mm", "qwen_saved_p_seq128", "qwen3vl_silu_layer0",
        "qwen64_vproj_mm", "mamba_seq64_input_proj", "phi4_seq64_lmhead_dx",
        "liger_fused_ce", "qwen_layer23_attention_state",
    ]
    write_json(OUT / "gap_plan.json", {
        "schema": "kernel-analyzer-systematic-bias-gap-plan-v1",
        "principle": "run only experiments that resolve a specific missing link in the common equation",
        "ordered_cases": [
            {
                "case_id": case_id,
                "experiment": next(row["next_decisive_test"] for row in rows if row["case_id"] == case_id),
            }
            for case_id in gap_order
        ],
    })

    summary = f"""# 八案例 Bias Formation 系统审计

## 结论

8 个案例都具有完整或语义闭合的 F+B 边界和因果成对轨迹，但它们**尚不能被解释为一个共同 property 的 8 个正例**。当前最严格的机制分层是：

- Liger：case-specific source mechanism；
- Phi：case-specific composite transport mechanism；
- saved-P：case-specific F/B numerical-contract mechanism；conditional formation 未测，四反事实 recurrence 已闭合且 local/feedback 累积同量级；
- layer-23：semantic-region transport/contract mechanism，不是单 kernel root；
- Qwen64 与 Mamba：partial source mechanisms；
- Qwen128 v_proj：source decomposition 与 trajectory repair 不是同一 contrast，暂不能拼接；
- Qwen3-VL SiLU：因果 backward implementation difference 和 trajectory 已闭合，bias formation mechanism 未闭合。

## 为什么会出现系统性 bias

统一解释不是“误差大”，而是条件化的一阶与高阶项：

`E[Δg|c] = E[T|c]E[ε|c] + Cov(T,ε|c) + E[R(ε)|c]`。

- `E[ε|c] != 0`：source arithmetic 在声明条件下已经有方向（Liger；MM source candidates）。
- `Cov(T,ε|c) != 0`：局部 residual 可近似居中，但真实 backward pairing 将其整流（Phi）。
- `E[R(ε)|c] != 0` 或 numerical contract 改变：saved/reconstructed state 与 backward 表示使语义区域产生方向（saved-P、layer-23）。
- optimizer 还可能把 centered gradient residual 变成 update bias，但 8 个案例中尚无严格 P5 positive。
- 进入轨迹后，`D_(t+1)=D_t+L_t+B_t+r_t`；local effect 与 feedback 决定差异持续还是抵消。固定 global carrier 不是必要条件。

## 当前可以声称什么

可以声称：多种 implementation difference 会通过 source asymmetry、backward transport 或 F/B contract 的不同路径形成训练相关的方向性更新，并在闭环轨迹中造成参数分离。

不能声称：已经发现一个跨全部 8 个案例的统一 property；也不能把 global-centered saved-P 称为 variance-only，或把 Qwen128 的 output-rounding source 与 accumulation repair trajectory 拼成同一因果链。

## 下一步

优先补三个最能改变结论的实验：Qwen128 output-rounding matched repair、saved-P conditional formation、SiLU conditional formation + sign-symmetric nonlinear control。其余缺口见 `gap_plan.json`。
"""
    (OUT / "scientific_summary.md").write_text(summary, encoding="utf-8")
    write_json(OUT / "summary.json", {
        "status": "COMPLETE_EXISTING_EVIDENCE_SYSTEMATIC_AUDIT",
        "cases": len(rows),
        "mechanism_family_clusters": len({row["mechanism_family"] for row in trajectory_index().values()}),
        "mechanism_verdict_counts": counts,
        "cross_case_property": "NOT_YET_IDENTIFIED",
        "next_gpu_work_is_gap_driven": True,
    })
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "cases": len(rows), "verdicts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
