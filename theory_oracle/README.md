# ForkCert Matched-State B/H/N Oracle

用 B（平均实现效应）、H（state-conditioned heterogeneity）、N（same-state runtime
variability）和 U（sampling uncertainty）描述实现差异，并分别连接语义事件、自然一步
转移、correctness authority 与 operator attribution。

## 核心文档

**[Matched-State Implementation Discrepancy Oracle v3](DISCREPANCY_ORACLE_DEFINITION_V3.md)** — 当前规范；统一 continuous、semantic、one-step transition、correctness 与 attribution ledgers。

[B/H/N Oracle v2](BIAS_VARIANCE_ORACLE_DEFINITION_V2.md) 是 v3 的统计分解基础，保留为演化记录。

[v1](BIAS_VARIANCE_ORACLE_DEFINITION_V1.md) 保留为历史审计材料，其中“零均值噪声安全”
和“非零 B 线性累积”等说法不再作为定义。

## 实现

| 文件 | 说明 |
|------|------|
| `src/forkcert/oracle.py` | 核心实现：B/H/N 分解、Verdict 判定、TrainingOracle |
| `tests/test_oracle.py` | 核心 Oracle 测试，包含 sign cancellation、sampling uncertainty、repeat/H identifiability、coverage 与 multi-module/step matched-state 反例 |
| `scripts/training_oracle_experiment.py` | synthetic twin-trajectory illustration；不是 matched-state safety 实验 |
| `scripts/oracle_demo.py` | 合成数据 + torch 模型的快速演示 |
| `theory_oracle/qwen3_grpo_natural_transition_v0_2.py` | Qwen frozen-state 真实 AdamW/AMP/clip/scheduler 一步执行器 |
| `theory_oracle/evaluate_qwen3_grpo_natural_transition_v0_2.py` | 四臂 fresh-process transition B/N、event 与 next-state evaluator |
| `tests/test_qwen3_grpo_natural_transition_v0_2.py` | transition clipping direction、B/N 分离与 vector direction 反例 |
| `theory_oracle/qwen3_historical_case_runner_v0_1.py` | patch-hidden Qwen3 historical case 的自然边界与通用 operation trace |
| `theory_oracle/qwen3_blind_locator_v0_1.py` | 只用 endpoint/region/trace Oracle 的盲定位器；不输出 root-cause claim |
| `theory_oracle/verify_qwen3_blind_protocol_v0_1.py` | 检查 opaque case 是否泄漏 issue、patch、fixed revision 或 source artifact |
| `theory_oracle/qwen3_checkpoint_operator_probe_v0_1.py` | Qwen3-1.7B 真实 checkpoint 的 eager/compiled scale-up gate 与 FX provenance |
| `theory_oracle/run_qwen3_compiler_grad_case_v0_1.py` | Qwen3-shaped higher-order-gradient compiler case runner |
| `theory_oracle/locate_qwen3_compiler_grad_case_v0_1.py` | 不依赖 bug metadata 的 semantic-operation blind locator |

当前算子定位工作流的权威入口是
[localization evidence ledger](../reports/LOCALIZATION_EVIDENCE_LEDGER_V0_1.md) 和
[frozen localization-method plan](../reports/LOCALIZATION_METHOD_PLAN_V1.md)。Qwen 专用
脚本保留为 development/regression evidence，不是冻结后方法的独立准确性评估。

`historical_evaluation_protocol_v0_1.py` provides the Phase-3 handoff:
`seal` freezes a generic locator certificate before a patch is disclosed, and
`score` consumes an independently evaluator-owned post-reveal truth label.
It reports coverage and stopping accuracy but cannot itself prove that the
analyst had no private patch knowledge.

## 一句话

> 在多个预声明 matched training states 上，对每个 operator/observable 分开估计 B/H/N/U，
> 再检查这些差异是否传播为语义事件或自然一步状态转移；没有独立真值时不声称 correctness。

## 历史文档

旧合同文档仍是 correctness、semantic impact、transition 与 attribution ledger 的来源；
B/H/N 是 discrepancy profile 的核心，而不是替代这些 ledger 的单一安全分数。

Claude 生成的 200-step MLP artifact 已按
[reclassification](TRAINING_ORACLE_SYNTHETIC_RESULT_RECLASSIFICATION_2026-07-18.md)
降级为 twin-trajectory illustration，不能作为“zero-mean noise 安全”的实验证据。

真实 Qwen 证据见 [scorer B/H/N findings](QWEN3_GRPO_BHN_FINAL_FINDINGS_2026-07-18.md)
和 [natural transition findings](QWEN3_GRPO_NATURAL_TRANSITION_FINDINGS_V0_2_2026-07-18.md)。
