# Liger Kernel Single-Step Diff Report
Generated: 2026-09-19 00:01:48

## 说明

本报告对每个 target 做**单步 forward + backward** 的 diff 测试，
排除多步训练累积误差的干扰，直接测试 kernel 单次调用的数值等价性。

判定标准：
- 🔴 **SIGNIFICANT**: `grad_diff_max > 1e-2` 或 `loss_diff > 0.1`（明确 kernel 数值问题）
- 🟡 **NOTABLE**: `grad_diff_max > 1e-3` 或 `loss_diff > 0.01`（需关注）
- 🟢 **NEGLIGIBLE**: diff 在 bf16 精度范围内（正常）

---

## 汇总

| Target | 结论 | loss diff (单步) | grad diff max | 说明 |
|--------|------|----------------|---------------|------|
| `orpo_loss` | 🔴 SIGNIFICANT | `6.3470e-01` | `1.5717e-03` | ORPO loss 单步：比较 LigerFusedLinearORPOLoss vs log-sp |
| `tvd` | 🔴 SIGNIFICANT | `5.2051e-01` | `N/A` | TVD loss：prob 输入 vs log_prob 输入的差异（API 文档 bug） |

---

## 🔴 `orpo_loss` — SIGNIFICANT

**配置**: `ORPO loss 单步：比较 LigerFusedLinearORPOLoss vs log-space odds ratio`（2 次重复，取最大 grad_diff）

| 指标 | 值 |
|------|----|
| loss (baseline log-space) | `1.1267e+01` |
| loss (LigerFusedLinearORPOLoss) | `1.0632e+01` |
| **loss diff (单步)** | **`6.3470e-01`** |
| lm_head.weight grad diff (max) | `1.5717e-03` |
| lm_head.weight grad diff (mean) | `2.2247e-05` |
| baseline grad norm | `1.1226e+00` |
| Liger grad norm | `7.4200e-01` |

> bl=log-space odds ratio (variant B), ts=LigerFusedLinearORPOLoss

---

## 🔴 `tvd` — SIGNIFICANT

**配置**: `TVD loss：prob 输入 vs log_prob 输入的差异（API 文档 bug）`（2 次重复，取最大 grad_diff）

| 输入类型 | Liger 输出 | 与正确值之差 | 梯度 norm |
|---------|-----------|------------|---------|
| `prob`（softmax 后） | `1.6266e-05` | **`5.2051e-01`** | `1.2353e-04` |
| `log_prob`（log_softmax 后） | `5.6417e-01` | `4.3645e-02` | `1.2353e-04` |
| 参考值（手动计算 TVD） | `5.2053e-01` | — | — |

**结论**: BUG: prob_input diff >> logp_input diff

> `LigerTVDLoss` 期望传入的是概率分布（`prob`），但计算逻辑假设输入是 `log_prob`，
> 导致传入 `prob` 时产生 4-5 个数量级的偏差。这是一个 **API 文档缺失 / 输入验证缺失** 的 bug。

---
