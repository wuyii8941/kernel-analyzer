# Qwen3 update-aligned shift retrospective findings v0.1

## Scope

复用 A/B/C 三个 selected states 的 eager/compiled natural parameter-update vectors，对 C0 草案
中的 reference-update-aligned endpoint 做 sanity check。没有新增 GPU execution。

这三个 states 不是目标训练分布样本；summary 只用于 endpoint selection，不承担 population
bias、operator attribution、long-run impact 或 correctness 结论。

## Result

| state | full update discrepancy | aligned shift | geometry |
|---|---:|---:|---|
| A | exact zero | zero | null |
| B | nonzero | positive, very small | almost entirely orthogonal to eager update |
| C | nonzero | negative, very small | almost entirely orthogonal to eager update |

B/C 的 relative discrepancy L2 都在约千分之几尺度；aligned component 比它小约两个数量级，
且 B/C 符号相反。三状态平均不具有 population 含义。

Machine-readable evidence：
`results/oracle_calibration/qwen3_update_aligned_shift_retrospective_v0_1/evaluation.json`。

## Interpretation

1. aligned shift 不是 raw L2 的重命名。它能区分“update 大小有差异”与“沿 baseline training
   direction 系统性加速/削弱”。
2. A/B/C 不支持一个跨 selected states 固定方向的 aligned bias，但样本设计不足以拒绝 target
   population 上的 average shift。
3. orthogonal discrepancy 不能自动叫 variance：在当前 repeats 下它是 deterministic local
   effect。它可能随 state 旋转，也可能通过曲率、optimizer state 或后续反馈产生 task effect。
4. 因此 aligned shift 适合做 update-dynamics endpoint，不足以单独承担“导致宏观训练差异”的
   bias。后者需要 task-level one-step transition functional 或长期 validation。

## Planning consequence

Oracle 使用两个并列、不能互相替代的 primary ledgers：

- **update-bias ledger**：aligned shift、coordinate-frame mean field、orthogonal magnitude；适合
  连接 operator repair；
- **task-transition bias ledger**：同一 pre-state 下两种 next states 在冻结 evaluation
  functional 上的差异；适合连接宏观训练含义。

只有 update ledger 成立时，结论是“operator contributes to average update shift”。要称
“contributes to macro-relevant bias”，至少还需要 task-transition transport；长期 harm 仍需独立
multi-step validation。

