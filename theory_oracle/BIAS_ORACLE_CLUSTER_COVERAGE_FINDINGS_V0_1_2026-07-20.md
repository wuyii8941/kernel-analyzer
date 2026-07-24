# Bias Oracle cluster-coverage findings v0.1

## Question

当 states 嵌套在少数 independent training trajectories 中时，trajectory-level t interval 是否
足以支持 population average-shift claim？把所有 states 当独立样本会造成多大错误？

## Design

- 5,000 Monte Carlo trials/cell；
- 8/12/16/24 trajectories；每条 24 states；
- true population mean 为零；
- top-level effects 覆盖 Gaussian、heteroskedastic Gaussian、Student-t3、centered lognormal 和
  5% rare-cluster mixture；
- 比较 trajectory means 上的 t interval 与错误地把所有 state means 当 independent 的 t interval。

Machine-readable result：
`results/oracle_calibration/bias_oracle_cluster_inference_coverage_v0_1/evaluation.json`。

## Main result

| trajectory law | J=8 trajectory coverage | J=24 trajectory coverage | naive state coverage（大致） |
|---|---:|---:|---:|
| Gaussian | 0.949 | 0.952 | 0.30 |
| heteroskedastic Gaussian | 0.957 | 0.953 | 0.28–0.31 |
| Student-t3 | 0.962 | 0.960 | 0.30–0.31 |
| centered lognormal | 0.838 | 0.885 | 0.26–0.30 |
| 5% rare trajectory regime | 0.347 | 0.708 | 0.00–0.37 |

结论不是“trajectory t 永远正确”。它在独立、近似对称 group estimates 下表现符合预期，但
强偏斜和 rare top-level regimes 会破坏 small-J mean inference。naive state inference 在所有
clustered scenarios 中都严重 overconfident。

## Theoretical implication

少量 trajectories 无法通过统计技巧识别一个未观察到、但对 population mean 有较大贡献的
rare regime。hierarchical bootstrap、更多 tokens 或更多同轨迹 states 都不能创造缺失的
top-level support。

若要以至少 `1-alpha` 的概率观察 prevalence 不低于 `p_min` 的 trajectory regime，最小独立
trajectory 数满足：

`J_tail >= ceil(log(alpha) / log(1-p_min))`。

例如 `alpha=0.05`：

- 10% prevalence 需要约 29 trajectories；
- 5% prevalence 需要约 59 trajectories；
- 1% prevalence 需要约 299 trajectories。

这只是“至少观察一次”的覆盖，不保证精确估计其 effect 或 prevalence。

## Oracle consequence

population ledger 必须分开：

1. **regularity-conditional average shift**：trajectory estimates 满足预声明近似独立/对称或
   working-law 条件时，使用 trajectory-level small-sample inference；
2. **tail coverage**：预声明 `p_min/alpha`，由 trajectory count 给出 design coverage，并报告
   observed extreme/tail effects；
3. **finite-bank effect**：若 regularity/tail gate 不足，只对实际 bank 作描述，不外推 natural
   training prevalence。

不能用样本 skewness test “证明没有 rare tail”；小样本未拒绝 skewness 不是 tail coverage。

## Planning decision

- 8 trajectories 只作为 regularity-oriented first confirmation batch，不是普遍充分样本量；
- 若论文目标包含 5% trajectory-regime coverage，Q-R 至少需要 59 independent trajectories；
- 若资源不支持该数量，必须收窄 claim：报告 regularity-conditional mean 与 finite-bank result，
  不声称覆盖 rare trajectory bias；
- bias-contributor screening 必须继承同一 population/tail scope，不能用少量 repair trajectories
  获得比 baseline B 更强的普遍性。

