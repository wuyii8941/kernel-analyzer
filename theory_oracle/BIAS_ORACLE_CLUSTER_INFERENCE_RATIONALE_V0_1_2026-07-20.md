# Bias Oracle cluster-inference rationale v0.1

## Decision

Qwen3 scalar primary endpoint 的 confirmatory population inference 以 independent trajectory
estimate 为顶层观测，而不是把 state、token 或 parameter coordinate 当独立样本。

对 trajectory `j`，先按冻结的 phase/state 权重计算 `B_j`。primary population estimate 为
trajectory estimates 的等权平均；primary interval/test 使用 `B_j` 上的 Studentized small-sample
procedure。轨迹内 states 用于提高 `B_j` 的精度和描述 phase/state heterogeneity，不增加顶层
自由度。

## Why this choice

1. 普通 cluster-robust asymptotics 在 cluster 数少时可能明显 over-reject。Cameron, Gelbach
   and Miller 的 Monte Carlo 研究正是针对这一问题：
   [Bootstrap-Based Improvements for Inference with Clustered Errors](https://doi.org/10.1162/rest.90.3.414)。
2. Ibragimov and Müller 提出对近似独立的 group-level estimates 使用普通 t statistic；在 group
   estimates 近似 Gaussian、可异方差的条件下，小样本性质清楚，适合把独立 trajectory 当
   group：
   [t-Statistic Based Correlation and Heterogeneity Robust Inference](https://doi.org/10.1198/jbes.2009.08046)。
3. Pustejovsky and Tipton 的 CR2/Satterthwaite 方法说明 conventional CRVE 在小样本会失真，
   并提供 small-sample correction；它适合作为 sensitivity analysis，而不是在当前尚未冻结
   regression working model 时直接宣布唯一方法：
   [Small-Sample Methods for Cluster-Robust Variance Estimation and Hypothesis Testing in Fixed Effects Models](https://doi.org/10.1080/07350015.2016.1247004)。
4. wild cluster bootstrap 可以改善某些 few-cluster 场景，但其保证依赖 cluster structure；
   cluster 极少、异质或不平衡时不能无条件信任：
   [The Wild Bootstrap with a Small Number of Large Clusters](https://doi.org/10.1093/restud/rdaa007)。
5. moving/block bootstrap 源于 stationary dependent sequence；Künsch 的一致性条件不允许我们
   对明显 early/middle/late 非平稳的训练轨迹不分层直接套用：
   [The Jackknife and the Bootstrap for General Stationary Observations](https://doi.org/10.1214/aos/1176347265)。

这组文献支持的是 inference design，不是说 DL compiler discrepancy 与经济学 panel 在机制上
相同。

## Primary procedure

1. 每条 trajectory 内按预声明 phase 权重、phase 内 state 权重计算 `B_j`；
2. trajectory seeds/data slices 相互独立是 construction gate；
3. `B_hat = mean_j(B_j)`；
4. primary confidence interval 使用 trajectory-level t interval，df=`J-1`；
5. 同时报告所有 `B_j`、range、sign 和 phase-conditioned estimates，避免一个 interval 隐藏
   transport failure；
6. CR2/Satterthwaite 和 wild-cluster bootstrap-t 作为 sensitivity checks；若它们与 primary
   产生实质冲突，verdict 为 `INDETERMINATE_METHOD_SENSITIVITY`，不挑有利结果。

## Assumptions and fail-closed conditions

- trajectories 必须独立生成；共享 ancestry/checkpoint 不是自动独立，需要在 query 中声明目标
  population 是“从该共同 anchor 出发的随机 trajectories”；
- `B_j` 应由足够多、按设计抽取的 trajectory-internal states 形成近似稳定 estimate；
- t procedure 需要 group estimates 的近似正态/对称条件。C1 simulation 必须覆盖 skew、heavy
  tail、rare-state mixture 和 unequal trajectory variance；
- 只有 4 个 calibration trajectories 不产生 population verdict；
- confirmation trajectory count 少导致 interval 宽是有效结果，不得用 states/tokens 补自由度；
- `P_R` 与 `P_C` 各自推断，不能把 anchor 类型当普通 trajectory 混入一个 t test。

## Precision and stopping

采用 two-stage design，而不是边看 confirmation sign 边增加数据：

1. calibration bank 只估计 trajectory-level variance、tail shape 和 measurement floor；
2. 在 confirmation 解盲前冻结 desired half-width `w_star`、`J_confirm` 和 resource cap；
3. `J_confirm` 由 calibration variance 和 t critical value 的 prospective calculation 选取，
   且不得低于合同下限；
4. confirmation 一次性运行冻结 evaluator；若 interval 仍过宽，返回 `INDETERMINATE_PRECISION`；
5. 任何扩样属于新版本合同，不复用旧 confirmation verdict 作无惩罚的 sequential peek。

若 practical tolerance 尚未实例化，`w_star` 可以只针对 shift-existence 的 measurement resolution
冻结；materiality ledger 保持 `UNINSTANTIATED`。

## C1 coverage audit

在设计匹配的 synthetic generator 上比较 trajectory-t、CR2 和 wild cluster bootstrap：

- exact null；
- fixed shift；
- phase-varying、population mean zero；
- correlated AR-like within-trajectory states；
- early/middle/late nonstationarity；
- unequal trajectory variance；
- skew/heavy-tail trajectory effects；
- rare-state mixture；
- same-state replay noise；
- unequal state counts and weighting traps。

选择标准按顺序为：null coverage/fail-closed、interval calibration、precision，再看 power。不能因某
方法更容易得到 nonzero bias 而选择它。

