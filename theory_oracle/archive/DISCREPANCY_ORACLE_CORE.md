# Discrepancy Decomposition Module（不是 Oracle 判定本身）

> Status correction — 2026-07-16: this document remains the measurement/decomposition module for Operator Oracle v0.1. Its earlier use of “current main Oracle” is superseded by `OPERATOR_ORACLE_V0_1_DEFINITION_2026-07-16.md`. Correctness is primarily judged against an input-conditioned semantic envelope; relative bias/heterogeneity/runtime variability explain discrepancy and support an explicit compatibility contract.

## 1. 研究中心

本模块不以 fork 为核心，而以 **operator-scale implementation discrepancy** 为测量对象：

> 对真实 operator input cases，同一 semantic operator 的 reference/compiled 实现产生怎样的 relative bias、input-conditioned heterogeneity 和 exact-input runtime variability；这些 operator discrepancy 是否传播到语义决策或一步 transition？

fork、clipping、sampling 等离散现象仅作为下游 stress/impact validation，不定义 correctness Oracle。relative discrepancy decomposition 本身也不定义 pass/fail。

这里 operator 是分析对象；operator input case 是统计样本；完整训练 state 仅用于复现该输入；region 仅是可能的联合干预单元，不能与 operator 等同。

## 2. 首先重新定义“误差”

在 Operator Oracle v0.1 中，若规范允许集合为 `S_o(x)`，primary conformance error 是 candidate 到该集合的违反/距离。以下 `D_o` 是 candidate-baseline discrepancy；只有在显式 `S4` compatibility contract 中才直接成为 verdict 对象。

### 2.1 没有独立真值时

对 observable `Y`，定义 paired implementation discrepancy：

```text
D(s, ξ) = Y_C(s, ξ_C) - Y_R(s, ξ_R)
```

此时 `D` 不是 mathematical error。可称：

- implementation discrepancy；
- reference-relative difference；
- paired transition discrepancy。

### 2.2 有独立真值时

若存在高精度或规范 reference `Y*`，才能定义：

```text
e_R = Y_R - Y*
e_C = Y_C - Y*
```

并进一步讨论 accuracy bias、correctness violation 或谁更接近真值。

### 2.3 Primary subject 是 operator

对 semantic operator instance `o` 和真实 operator input case `x`，定义：

```text
D_o(x, ξ) = Z_C,o(x, ξ_C) - Z_R,o(x, ξ_R)
```

operator identity 同时包含 call site/phase、attributes、shape、dtype、layout/device signature 和 compiler configuration。训练 state 只负责产生并复现真实 input case `x`，不是分析尺度。

`D_o` 可以是 tensor/vector 或预声明 invariant representation，不要求先平均成 scalar。operator family 是跨 instances/signatures 的受控汇总。region 是可能的多算子 intervention unit，physical kernel 是实现 artifact；二者都不能与 operator identity 混用。

## 3. 定义性分解

固定 operator input population `Q_o` 和 randomness protocol。令：

```text
m_o(x) = Eξ[D_o(x, ξ) | X=x]
B_o = EX~Q_o[m_o(X)]
h_o(x) = m_o(x) - B_o
η_o(x, ξ) = D_o(x, ξ) - m_o(x)
```

则：

```text
D_o(x, ξ) = B_o + h_o(x) + η_o(x, ξ)
```

其中：

- `B_o`：真实 operator input population 上的 implementation-relative operator bias；
- `h_o(x)`：input-conditioned deviation / effect heterogeneity；
- `η_o(x,ξ)`：exact-input runtime fluctuation。

operator output 的 coordinate variation 不能自动称为 runtime variance；operator-family、checkpoint 和 downstream endpoint aggregation 都必须声明其 conditioning/weighting。

这是对象的定义性分解，不是假设三项一定显著非零。实际哪一项存在、大小多少，需要实验估计。

需要强调：主数学对象首先是条件分布 `Law(D_o | X=x)`，而不是它的 mean/variance。relative bias、input heterogeneity 和 runtime variance 只是该分布的摘要；若存在重尾、多峰、偏态、异方差或 rare extremes，它们不构成完整 error model。

在二阶矩存在时，total variance law 给出：

```text
Var(D_o) = VarQ_o(m_o(X)) + EQ_o[Varξ(D_o | X)]
```

第一项是跨 operator input cases 的 effect heterogeneity，第二项才是 exact-input runtime variability。

## 4. “Bias”一词的使用规则

### 4.1 当前可用

若明确写成 relative/implementation bias，可将 `B_o` 或其预声明 family aggregation 解释为：

> 相对于 baseline、在特定 `Q` 上的平均方向偏移。

### 4.2 当前禁止

没有 `Y*` 时，不将 `B_o` 或其聚合称为：

- mathematical bias；
- compiler accuracy bias；
- systematic correctness error。

### 4.3 Bias field 及其聚合不是 compiler 固有常数

它依赖：

- operator input population `Q_o`；
- observable `Y`；
- randomness contract；
- implementation configuration；
- vector discrepancy 的投影或汇总方式。

换 operator input population、signature、checkpoint、configuration 或聚合方式，operator bias structure 可以改变大小甚至符号。

## 5. 需要同时回答的两个 variance 问题

### 5.1 每个实现自身的条件 variance

```text
Var(Y_R | s), Var(Y_C | s)
```

回答哪个实现自身更不稳定，以及 compiler 是否改变 output/update 的噪声结构。

### 5.2 Paired discrepancy 的条件 variance

```text
Var(Y_C - Y_R | s)
```

它依赖两边 coupling，因为：

```text
Var(Y_C-Y_R | s)
= Var(Y_C|s)+Var(Y_R|s)-2Cov(Y_C,Y_R|s)
```

因此报告 variance 时必须同时说明 marginal variance 与 paired coupling，不能只报差值 variance。

## 6. 误差来源不能预先归类

| 来源 | 可能表现 |
|---|---|
| fixed reduction tree | deterministic input-conditioned shift，也可能形成非零 operator relative bias |
| nondeterministic reduction order | runtime variability，并可能同时改变条件均值 |
| fixed reassociation/cast placement | deterministic discrepancy，方向可能随 state 改变 |
| ordinary fixed-mode floating-point rounding | 通常是确定性的，不自动属于 variance |
| stochastic rounding | runtime randomness；均值是否为零取决于条件和传播 |
| autotuning | cache 后可成为 fixed configuration；重新选择时是更高层随机性 |
| algorithmic RNG | intended stochasticity，需与 execution noise 分开 |

分类由固定 state 下的重复行为和条件分布决定，不由来源名称决定。

## 7. Observable hierarchy

主 Oracle 不应只选一个方便观察的 scalar。需要建立 observable hierarchy：

### 7.1 Operator/intermediate outputs

用于研究 discrepancy 在哪里首次产生、怎样传播。高维输出需要预先定义坐标、方向或结构化摘要，避免向量均值互相抵消。

### 7.2 Model outputs and losses

用于研究局部 discrepancy 是否到达模型级 observable。

### 7.3 Gradient and update

用于研究训练一步中的实际 forcing：

- mean update discrepancy；
- covariance/noise discrepancy；
- 与 reference gradient 或 task-relevant direction 的投影；
- optimizer state transition discrepancy。

### 7.4 Discrete events

只作为下游影响或非线性转换验证，不作为主分解定义。

## 8. Transition-level extension

训练更自然的对象是一步 transition kernel：

```text
K_I^T(· | s) = Law(S' | S=s, implementation=I)
```

比较两种实现的：

- conditional mean transition / drift；
- conditional covariance / noise structure；
- tail behavior；
- task-relevant projected update。

这里可借用“drift difference”和“noise/diffusion difference”的直觉，但不能假设训练长期差异是局部 drift 的线性累加。

## 9. Operator measurement profile（解释 Oracle，不替代 verdict）

在 expected relation、acceptable set 和 verdict function 冻结前，对每个 operator/endpoint 只能报告：

1. target `Q`；
2. implementation/configuration；
3. randomness 与 coupling protocol；
4. reference 与 compiled 各自的 within-state variability；
5. operator relative bias `B_o` 与预声明 output projections；
6. input-conditioned mean function `m_o(x)` 或其可解释结构；
7. effect heterogeneity；
8. paired discrepancy variance 与 covariance；
9. tails/quantiles，防止 variance 掩盖非高斯和 rare extremes；
10. task-relevant projection；
11. single-operator repair/injection 的 identifiable status；region intervention 另行报告；
12. 是否存在独立 truth，从而允许 correctness 解释。

## 10. 探索实验顺序

## P-1. 使用已经定义的 Oracle 判定关系

在正式 operator measurement 前，从 Operator Oracle v0.1 实例化 semantic envelope、estimand map、population acceptable set、uncertainty/power protocol 和显式 verdict。若缺少合同字段，结果为 `UNINSTANTIATED`，本模块只输出 measurement profile。

## P0. 定义 `Q`、observable 与 randomness contract

先规定总体和对象，避免用观察到的异常反向定义 error。

## P1. Self-pair calibration

分别估计 reference-reference、compiled-compiled 和 cross-implementation repeats，确认 state 完整性、marginal noise 和 paired covariance。

## P2. Decomposition discovery

在 population states 上比较候选尺度的 relative bias、conditional mean、heterogeneity 和 runtime variability，并检查抵消、增量信息、covariance、重尾、异方差和多峰，不能默认 Gaussian。

## P3. Controlled source perturbations

按机制改变 deterministic order、runtime randomness 或 configuration，观察改变的是：

- mean shift；
- input-conditioned structure；
- marginal variance；
- paired discrepancy variance；
- tail risk。

目的不是把来源预先贴标签，而是用干预判断其统计角色。

## P4. Cross-observable propagation

沿 operator output、model output、gradient、update 比较同一 discrepancy 成分怎样被放大、抵消或旋转。

## P5. Operator attribution

repair/injection 的 endpoint 改为选定尺度上的 bias/variance structure、其 semantic/transition projection 或 tail，而不是 fork 是否消失。

需区分 operator 角色：

- discrepancy-generating operator；
- discrepancy-propagating operator；
- noise-amplifying/suppressing operator；
- boundary-converting operator。

region 只在无法单独干预 operator 时作为独立 treatment 报告；region effect 不得分配给其中某个 operator。

## P6. Downstream impact validation

最后再检查这些分量是否与 discrete event、one-step behavior 和长期训练结果有关。若无关，不否定 discrepancy Oracle，但会限制 training-impact claim。

## 11. Baselines

新的分解 Oracle 必须与以下简单方法比较：

- max/mean absolute error；
- relative error；
- ULP difference；
- global signed mean；
- single-run cross-implementation difference；
- 单一 variance 或 standard deviation；
- tolerance-based pass/fail。

增量价值必须体现在：

- 能区分 state heterogeneity 与 runtime noise；
- 能揭示 mean cancellation；
- 能识别 covariance/coupling；
- 能预测或解释下游 update/impact；
- 能为 operator interventions 提供不同的归因 endpoint。

## 12. 当前 kill criteria

- 分解结果对 `Q` 的合理变化完全不稳定且无法解释；
- repeats 无法区分 state 未冻结与真实 runtime noise；
- 只报均值/方差就足以被重尾和 rare extremes 推翻；
- structured/vector discrepancy 被任意 scalar norm 掩盖；
- decomposition 不比 mean absolute error 提供额外解释力；
- operator interventions 对不同分量没有稳定、可复现结构；
- 没有 truth reference 却将 relative mean shift 宣称为 correctness bias；
- one-step decomposition 被直接外推为长期收敛结论而无动力学假设。
