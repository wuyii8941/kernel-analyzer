# Theoretical Audit：Error / Mean Discrepancy / Variability

## 1. 审查结论

`state distribution + reference + observable + randomness protocol` 是必要条件，但不足以完整定义 error/bias/variance Oracle。

当前还必须补充：

1. intended mathematical problem 或 relational contract；
2. error geometry/scale；
3. complete conditional discrepancy law 与 tails；
4. configuration 和时间层次；
5. numerical conditioning/stability；
6. acceptance loss/tolerance；
7. identification assumptions。

## 2. 五个合适但边界不同的理论领域

### 2.1 Numerical analysis

提供：

- forward error；
- backward error；
- conditioning；
- algorithmic stability；
- roundoff propagation。

它回答 numerical discrepancy 是问题本身敏感，还是实现不稳定。仅比较 `Y_C-Y_R` 属于 implementation-relative forward discrepancy，无法单独区分 conditioning 与 implementation quality。

### 2.2 Metrology and method comparison

提供：

- measurand/reference value；
- repeatability/reproducibility；
- method agreement；
- uncertainty budget。

它约束 bias/trueness 术语：没有独立 reference quantity value 时只能讨论 method difference/mean discrepancy。

### 2.3 Hierarchical statistics and design of experiments

提供：

- state、configuration、trajectory、repeat 的层级设计；
- conditional mean/variance；
- effect heterogeneity；
- sampling uncertainty；
- covariance 和 interaction。

它提供估计方法，不自动定义 correctness 或 practical relevance。

### 2.4 Probability and uncertainty quantification

提供：

- complete conditional laws；
- tails、quantiles、rare-event risk；
- distribution propagation；
- dependence/coupling；
- multivariate covariance/geometry。

它防止将 mean/variance 偷换成完整误差分布。

### 2.5 Software test-oracle theory

提供：

- specification 与 expected behavior；
- partial/relational oracles；
- differential/metamorphic relations；
- verdict、soundness、false positives 和 coverage。

统计上检测到 implementation difference 仍只是 measurement；只有加入 acceptance relation 才形成 operational Oracle。

### 2.6 Causal inference/interventional debugging（后续）

提供 operator repair/injection 的干预语义。它不参与最初 error decomposition，但决定能否把分量变化归因于算子。

## 3. 当前定义的主要遗漏

### 3.1 Forward discrepancy 不等于完整 numerical error

`Y_C-Y_R` 只说明两个实现输出差多少。还需区分：

- forward discrepancy：输出空间差异；
- backward error：该结果能否解释为对输入/state 的小扰动；
- conditioning：数学问题对输入扰动本来有多敏感；
- stability：实现是否额外放大扰动。

若问题本身病态，微小实现扰动产生大输出差异不自动表示实现错误；若问题条件良好，大差异则更可疑。

### 3.2 Mean/variance 不等于完整 distribution

相同 mean/variance 的分布可以有完全不同的：

- tail probability；
- skewness；
- multimodality；
- rare catastrophic values；
- dependence structure。

因此 primitive estimand 应是 `Law(D | S=s, configuration)`；moment decomposition 是摘要。

### 3.3 Vector/tensor error 没有天然 scalar

使用 mean、norm、max error 或 task-direction projection 会产生不同结论。必须声明 error geometry：

- coordinatewise；
- absolute/relative/ULP；
- norm；
- angular/directional；
- task-relevant projection；
- structured covariance。

否则“bias 接近零”可能只是坐标抵消。

### 3.4 Coupling 改变 paired discrepancy variance

两实现各自的 marginal law 可以相同，但不同 coupling 下 `Y_C-Y_R` 的 variance 不同。必须分开：

- implementation marginal variability；
- coupled paired-difference variability；
- actual deployment coupling。

### 3.5 Configuration mixture 不等于 runtime noise

hardware、compiler flags、autotuning variant、kernel selection 和 schedule 可能有嵌套层次。固定后是条件实现；重新抽取时才贡献相应层次的 variability。

### 3.6 时间依赖和非平稳性

训练 states 沿轨迹相关，且 early/mid/late training 的 discrepancy law 可能不同。单个 `Q` 的平均可能掩盖时间分段或 regime change。

### 3.7 Reference 自身有不确定性

eager 可能随机、数值不稳定或有 bug。相对差异需要同时报告 reference marginal law；有第三方 truth 时才可校准两边 accuracy。

### 3.8 Identification 不等于 estimation

即使统计 estimator 正确，如果 state 未完整冻结、randomness source 混合或 configuration 隐藏，分量仍不具有预期因果/机制含义。

### 3.9 Statistical difference 不等于 Oracle failure

需要预先定义：

- equivalence tolerance；
- maximum permissible error；
- tail-risk threshold；
- application loss；
- specification relation。

否则只能输出 discrepancy characterization，不能输出 correctness PASS/FAIL。

### 3.10 Local one-step law 不等于长期 law

自由训练改变 state visitation。长期影响需要动力学稳定性、覆盖和误差传播假设，不能由局部 mean/variance 直接推出。

## 4. 最小完整比较契约

建议每个 formal claim 至少绑定下面九项：

1. **Target relation**：预期保持的数学或实现关系；
2. **Reference status**：baseline、high precision 还是 specification；
3. **State population `Q`**：结论针对哪些 states；
4. **Observable `Y`**：比较什么层级的对象；
5. **Error geometry `ρ`**：差异如何度量或投影；
6. **Randomness law `Π`**：哪些随机性存在及其分布；
7. **Coupling `κ`**：两边 randomness 如何配对；
8. **Configuration hierarchy `C`**：哪些 compiler/hardware choices 固定或随机；
9. **Acceptance functional `L/τ`**：什么差异被认为不可接受。

只具备前八项时得到 measurement model；九项齐全时才能形成 operational relational Oracle。若 target relation 连接到独立真值，才可能进一步成为 correctness Oracle。

## 5. 推荐的 primitive object

不是先定义三个 scalar，而是先定义：

```text
Lawρ(D | S=s, configuration=c; randomness law Π, coupling κ)
```

再从中导出：

- conditional mean discrepancy；
- average mean discrepancy under `Q`；
- state heterogeneity；
- marginal and paired runtime variability；
- covariance/dependence；
- quantiles/tail risk；
- forward/backward/stability diagnostics；
- acceptance violation probability。

## 6. 最需要避免的概念偷换

- discrepancy → mathematical error；
- mean discrepancy → truth-relative bias；
- state heterogeneity → runtime variance；
- estimator uncertainty → program randomness；
- mean/variance → complete distribution；
- paired variance → implementation marginal variance；
- large forward error → unstable implementation；
- statistical significance → practical failure；
- difference detection → correctness Oracle；
- local transition difference → long-run convergence failure；
- intervention association → operator root cause。

## 7. 当前更严谨的核心表述

> 在明确 target relation、reference status、state population、observable、error geometry、randomness law、coupling 和 configuration hierarchy的条件下，刻画两实现 discrepancy 的完整条件分布；将 mean discrepancy、state heterogeneity、runtime variability 和 tail behavior作为派生摘要；在加入独立 specification 或 acceptance functional 后，才将该 characterization 升级为 operational Oracle。

