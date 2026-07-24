# Matched-State Implementation Discrepancy Oracle v3

状态：规范定义。它定义 implementation-relative impact；只有额外 authority 才能实例化 correctness。

> 2026-07-20 scope update：v3 的 B/H/N/U 结构继续有效，但 tensor mean-field norm 是
> continuous-effect magnitude，不自动等于跨训练 step 具有共同意义的 directional bias。
> Qwen3 population bias 的主 endpoint、trajectory sampling 和 confirmation gate 以
> `QWEN3_BIAS_ORACLE_CALIBRATION_PLAN_V0_1_2026-07-20.md` 为准。

## 1. Oracle 不是一个数，而是一条带合同的查询

一次合法查询必须预先声明：

1. `state distribution`：要对哪些训练/推理 states 作结论；
2. `aggregation/conditioning contract`：trajectory、phase、state、token/decision 如何分层加权，哪些 baseline-defined 条件决定成员资格，以及零 exposure state 如何处理；
3. `reference/candidate`：比较哪两个固定 realization；
4. `observable hierarchy`：连续 decision variables、语义事件、一步状态转移中的哪些字段；
5. `randomness protocol`：哪些 RNG/执行随机性固定，哪些开放；
6. `coupling protocol`：两边是否共享输入、RNG draw、调度条件；
7. `acceptance authority`：容忍边界来自规范、下游需求还是仅为 finite-bank exact compatibility；
8. `correctness authority`：高精度 reference、形式规范、verified implementation 或已确认 wrong-code；没有时为空。

不带这些条件的“compiler bias/variance”没有唯一含义。

## 2. 基本观测与四类统计对象

对 implementation `i`、matched state `s`、same-state repeat `r` 和 observable `k`，记录原始观测 `Y(i,s,r,k)`。核心 paired discrepancy 为：

`D(s,r,k) = Y(candidate,s,r,k) - Y(reference,s,r,k)`。

Oracle 报告四个互不替代的对象：

- **B，mean implementation effect**：先在 repeats 上求每个 state 的平均 effect，再严格按预声明的 trajectory/phase/state/decision 层级权重与条件集合求 mean-effect field。相同 states 在不同权重合同下得到的是不同 estimand，不能共享同一个“global B”名称。向量 observable 必须保留 field/direction；mean-field norm 是无符号 effect magnitude。它不能替代预声明的、跨 state 具有共同语义的 directional functional，因此不能仅凭 norm 称为训练 bias。
- **H，state-conditioned heterogeneity**：不同 states 的 mean effect 变化，并在可识别时扣除 repeat noise。H 不是“运行时方差”。只有一个 state 时 H 必须是 `UNIDENTIFIABLE`。
- **N，within-state runtime/replay variability**：相同 state 与协议下 paired effect 在 repeats 间的变化。必须同时说明 reference self、candidate self 和 paired-difference variability。N=0 只对当前协议成立。
- **U，sampling uncertainty**：有限 state/run clusters 造成的总体估计不确定性。U 不能混入 H/N；tensor coordinates 和 tokens 不能冒充独立 state clusters。

这里的 B 是 implementation-relative effect。没有 truth authority 时，不称 trueness bias 或 correctness error。

## 3. 三层 endpoint ledger

### L1：continuous discrepancy profile

可实例化在 operator output、module output、decision variable、loss、gradient 或 update field。它回答差异的方向、大小、state dependence 和 runtime variability，不单独判断应用语义或正确性。

### L2：semantic-event profile

将同一连续观测映射到预声明离散事件：clip、argmax、top-k、routing、overflow、step skip 等。二元事件至少报告两个方向、directional difference 和 total disagreement；sampling 必须比较选择分布/转移概率，不能把一次 token draw 当成分布结论。

L2 不能替代 L1：事件不变时仍可能有 update effect；L1 非零也不保证跨过任何语义边界。

### L3：matched one-step transition profile

从完全相同的模型、optimizer、scheduler/scaler、RNG 和 minibatch state 出发，比较下一状态中的：

- loss；
- scaled/unscaled/clipped gradients；
- parameter update field；
- parameters/buffers；
- optimizer moments/counters；
- scheduler/scaler；
- discrete transition events。

自由运行的 twin trajectories 不属于 L3 estimator，因为两边当前 state 已不同；它们只能作为 divergence/long-run validation。

## 4. Verdict 必须分 ledger

- `construction`：matched state、identity、coupling 与数据完整性是否有效；
- `impact compatibility`：相对于预声明 boundary，continuous/event/transition 是否相容；
- `population inference`：目标 state distribution 上能否作总体结论；
- `correctness`：是否有独立 truth/spec authority；
- `operator attribution`：是否有 realization-preserving repair/injection intervention；
- `long-run harm`：是否有独立长期 endpoint。

合法 verdict 为 `ACCEPT / REJECT / INDETERMINATE / INVALID / UNINSTANTIATED`。其中：

- `ACCEPT` 需要 effect interval 的上界不超过预声明 tolerance；点估计较小不够；
- `REJECT` 只针对该 ledger 与 query scope；impact REJECT 不是 correctness REJECT；
- 缺 acceptance/correctness authority 时必须 `UNINSTANTIATED`；
- identity 或 coupling gate 失败必须 `INVALID`，不能当成零 effect。

## 5. Operator analysis 的连接

operator 不是 state，也不是统计尺度。operator 是 intervention target。

- repair：在 candidate execution 中把 operator realization 换回 reference，估计该 intervention 对同一 L1/L2/L3 endpoint 的 effect；
- injection：在 reference execution 中引入 candidate realization/discrepancy，估计另一方向的 intervention effect。

只有替换保持 surrounding fusion、layout、inputs 和其他 compiler decisions 不变时，才可接近 operator causal effect。否则只能称 intervention-dependent attribution。需要区分：

- discrepancy-generating operator；
- discrepancy-propagating operator；
- 把连续差异转成边界/状态变化的 operator or decision site。

repair 消除整体 effect 不自动证明单一 root cause；替代原因、高阶交互和 treatment noncompliance 必须单独报告。

## 6. Correctness 与 impact 的边界

默认 eager 只是 baseline。若添加高精度 reference、形式规范或 verified wrong-code witness，同一 measurement 可新增 truth-relative correctness ledger；否则只能报告 reference-relative discrepancy 与 semantic/transition impact。

一个合法但不同的浮点 realization 产生 persistent impact，可能是 reproducibility risk 或 application compatibility failure，不自动是 compiler bug。

## 7. 当前 Qwen 实例给出的反例

在预选 run-B step-29 state 上，fresh history-conditioned Inductor scorer 与 eager scorer：

- 两边 self repeats 都 bitwise stable，paired N 为零；
- scorer difference 传播到 loss、clipped gradient 和 AdamW parameter update；
- exact next state 不同；
- clipping、gradient-clip trigger、AMP skip 等事件没有 disagreement。

因此“没有 fork => 没有训练影响”为假；同样，“有 update difference => 长期有害/incorrect”也不能推出。L1、L2、L3 必须是相关但独立的 ledgers。

该 candidate 是 final-state 上构建的 fresh history-conditioned realization；snapshot 不足以重建原训练的十次 historical compile-event lineage。这个限制属于 treatment scope，不能隐藏。

## 8. 规范完备性的含义

“完备”不表示列举深度学习训练中的每个事件，也不表示一个 endpoint 覆盖所有任务。这里要求的是结构完备：

- 任意新增 operator/decision/update endpoint 都必须落入同一 query contract；
- B/H/N/U 的随机性来源与可识别性明确；
- continuous、semantic、transition、correctness、attribution 与 long-run claims 不互相偷换；
- 缺少 state coverage、repeat、authority 或 treatment identity 时能 fail closed。

跨模型、checkpoint、hardware 和 stochastic law 的经验普遍性仍需要新的 state distributions；它不是定义本身自动保证的。
