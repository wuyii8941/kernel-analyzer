# Bias Oracle confirmation decision mode v0.1

状态：**理论决策，尚未完成 machine-readable spec/code migration，不能启动 confirmation**。

## 1. 需要纠正的概念绑定

当前 precision template 把下面三件事绑定得过紧：

1. `shift existence`：目标分布上的平均 implementation-relative effect 是否与声明的 null/floor 分开；
2. `estimation precision`：区间是否窄于一个事前目标；
3. `practical materiality`：effect 是否大到足以影响应用。

只有第一项是“是否存在 B”的必要判定。第二项只有在存在独立、可辩护的目标半宽时才能成为 gate；
第三项需要外部应用容忍度。不能为了让 confirmation planner 产生样本量而用 calibration mean 的比例、
资源预算或任意常数伪造后两项。

一个直接反例是：某 endpoint 的 adjusted interval 很宽，但完整位于零的一侧，trajectory signs 和
sensitivity 也一致。它已支持“存在有方向的平均 shift”，虽然 effect 大小估计不够精细。此时完全禁止
operator attribution 不是统计必然要求；contributor study 可以另行冻结自身的精度。反过来，一个很窄、
覆盖零的区间只证明在该宽度下未检出 shift，并不因为“精度达标”就存在 B。

## 2. 两种合法 confirmation mode

### Mode E：fixed-resource existence confirmation（本项目第一版推荐）

在读取 confirmation outcome 前冻结：

- endpoint/global/phase family 与 multiplicity；
- exact-zero 或独立 null-control 支持的 `shift_existence_floor`；
- independent trajectory 数 `J_fixed`、seeds/data/state bank；
- trajectory-level interval、studentized sign-flip sensitivity 和 fail-closed construction；
- resource cap 和停止规则。

`J_fixed` 由可执行资源上限、minimum trajectory count 和 multiplicity-adjusted sign-flip p-value 分辨率
共同确定，不声称达到某个没有来源的统计 power/half-width。最终输出完整 adjusted interval 和实际
half-width：

- interval 完全位于 `[-floor,+floor]` 外侧且 sensitivity 不冲突：
  `REPRODUCIBLE_AVERAGE_SHIFT`；
- 否则：`NO_STABLE_AVERAGE_DETECTED_AT_FIXED_RESOURCE` 或 method/construction indeterminate；
- 未检出不得改写为 B=0、practical equivalence 或实现等价。

Mode E 的优点是没有虚构 effect-size threshold；代价是可能 power 不足，未检出结果通常较弱。

### Mode P：precision-targeted confirmation（有独立精度依据时使用）

只有 endpoint 拥有不依赖 candidate mean/sign 的 `desired_half_width` 和正 variance floor 时才启用。
planner 使用四条 calibration trajectory dispersion 的单侧上界，事前计算 `J`。confirmation 后另检验
实际半宽是否达到目标；未达到时保留 shift-existence 票，但 precision 票为 indeterminate。

Mode P 可以支持“以预先要求的精度回答了问题”，但 measurement resolution、应用 tolerance 和最小有意义
effect 仍是不同来源，不能互换。

## 3. 与 operator attribution 的关系

进入 Bias contributor study 的最小统计门槛应是：

- construction/identity/independence gates 全部有效；
- 某个冻结 global 或 phase endpoint 得到 `REPRODUCIBLE_AVERAGE_SHIFT`；
- direction 对 multiplicity-adjusted interval 和 sensitivity 稳定；
- attribution state distribution 与已确认 claim 完全相同。

在 Mode E 中，不要求一个虚构的 desired half-width；repair/injection contribution 自己需要独立 pilot、
precision plan 和 confirmation。此时 operator 结论仍只是“对已确认 B 的 intervention-dependent
contribution”，不是 root cause、correctness 或长期危害。

若 operator attribution 的科研问题要求精确回答“贡献了 B 的多少比例”，则必须为 contribution endpoint
进入 Mode P；不能借用总体 B 的区间宽度。

## 4. 与 null controls 的关系

reference/reference、candidate/candidate 和 identical-post-state evaluator controls 可以：

- 发现 acquisition/runtime/evaluator pipeline 自身的非零差异；
- 支持 exact-zero null 是否可信，或给出非零 measurement envelope；
- 暴露 R=2 下未识别的 runtime variability 边界。

它们不能自动给出 desired half-width、应用危害阈值或长期训练 tolerance。若四轨迹 null contrasts 全为零，
Mode E 可以使用 `EXACT_ZERO_NULL`，同时明确“未观察到 measurement variability”；Mode P 仍需要额外正的
目标宽度与 variance floor。

## 5. 推荐决策

本项目当前目的是先确认 implementation-relative Bias 是否存在，并为后续 operator attribution 提供有方向
的 endpoint，而不是证明某个应用容忍度内等价。因此第一版推荐：

> global/预声明 phase claims 使用 Mode E；所有区间宽度完整报告，但不因缺少外部 precision target
> 发明半宽。practical materiality 保持 `UNINSTANTIATED`。若之后得到外部科学精度需求，再为相应
> endpoint 单独升级为 Mode P。

现有 `QWEN3_CONFIRMATION_PRECISION_SPEC_TEMPLATE_V0_1.json` 和 planner 只实现了 Mode P。迁移完成并通过
fail-closed tests 前，confirmation 仍保持 `UNINSTANTIATED_DO_NOT_RUN_CONFIRMATION`。
