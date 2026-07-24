# Qwen3 Bias contributor confirmation gate v0.1

## 1. 目的

本文件规定：在某个训练 endpoint 的 implementation-relative average shift `B` 已由独立
confirmation bank 确认之后，什么证据才允许说某个 operator intervention 对该 `B` 有贡献。

这里的结论不是 correctness、root cause、必要性或充分性。没有独立规范时，eager 仍然只是
baseline。

## 2. 启动条件

只有同时满足以下条件，才能启动 contributor confirmation：

1. query、state distribution、endpoint、reference/candidate、randomness protocol 已冻结；
2. endpoint 在独立 confirmation trajectories 上得到 `REPRODUCIBLE_AVERAGE_SHIFT`；
3. sensitivity analysis 没有否决该 shift；
4. endpoint 不是仅有 magnitude 而没有固定方向的量；
5. 候选生成规则、候选集合和每个 intervention 的精确单位，在查看 contributor-bank 结果前冻结。

如果 endpoint 只表现为 H（state-conditioned effect），可以继续做 conditional attribution，但不能称为
“总体 Bias contributor”。如果 endpoint 的 B 未确认，则 gate 关闭。

## 3. 数据隔离

至少三类、通常四批数据承担不同角色：

- calibration bank：估计尺度、设计精度和发现候选；
- endpoint-confirmation bank：只确认 B 是否在独立 trajectories 上存在；
- contributor-precision pilot：候选和 intervention version 冻结后，只估计 `C_o^repair` 的
  trajectory-level dispersion 与施工可行性；
- contributor-confirmation bank：按已冻结精度设计确认 intervention effect。

最终一批必须与前三种角色在 trajectory seed、data range 和自然训练状态上独立。看过
endpoint-confirmation 结果后选择候选是允许的，但这些候选只能在新的 contributor bank 上获得确认性
结论。不能在同一批 states 上筛选贡献最大的 operator，再把同一批差异当作其置信证据。

候选集合必须先单独写入并哈希一个 candidate-freeze artifact；contributor precision plan 与最终
confirmation manifest 都引用这个 artifact。不能让 precision plan 和最终 manifest 互相引用形成循环，
也不能在 pilot 后静默改写候选列表。

总体 B 的 trajectory variance 不能直接代替 repair contribution 的 variance，因为 `D` 与
`D-D[-o]` 是不同随机变量。若没有独立 contributor pilot，可以使用外部依据冻结保守 trajectory count；
否则必须先运行 contributor-precision pilot，再用它的 dispersion（不能用 mean/sign）冻结 desired
half-width、variance floor、multiplicity-adjusted alpha、resource cap 和最终 trajectory count。pilot
trajectories 不进入最终自由度，也不能据其 effect 大小删选候选。若 pilot 暴露 intervention-integrity
失败，修改 intervention 后必须产生新 version，并重新开始其 precision pilot。

precision 还必须满足 sensitivity 的离散分辨率。exact sign-flip 在 `J` 条 trajectories 下最小双侧 p 值
为 `2/2^J`；例如 `J=8` 时为 `2/256`。若 multiplicity-adjusted alpha 更小，不能保留 8 条轨迹并期待
sensitivity 通过，必须增加 J 或将设计判为不可行。

precision input spec 与 precision plan 是不同 artifact。input spec 在读取 pilot dispersion 前冻结
half-width、variance floor、directional floor、multiplicity、tail 和 resource cap；planner 只读取 pilot
trajectory variance，不读取 mean/sign，并输出 plan。最终 validator 会从冻结 spec、candidate artifact
和 pilot summary 重新计算 plan，不能接受手工填写的 `valid=true`。

## 4. Repair estimand

对冻结 endpoint `Y` 和 intervention unit `o`，在同一个 matched state、同一个 transition repeat 下
运行三条配对路径：

- `R`：reference；
- `C`：完整 candidate；
- `C[-o]`：只施加已冻结 repair 的 candidate。

定义完整实现差异 `D = Y_C - Y_R`，repair 后差异 `D[-o] = Y_C[-o] - Y_R`。目标贡献量是

`C_o^repair = E[D - D[-o]] = E[Y_C - Y_C[-o]]`。

期望跨目标 trajectory/state distribution 计算；同状态 repeats 只用于估计 runtime variability，不能扩充
trajectory 自由度。还必须同时报告 residual bias `B[-o] = E[D[-o]]`、state-conditioned
heterogeneity 和 same-state runtime variability。

统计实现 `bias_oracle_contributor_v0_1.py` 要求每个 trajectory/phase/state/repeat 都有完整
reference、full-candidate、candidate-repair 三臂；任一缺失即 fail closed。它分别形成 baseline、residual
和 contribution profiles，贡献量直接使用 paired `Y_C-Y_C[-o]`，因此不会把两臂独立方差相加，也不会
丢掉 common-random-number covariance。
design manifest 同时冻结 validator、precision planner、profile estimator 和 trajectory sensitivity 的路径
与 SHA256；采集后
修改统计代码必须形成新的 manifest/version，不能沿用原确认性 claim。

令 `q` 为 endpoint-confirmation 阶段冻结的 B 方向（标量时为 `sign(B)`；高维 U2 时为只由
calibration 冻结、在 confirmation 复现的投影方向）。主确认量是线性的 `q*C_o^repair`，而不是先对
每条轨迹取绝对值或 L2。 “有可复现方向贡献”至少要求：

1. contributor bank 上未修复 baseline B 能运输并保持预声明方向；
2. `q*C_o^repair` 的 multiplicity-adjusted interval 高于预声明 contribution floor；
3. intervention-integrity、tail 和 sensitivity gates 全部通过。

每个 candidate 的 desired half-width、directional contribution floor 与 variance floor 必须分别携带
可审计 `threshold_sources`，明确不使用 contributor-pilot mean/sign。应用归因容忍度不能冒充运行方差
下限，exact-zero null 也不能冒充目标精度；只有阈值数值、缺少来源或 source role 错配时 precision plan
均无效。

Candidate freeze 还必须链接完整 intervention-unit universe census。若 coverage mode 是
`EXHAUSTIVE_ELIGIBLE_UNIVERSE`，所有 eligible IDs 必须进入 primary family，才允许“覆盖冻结全集”；
若是 `PREDECLARED_SUBSET_OF_UNIVERSE`，claim scope 只能是 `SELECTED_CANDIDATES_ONLY`。
这一区分 coverage，不改变 source operator、generated kernel 与 region 的 intervention-unit 边界。
所谓冻结全集严格限于 endpoint-confirmation state bank 中 census 实际观察到的 units：coverage evidence
必须逐 state 完整，state IDs/order 与 confirmation evidence 完全一致，observed-unit union 等于 universe，
并绑定 raw census artifact hashes。它不支持“覆盖所有可能模型状态/程序运行”的外推。

`|B[-o]| < |B|` 不是仅凭两个点估计就能成立的附加条件。如果需要声称“绝对 Bias 降低”，必须用
预声明的 joint/simultaneous uncertainty rule 证明 residual 与 baseline 方向及差值；否则只描述
`B[-o]`。若 repair 沿原 B 方向移除了分量但越过零并产生更大的反向 residual，应报告
`DIRECTIONAL_CONTRIBUTOR_WITH_OVERSHOOT`，不能称为 Bias reducer。

这估计的是“该 repair 在该 query 与实现上下文中的平均贡献”。即使 residual 变成零，也不自动证明
`o` 是唯一 root cause 或数学错误来源。

## 5. Injection estimand

若存在保持 reference 其余上下文不变的有效 injection，定义 `R[+o]`，并估计

`C_o^inject = E[Y_R[+o] - Y_R]`。

Injection 回答的是：候选 discrepancy 在 reference context 中被单独引入时，能否产生相同 endpoint
方向。Repair 回答的是：在 candidate context 中移除该 intervention 后，既有 B 改变多少。

二者可能不对称，原因包括非线性传播、上游输入不同、误差抵消、阈值转换和高阶交互。repair 非零而
injection 为零，或反过来，都不是逻辑矛盾。缺少 injection 时可以报告 repair contributor，但不能使用
充分性语言；即使 injection 复现全部 B，也只支持冻结上下文中的生成能力，不支持跨状态的无条件充分性。

## 6. Operator 与 generated region 的边界

claim unit 必须等于实际 intervention unit：

- 若只替换一个可识别 operator invocation，并能证明 fusion、layout、调用关系和其他 compiler choices
  未变，才可称 `operator intervention effect`；
- 若替换的是一个 fused callable、kernel 或 generated region，结论只能归于该 callable、kernel 或
  region；不能把贡献分摊给其中的 `sum`、`cast`、`rsqrt` 等 constituent operators；
- 若 graph break 或 replacement 改变了周边 fusion/layout/schedule，结论降级为
  `intervention-package effect`。

因此 operator 就是 operator，region 就是 region；二者不合并成同一个分析单位。

此外，候选 ID 不能只是 `sum`、`mm` 或某个 kernel family 名。每个 candidate 必须冻结 unit-specific
structural identity，并链接覆盖 confirmation state bank 的 state-realization map：每个 state 逐一列出
exact call IDs、expected count 与 call-order digest，或显式验证该 unit 未出现。map 缺 state、把不同
callsite 合并到一个 family ID、或 unit type 与 identity semantics 不一致时，operator claim 无效。

每个 intervention 至少要验证 pre-state identity、baseline anchor、精确调用身份与次数、恰好一次预期
替换、无意外重新编译、未干预部分的 realization identity、恢复后 anchor、sham control 和完整 endpoint
有效性。任一关键 gate 失败时结果是 `INVALID_INTERVENTION`，不是 null effect。

## 7. 候选、多重比较与交互

候选集合可以由结构 census、calibration discrepancy 或旧 selected-state pilots 产生，但它们只是 discovery
证据。冻结时必须列出全部 primary candidates；不能在 contributor bank 上只报告最显著者。

候选 universe 必须先选择且只选择一种分析单位（source operator invocation、generated-kernel invocation、
generated region 或明确声明的 intervention subspace），并哈希绑定产生 endpoint confirmation 的冻结
state bank。scope kind 与实际 intervention unit 不一致、或归因时换用另一个 state bank，均使设计无效。
若声称 `EXHAUSTIVE_ELIGIBLE_UNIVERSE`，还必须有独立的逐-state census/coverage evidence 证明列举
完整；仅在 manifest 中把一个有限列表标成 exhaustive 不是覆盖证据。validator 会核对 confirmation
state census、observed-unit union 与 raw artifact hashes；任一不完整时结论限于 selected candidates。

primary repair family 的 simultaneous error control 必须预声明，例如 Bonferroni simultaneous intervals
或预先验证的 max-statistic procedure。Injection 若承担确认性 claim，应作为独立 hypothesis family 冻结。
统计自由度来自独立 trajectories，而不是 states、tokens、tensor coordinates 或 repeats。

singleton repair 默认不具有可加性。若 `a`、`b` 都进入 pair audit，定义

`Gamma_ab = C_{a,b}^repair - C_a^repair - C_b^repair`。

非零 `Gamma_ab` 表示贡献依赖 coalition/context。pair/coalition 必须在查看其结果前冻结，或在新的 heldout
bank 上确认，并纳入相应 multiplicity family。没有足够 interventions 时，应报告“interaction unresolved”，
而不是强行把总 Bias 分解到每个 operator。

## 8. 角色描述

为了说明传播链，可以区分三种证据角色，但它们不是互斥的 operator 类型：

- discrepancy generation：局部候选实现首先产生可测差异；
- discrepancy propagation：上游差异通过该计算被放大、旋转、抵消或重新分布；
- boundary conversion：连续差异在此改变离散事件或更新控制流。

这些角色需要不同的 intermediate taps、repair/injection 和 semantic endpoints 支持。只有 full-step repair
不能独自区分三者。

## 9. Verdict 与允许的表述

- `CONFIRMED_DIRECTIONAL_INTERVENTION_CONTRIBUTOR`：baseline B 可运输，沿冻结方向的贡献 interval
  通过，所有 validity/sensitivity gates 通过；
- `CONFIRMED_ABSOLUTE_BIAS_REDUCER`：除上一项外，预声明的 simultaneous residual rule 也证明绝对
  Bias 降低且无 overshoot；
- `DIRECTIONAL_CONTRIBUTOR_WITH_OVERSHOOT`：方向贡献成立，但 repair 产生更大的反向 residual；
- `STATE_CONDITIONAL_CONTRIBUTION_ONLY`：存在可复现 conditional effect，但总体平均贡献未确认；
- `NO_DETECTED_CONTRIBUTION_UNDER_POWER_CONTRACT`：有效且达到预声明精度，但未检出贡献；不等于无贡献；
- `INTERACTION_DEPENDENT`：singleton 结论随冻结 coalition 明显变化；
- `INDETERMINATE`：精度、tail、transport 或 method sensitivity 不足；
- `INVALID_INTERVENTION`：干预没有隔离到声明单位或执行合同失败；
- `UNINSTANTIATED`：候选、bank、阈值或 multiplicity 尚未冻结。

允许的最强表述是：

> 在已声明的 state distribution、endpoint、randomness protocol 和 repair intervention 下，该单位对
> implementation-relative average shift 有可复现贡献。

除非另有独立 specification 与更强识别设计，不使用 compiler bug、correctness error、唯一 root cause、
必要原因或充分原因等表述。

## 10. 当前项目状态

现有 SiLU、cast、RMSNorm 和 generated-kernel repairs 是 intervention mechanics、null control、state
dependence 和候选发现证据。它们没有使用独立 contributor-confirmation bank，因此不能升级为 population
Bias contributor。

当前执行顺序不变：先完成四条 calibration trajectories，再冻结 endpoint confirmation；只有某个 endpoint
的 B 在独立 confirmation bank 上成立，才实例化本 gate 的 manifest 和新的 contributor bank。

实例化顺序为：candidate-freeze artifact → contributor precision plan → contributor-confirmation manifest。
模板分别为 `QWEN3_BIAS_CONTRIBUTOR_CANDIDATE_FREEZE_TEMPLATE_V0_1.json`、
`QWEN3_BIAS_CONTRIBUTOR_PRECISION_INPUT_SPEC_TEMPLATE_V0_1.json`、
`QWEN3_BIAS_CONTRIBUTOR_PRECISION_PLAN_TEMPLATE_V0_1.json` 和
`QWEN3_BIAS_CONTRIBUTOR_CONFIRMATION_MANIFEST_TEMPLATE_V0_1.json`。
