# Qwen3 boundary-conditional B confirmation contract v0.1

## 目的

这个合同只修正一个问题：global average B 可能因 state effect 正负抵消而漏掉与语义边界对齐的
implementation effect。它不把任意高 H、事后发现的 subgroup 或一次 fork 升级为 Bias。

## 条件变量必须来自 reference/pre-intervention 一侧

对 advantage 非零且 completion mask 有效的 token，定义 signed clipping margin：

```text
M_R = sign(A) * (exp(logp_R - old_logp) - 1) - epsilon
M_C = sign(A) * (exp(logp_C - old_logp) - 1) - epsilon
```

`M > 0` 当且仅当该实现触发 GRPO/PPO clipping。条件集合只能用 `|M_R| <= tau` 定义；不得用
compiled margin、两实现 delta、是否发生 fork、update 大小或 operator observation 定义。reference 在这里
只是 baseline anchor，不是真值。

v0.1 还要求 reference scorer log-probs 在两个 fresh-process repeats 中完全一致，才允许用 repeat-1 的
margin 固定 condition mask。否则，同时用 repeat-1 的 runtime noise 选择 near-boundary tokens 并参与
effect 计算会形成 selection-on-noise。若 reference repeats 不一致，本版本必须返回
`UNINSTANTIATED_STOCHASTIC_REFERENCE_ANCHOR`，不能继续给 conditional B；后续版本需要额外的、与两个
effect repeats 独立的 reference anchor execution。两次 exact 只是对“reference conditional output 是点质量”
这一识别假设的诊断，不是数学证明；该假设及 fail-closed fallback 必须在 confirmation spec 中事前冻结。
candidate repeat 是否变化仍由 paired N 描述。

## 三个必须分开的量

对每个 matched state，先在其 near-boundary eligible tokens 内计算：

1. signed continuous effect：`mean(M_C - M_R)`；
2. signed semantic effect：`mean(I[M_C>0] - I[M_R>0])`；
3. semantic disagreement：`mean(I[(M_C>0) != (M_R>0)])`。

前两个分别属于 conditional signed numerical role 和 signed semantic role。第三个是非负 impact
role，不得叫 B，也不得作为“只找 Bias contributor”的替代 endpoint。三类 endpoint 可在同一
prospectively frozen boundary family 中共享 multiplicity correction，但不共享 estimand 或 verdict 名称。

## 层级权重

conditional estimand 保持主 Oracle 的层次：

```text
equal trajectory
  -> equal early/middle/late phase
    -> equal states with at least one reference-near-boundary eligible token
      -> equal eligible near-boundary tokens within state
```

若任一冻结 phase 少于两个 exposed states，该 trajectory/tau 的 conditional profile 未识别；不得删除该
phase、不得改成 exposure-pooled token mean，也不得用其他 phase 补齐。要求两个 states 是为了避免把单
state 的 state dispersion 机械记为 0，不是 practical-effect threshold。

这个 estimand 的目标分布是 baseline-defined conditional distribution，不等于原始 global state
distribution。更精确地说，它是
`EQUAL_TRAJECTORY_PHASE_EXPOSED_STATE_NEAR_TOKEN_V0_1` 定义的层级、state-balanced conditional
estimand，不是把全部 near-boundary tokens exposure-pooled 后得到的普通 token mean。它必须始终以
“该固定层级权重下的 `B | |M_R| <= tau`”身份报告，不能写成无条件 compiler Bias；若要研究
exposure-weighted token population，必须另立 estimand/family，不能静默更换权重。

为避免它与原 all-eligible/global diagnostic 混淆，后者另有机器可读权重合同
`EQUAL_TRAJECTORY_PHASE_SAMPLED_STATE_ELIGIBLE_DECISION_V0_1`：每条 trajectory、phase、sampled state
等权，再在 state 内平均 eligible decisions；任一 sampled state 无 eligible decisions 时该合同未实例化。
boundary conditional 仍使用本节的 exposed-state 合同，二者不能互换。

reference-anchor exactness 也不能在四轨迹聚合时丢失：multi-trajectory summary 必须证明四条 trajectory
各自 24/24 states 的 reference scorer repeats 完全一致，并保留 96-state audit；family freezer 会再次验证。
任一 state 不 exact 时，即使该 tau 的 exposure support 已足够，也不得冻结正式 conditional family。

## tau family 的 calibration 与冻结规则

当前 calibration-only 候选网格是：

```text
0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05
```

它是在 calibration-0 已有部分 records 后加入的 exploratory family，不声称在整个 calibration 前盲设；
但 acquisition states 原本已由独立 capture plan 固定，没有按 tau、fork 或 delta 选择。

四条 calibration trajectories 完成后：

- 只允许根据 reference-side support 判断某个 tau 是否具有可定义性；
- 一个 tau 只有在每条 trajectory 的每个 phase 至少有两个 exposed states 时才可保留；
- 不允许根据 observed margin-shift 大小、符号、fork rate 或 task/update effect 挑 tau；
- 所有 support-complete tau 必须一起进入 confirmation endpoint family 和 multiplicity；
- 若没有 tau support-complete，则本版本 boundary-conditional family 为 `UNINSTANTIATED_SUPPORT`；若扩大
  网格，必须建立新版本，不能静默修改 v0.1。

这条规则只提供最低可识别性，不声称两个 exposed states 已经足够稳定；它区分“profile 能否构造”和
“估计是否精确”。
精度由 independent trajectory interval 完整报告；Mode E 不虚构 desired width，Mode P 仍需要独立精度来源。

## 独立 confirmation gate

每个保留 tau 都是一个明确 endpoint。只有满足以下条件才可进入对应 operator repair/injection：

1. tau family、公式、权重、shift floor、multiplicity、trajectory bank 和代码哈希在 confirmation outcome
   产生前冻结；
2. confirmation 使用 calibration 未出现过的 trajectory/data/RNG seeds；
3. 每个 trajectory×phase 至少有两个 reference-side exposed states，否则该 endpoint fail closed；
4. signed conditional effect 的 simultaneous interval 位于 shift floor 同一侧；
5. trajectory-level sign-flip sensitivity 不否决主结论；
6. fixed-resource 或 precision-targeted 模式产生其事前承诺的 interval report。

通过后只能声称：在该 baseline-defined conditional distribution 上存在 implementation-relative signed
effect。它不证明 eager 正确、不证明长期训练受害，也不证明某个 operator 是 unrestricted root cause。

## 与 operator analysis 的连接

repair/injection 必须使用同一个 tau、同一个 reference-side condition 和同一个层级权重。算子干预后不得
重新按 repaired/compiled margin 选择 states 或 tokens，否则 intervention 会改变分析集合，产生
post-treatment selection。

权重合同 ID 必须由 calibration summary 逐级传到 support family、confirmation result、attribution bridge
和 contributor record bank；任一环节缺失或变化均 fail closed。这样“global B 抵消后的 conditional B”
不会再次因 exposure 较多的 states 获得更大隐含权重而改变含义。

归因结果只能写成“candidate intervention 对已确认 conditional estimand 的贡献”。若 repair 改变 fusion、
layout、kernel family 或其他非局部 realization，则只能称 intervention-dependent attribution。

还必须区分两种 state bank。直接复用 endpoint-confirmation 中的 exposed states/masks，只能做
same-bank mechanism pilot：它回答“在这些已经确认 endpoint 的 states 上，干预会怎样改变效应”，不能
推广成目标分布上的 population operator contribution。正式贡献确认需要另一组独立 trajectories；其
trajectory ID、seed 与 data slice 必须同时避开 calibration 和 endpoint confirmation。新 bank 先仅运行
reference anchor、冻结 masks 和完整 8×3 state census，确认每个 trajectory×phase 至少两个 exposed
states，随后才允许产生任何 repair/injection outcome。

## 当前状态

- calibration-0 partial diagnostic 已能逐 token 重建并验证 eager/compiled clip decisions；
- 当前 20/20 个完整 states 的 reference scorer log-probs 在两个 repeats 中完全一致，因此 exploratory
  repeat-1 anchor 在这些 states 上没有观察到 selection-on-noise；这不证明其他配置的 reference N 为零；
- 当前只有 early 8 states、middle 8 states、late 4 states 完成；`tau=0.05` 已在三个 phase 各有至少两个
  exposed states，因而该 partial trajectory 内 conditional estimand 已定义，其他 tau 仍可能缺少 phase support；
- partial leave-one-state-out influence diagnostic 显示 early 的 all-eligible phase mean 可因删除一个 state 而翻转符号；
  middle mean 在加入 step-151 后暂时对删除任一单 state 保持正向。该产出只描述 finite calibration-bank influence；少于三个 exposed states 的
  conditional LOO stability 明确不可解释，任何 LOO 结果都不能替代 independent-trajectory confirmation；
- core-endpoint LOO diagnostic 进一步显示 endpoint-specific 结构：training loss 的 early 对单 state 删除不稳、
  middle 暂时稳定为负；U1 early 不稳而 middle 暂时稳定为负；T1a 则在完整 early/middle 各八个 states 上
  分别保持正向和负向，形成
  global mean 可能抵消的真实 phase-conditional pattern。它只支持保留预声明 conditional family 的设计，
  不在 calibration-0 或 late phase 缺失时确认 conditional B；
- step-151 产生两个确定性 clip off→on；它们在 reference-defined `tau=0.05` 中可见，而该 state
  在 `tau<=0.01` 中没有 exposure。加入该 state 后，已观测 states 的 `tau=0.05` 净 directional mean 恰好为零，
  但 semantic disagreement 仍非零。这是校准数据中实际观察到的“净事件变化抵消，但行为不一致仍存在”，
  不是 population semantic-effect confirmation；
- step-153 的 all-eligible continuous margin shift 为正，但其在所有 `tau<=0.05` 中均无 reference exposure，
  也无 semantic event change。它改变 middle all-eligible mean 而不改变任何 boundary profile，因此这两类
  estimand 不得相互代替；
- step-166 的 all-eligible continuous margin shift 为正，但在 `tau=0.05` 的四个 reference-exposed tokens
  上 conditional mean margin shift 为负，并且 semantic disagreement 为零。这进一步表明 all-token
  shift、boundary-conditional shift 和 semantic effect 是三个不同对象；任一对象都不能由另一个代理；
- step-204 的完整 record 在 transition、task、update 与 semantic endpoints 上均为 eager/compiled 精确
  相同，但该 state 没有 eligible clipping decisions。其 all-eligible clipping-margin mean 因而是
  `UNINSTANTIATED_NO_ELIGIBLE_DECISIONS`，不是零。聚合器保留该 state 并令相应 state-balanced phase/global
  estimand fail closed，不允许 NaN、零填充或 complete-case 删除；boundary conditional 仍按 exposed-state
  合同独立判断 support；
- step-222 有 512 个 eligible decisions、all-eligible margin shift 为负，但所有预设 `tau<=0.05` 均无
  reference-near-boundary exposure，也无 semantic disagreement；其 T1a 为正，而 T1b、training loss 和
  update-aligned effect 为负。该 state 再次表明 continuous、task、update 与 boundary/semantic estimands
  方向不必一致；同时 late phase 仍没有 boundary support，且因 step-204 endpoint 未实例化，原 state-balanced
  late/global eligible-conditioned mean 不得 complete-case 计算；
- step-224 有 512 个 eligible decisions，all-eligible margin shift 为负；它首次为 late phase 在预声明
  `tau=0.05` 提供 boundary support，三个 reference-near-boundary decisions 上 conditional margin shift
  也为负，但 directional event shift 与 disagreement 均为零。一个 exposed state 不满足每 phase 至少两个
  exposed states 的 support 合同，因此这只是新增 conditional support，不是 conditional-B verdict。该 state
  的 training loss、clipped-gradient alignment、update alignment、T1a 与 T1b 的方向也不一致，并在没有
  event change 时产生确定性的 next-update difference；三层 endpoint 是并列 ledger，不是必然传播链；
- step-227 的 all-eligible margin shift 为正，但 `tau=0.05` 的五个 reference-exposed decisions 上
  conditional margin shift 为负，并确定性地产生一个 `on→off` event；其 loss/clipped-gradient 与
  update/T1a/T1b 方向也分裂。该 state 使 calibration-0 的 `tau=0.05` 首次满足三个 phase 的 support
  合同，但这仍只是 calibration construction，不允许形成 population conditional-B 或 semantic verdict；
- 独立 boundary confirmation evaluator 与 spec template 已实现：它重新验证基础 confirmation manifest、
  source audits、24-state census、family/spec/analysis hashes，使用 simultaneous interval 与 trajectory
  sign-flip sensitivity，并分别输出 continuous conditional shift、signed semantic shift 和不含 B 命名的
  nonnegative semantic disagreement；
- resource planner 在 bank 冻结前只用 family comparison count、family alpha、minimum J 与 resource cap
  计算 sign-flip 离散 p-value resolution；不读取 calibration effect mean/sign/variance。最终 spec 必须绑定
  该 resource plan 和实际 confirmation manifest；
- 当前四轨迹 calibration、support-complete family manifest 和实际 confirmation bank 均未完成，因此 evaluator
  尚未实例化运行，boundary diagnostic 仍不能触发 operator analysis；
- global/phase contributor validator 仍不接受 boundary identity；boundary endpoint 必须通过专用
  attribution-design gate，不能因 evaluator 给出 ready endpoint 而绕过。
- write-once boundary attribution bridge 已实现：它只接受 independently confirmed 且 operator-ready 的
  endpoint，冻结 exposed states 与 reference-anchor mask hashes；它本身不选择 operator，不声称 root cause。
- boundary contributor record-integrity gate 已实现：repair/injection 三个 arms 必须逐 state/repeat 复用
  bridge 中同一 reference mask 和 cardinality，且不得加入 bridge 外 state；它分别构造
  signed endpoint 的 `candidate-intervention` 与 `intervention-reference`。对 disagreement，validator 强制使用关系语义：
  `candidate_value=D(reference,candidate)`、`intervention_value=D(reference,intervention)`，且 `reference_value=0`；
  不允许把 pairwise disagreement 伪装成 arm-specific scalar。当前 bridge 明确仅支持复用 endpoint-confirmation
  bank 的 mechanism pilot，不自行给出贡献 verdict，也不允许 population operator-contribution claim。
- 独立 contributor-bank freezer 已实现：它沿 bridge 追溯 endpoint-confirmation manifest 与 calibration
  exclusions，拒绝复用 trajectory/seed/data slice；要求至少八条新 trajectories、每条完整 24-state 8×3
  census、事前冻结的 reference masks、exact-reference-repeat gate 和每个 trajectory×phase 至少两个
  exposed states。通过只表示可以开始独立 intervention measurement，仍不等于算子贡献已确认。
- 独立 repair/injection record validator 与 trajectory-level evaluator 已实现。record validator 要求
  candidate/intervention family 在 outcome 前冻结，并逐 state/repeat 复用独立 bank masks。evaluator 先在
  新 bank 上检验原 `candidate-reference` conditional effect 是否沿 endpoint-confirmation 冻结的方向
  transport；只有 baseline transport 与 `candidate-intervention`（repair）或
  `intervention-reference`（injection）贡献都通过 multiplicity-adjusted trajectory interval、sign-flip
  sensitivity 和 fixed-resource precision reporting，才允许 intervention-dependent population contribution
  claim。稳定干预响应但 baseline 未 transport 时必须保持 `INDETERMINATE_BASELINE_DID_NOT_TRANSPORT`。
- primary operator estimand 是 absolute directional intervention contrast，不是“解释比例”。repair 另行输出
  `repair-reference` residual 与 `|candidate-reference|-|repair-reference|` absolute reduction，默认仅作
  descriptive diagnostics。若 repair 从正侧越过零点到同等负侧，directional contribution 可以显著而
  absolute reduction 为零；因此不得自动写成“解释了 200% 的 Bias”。任何 explained fraction 都需要单独
  预声明的 ratio estimand 与 near-zero denominator inference，本版本保持 `UNINSTANTIATED`。
- 合成的端到端 routing 反例已经覆盖 global B 抵消、conditional B 稳定非零的情形：只有具体的
  boundary-conditional endpoint 获得 bridge，global endpoint 与 nonnegative disagreement 均不能借此进入
  signed-B attribution。
