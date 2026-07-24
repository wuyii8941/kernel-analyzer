# Qwen3 bias Oracle calibration plan v0.1

状态：下一阶段权威计划。它取代“Oracle 已经可以直接进入全算子归因”的旧推进顺序，
但不否定已经完成的 selected-state measurement、transition 和 region-repair 证据。

面向执行与组内沟通的简版入口：`BIAS_ORACLE_NEXT_STEP_ONE_PAGE_CN_2026-07-20.md`。

## 1. 当前需要纠正的结论

现有工作已经证明：

- matched-state paired measurement 可以运行；
- selected states 上存在 deterministic implementation discrepancy；
- 同状态 repeats 在当前协议下可以 exact；
- discrepancy 可以传播到 gradient、parameter update 和 next state；
- 两个已测 region repair 的 effect 明显依赖 state。

现有工作尚未证明：

- Qwen3 目标训练状态分布上存在 material average bias；
- 任一 operator/region 对这个 population bias 有稳定贡献；
- 局部 bias 导致长期精度、收敛速度或训练时间变化；
- eager 是 truth，或 compiled 存在 correctness error。

A/B/C 和 20-state event bank 都是 pilot/selected-state evidence。它们不能承担 population
bias 结论。下一阶段不继续以“增加 repair family 数量”为主线。

## 2. Oracle 的对象

对固定 query 中的 state `s`、repeat `r` 和 endpoint `Y`，定义配对差值：

`D(s,r) = Y_compiled(s,r) - Y_eager(s,r)`。

eager 在这里是 baseline，不是数学真值。对同一个 query，Oracle 分开估计：

- `m(s) = E_r[D(s,r) | s]`：该 state 上可重复的 implementation effect；
- `B = E_{s~P}[m(s)]`：目标 state distribution `P` 上的 average shift；
- `H = Var_{s~P}(m(s))`：state-conditioned heterogeneity；
- `N = E_{s~P}[Var_r(D(s,r) | s)]`：同状态 paired runtime variability；
- `U`：有限 trajectory/state clusters 导致的 sampling uncertainty。

`H` 不叫 runtime variance，`U` 不进入 `H/N`。固定 reduction、cast placement 或
reassociation 可以贡献 `m(s)`、`B` 或 `H`；只有 same-state replay 自身变化才贡献 `N`。

这些名称依赖 endpoint：上游零均值 replay noise 经过 clipping、normalization、optimizer 或
其他非线性后，可能在下游 endpoint 形成非零 `B`。因此 Oracle 可以在各 endpoint 分解
`B/H/N`，但不能仅凭分解把某个数值机制永久贴成“bias mechanism”或“variance mechanism”。

## 3. 训练 bias 的主 endpoint

单纯要求每个 step 的 parameter-coordinate discrepancy 同号过强；只报告 discrepancy norm
又没有方向。Qwen3 confirmatory query 使用两个并列的 primary ledgers：update bias 用于连接
算子，task-transition bias 用于连接宏观训练含义。二者不能互相替代。

### 3.1 Update ledger U1：reference-update-aligned forcing

令 `U_R(s)` 和 `U_C(s)` 是从同一完整 pre-step state 得到的一步 parameter update，
`delta_U(s)=U_C(s)-U_R(s)`。定义：

primary signed endpoint 定义为：

`g(s) = <delta_U(s), U_R(s)>`。

它没有 state-specific 分母：reference update 很小时不会因除法而支配总体均值，`U_R=0` 时自然为 0。
原来的相对比例
`a_rel(s)=<delta_U(s),U_R(s)>/||U_R(s)||^2` 继续作为描述性 scale-free profile；它只在
`||U_R(s)||>0` 时定义，不能用任意 epsilon 补值，也不能在没有事前 norm-floor/conditional-population
合同的情况下承担 primary B claim。

解释：

- `g(s)>0`：candidate discrepancy 在 baseline update 方向上做正向贡献；
- `g(s)<0`：candidate discrepancy 在该方向上做负向贡献；
- `E_P[g(s)]`：跨 step 方向变化后仍有一致符号语义的 aligned forcing，且自然降低 tiny-update states
  的权重。它的单位是 update-squared，不应被描述为百分比加速。

方向与幅度权重 `U_R(s)` 在看到 candidate discrepancy 前由 baseline 定义，因此不是事后选择。
它仍然是 baseline-relative，不表示 baseline update 正确或更好。

该内积还依赖固定 parameter coordinate system、optimizer realization、learning-rate/scale 与模型
parameterization；对函数等价的重参数化并不保持不变。因此 U1 的总体 B 只能解释为“这个冻结训练合同
中的 aligned update forcing”，不能称为 compiler 固有常数。较大 baseline updates 对 aligned-dot B
贡献更大是 estimand 的组成部分；relative-U1 与 phase profile用于揭示这种 scale dependence，T1 则提供
不直接依赖参数欧氏坐标的一步任务 functional。

### 3.2 Update ledger U2：coordinate-frame mean update shift

保留 `E_P[delta_U(s)]` 及预声明归一化范数。这能检测固定参数坐标中的平均漂移，但不能
单独覆盖“每一步随训练方向一起旋转”的系统效应。高维 norm 的 confidence procedure
必须以 trajectory cluster 为抽样单位；当前 coordinatewise normal approximation 只作 pilot。

### 3.3 Task-transition ledger T1：task-level one-step transition shift

在不参与训练更新的固定 evaluation functional 上测量更新后差异：
`t(s)=F(S'_C)-F(S'_R)`。Qwen3 实例冻结两个不合并 endpoints：T1a 是 disjoint baseline-anchored
rollout bank 上的 post-update GRPO surrogate loss shift；T1b 是外部 arithmetic correct-answer
bank 上的 teacher-forced NLL shift。具体 bank identity 必须在读取 compiled result 前冻结。
它们用于判断 update shift 是否有即时任务含义，不自动预测长期训练。

loss、gradient、clip/AMP/skip 等仍保留为 propagation/semantic ledgers，不与 U/T ledgers
合并成一个 verdict。若 T1 未实例化，只能确认 average update shift，不能把它称为
macro-relevant bias。

## 4. 目标 state distributions

不得把不同 provenance 的 states 静默池化。至少建立两个独立 query：

1. `P_R`（primary）：按预声明规则从多条独立 eager-anchored 训练轨迹抽取 states；回答
   “在 baseline 会访问的 states 上切换到 compiled，一步平均 effect 是什么”。
2. `P_C`（sensitivity）：从 compiled-anchored 轨迹抽取并冻结 states，再做 matched probe；
   回答“在 candidate 实际访问的 states 上切换 realization，effect 是否保持”。

若二者不同，报告 distribution dependence，不将其平均成“compiler 固有 bias”。
每条轨迹是最高层 sampling cluster；phase/step 是轨迹内 block；token/tensor coordinate 不是
独立 population sample。

## 5. State-bank sampling design

### 5.1 Inclusion rule

- 使用多个独立 trajectory seeds；
- 对 early/middle/late 等预声明训练阶段分层；
- 在每个 phase 内按固定随机规则抽 step；
- 不依据 fork、raw delta、已知 repair effect 或 loss anomaly 选择 state；
- 保存完整 model、optimizer、scheduler/scaler、RNG、minibatch 和 realization identity。

Q-R calibration 不再使用旧 30-step fresh-optimizer restart 作为“全训练”代理。它从固定
Qwen3-0.6B base revision 启动 4 条完整 300-step eager-anchored GRPO trajectories；early、
middle、late 分别为 optimizer steps 1–100、101–200、201–300。每个 phase 的 8 个 pre-step
states 由 manifest hash 排序预先选择。optimizer/scaler/scheduler 从 step 1 自然演化，不在 phase
边界重置。

初始 calibration bank 以 `4 trajectories × 3 phases × 8 states = 96 states` 为预算起点，
每个 implementation 至少两次 same-state replay。这个 bank 只用于估计尺度、冻结 endpoint、
检查 controls 和做 prospective precision calculation，不产生 population verdict。

confirmation 使用全新的 trajectories。calibration bank 估计 regular trajectory-level scale 后，
在 confirmation 解盲前一次性冻结 trajectory 数量（regularity-conditional mean 第一批不得低于
8）、desired interval half-width、tail prevalence target 和 resource cap。8 不是 universal
sufficiency。若要求以 95% 概率覆盖 prevalence 至少 5% 的 trajectory regime，design 至少需要
59 条独立 trajectories。不能边看 effect sign/significance 边扩样，也不能通过增加同轨迹
tokens/states 假装增加 top-level coverage。

若 self/paired repeats exact，后续预算优先增加 trajectories 和 states；若检测到 replay
variability，再增加 repeats 并明确开放的随机源。

### 5.2 Weighting

primary population summary 对 trajectory 等权，再按预声明 phase/step 权重聚合。不得让
token 数、sequence length 或某条轨迹记录更多 states 而隐式取得更大权重。另报 phase-
conditional 和 trajectory-conditional summaries。

## 6. Oracle calibration controls

在真实 confirmatory verdict 前，必须通过以下区分能力检查：

| control | 预期 Oracle 结论 |
|---|---|
| eager/eager 与 compiled/compiled replay | measurement floor；只识别 N，不产生实现 B |
| 已知固定方向的小扰动 | B 非零，H 可小，N 可零 |
| state-dependent、总体严格抵消的确定性扰动 | global B 近零，H 非零，N 近零 |
| same-state repeat 随机变号扰动 | N 非零，不冒充稳定 m(s) |
| 罕见 state 才触发的确定性扰动 | tail/conditional effect；是否形成 B 取决于 P 和权重 |

控制项既检查 estimator，也检查 verdict wording。合成 control 证明区分能力，不证明真实
compiler effect；真实 eager/compiled bank 才承担 empirical claim。

## 7. Uncertainty 与判定门槛

- scalar primary endpoint 先产生每条 independent trajectory 的冻结加权 estimate，再以
  trajectory-level small-sample t procedure 作为 primary inference；CR2/Satterthwaite 与 wild
  cluster bootstrap-t 只作预声明 sensitivity checks；不得按 token 或 tensor coordinate inference；
- population claim 另设 tail-coverage ledger：预声明 `p_min/alpha`，并用 independent trajectory
  count 证明 design coverage；强偏斜/rare-regime gate 不足时降级为 finite-bank 或
  regularity-conditional claim；
- calibration/confirmation states 分开；所有方向、归一化和 tolerance 在 confirmation 前冻结；
- 报 effect interval，而不只做 `B=0` significance test；
- practical tolerance `tau_B` 必须来自独立外部应用/科学容忍合同，不能由 measurement floor、
  negative-control envelope 或观测到的 compiled effect 代替；measurement 与 materiality sources
  分别冻结。

每个 endpoint 分两个 ledger 判定，不能用 practical threshold 掩盖“是否存在平均偏移”。

**Shift-existence ledger：**

- `REPRODUCIBLE_AVERAGE_SHIFT`：相对 measurement floor 的 interval 排除零，并在 independent
  confirmation trajectories 上保持预声明方向；
- `NO_STABLE_AVERAGE_DETECTED`：未建立总体方向，但不自动等于 practical equivalence；
- `INDETERMINATE` 或 `INVALID`：精度、transport、identity 或 coupling 不足。

H 是单独的估计 ledger，不再因一个截断后的正点估计改变 B verdict。未约束 variance-component
estimate、非负截断描述和 trajectory-level uncertainty 必须并列报告。条件方向主张则写成明确的
`endpoint::phase=<phase>`：只有 early/middle/late 全部事前进入 multiplicity 并在独立 trajectories 上
确认后，才称该条件分布存在 B；否则 H/phase profile 只描述“effect 如何变化”。

**Materiality/compatibility ledger：**

- `MATERIAL_AVERAGE_SHIFT`：effect interval 在预声明方向上完全超过 `tau_B`；
- `DETECTED_BUT_BELOW_MATERIALITY`：可区别于 measurement floor，但未超过 practical bound；
- `PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT`：B interval 完全落入 equivalence region；只约束平均
  shift，不表示逐 state、tail event 或两个实现整体等价；
- `UNINSTANTIATED_MATERIALITY`：没有独立 practical tolerance，仍可报告 shift existence。

correctness verdict 继续独立；没有 truth/spec authority 时为 `UNINSTANTIATED`。

此外必须报告 `TAIL_COVERAGE_SUFFICIENT / TAIL_COVERAGE_INSUFFICIENT /
REGULARITY_CONDITIONAL_ONLY`。tail ledger 不由普通 mean p-value 替代。

## 8. Bias contributor analysis 的启动门

只有 U1/U2 或预先选择的 T1 endpoint 在 confirmation bank 上得到
`REPRODUCIBLE_AVERAGE_SHIFT`，才启动该 endpoint 的算子 bias-contributor 主线。materiality
可以尚未实例化，但此时贡献结论只能解释 measured average shift，不能称训练风险或危害。
若 global B 未确认，仍可报告 conditional region effects，但不能称“贡献总体 bias 的算子”。

候选可以用 calibration/旧 pilot 产生，但候选集合冻结后必须使用新的 contributor-confirmation
trajectories；不能在 endpoint-confirmation bank 上筛选贡献最大的候选，再用同一批数据确认它。对候选
intervention `o`，在 contributor bank 的配对 states 上定义：

`C_o^repair = B_compiled - B_compiled_with_o_repaired`。

若 `C_o^repair` 在 heldout trajectories 上沿 endpoint-confirmation 冻结的 B 方向超过预声明 floor，且
multiplicity-adjusted interval、transport、validity 和 sensitivity gates 全部通过，则称：

“在 query `P/Y/protocol` 和该 repair intervention 下，`o` 对 implementation-relative bias
有可复现的方向贡献。”

“repair 使绝对 Bias 降低”是更强的结论，必须另有 simultaneous residual rule；不能通过比较两个点估计
得到。若 repair 越过零并产生更大的反向 residual，只能报告 directional contribution with overshoot。

不追踪贡献 N 的 operator。N 只在 Oracle 层作为混淆和精度来源受到控制。

限制：

- repair contribution 不自动等于 root cause、必要性或充分性；
- 多个候选可能抵消或交互；top candidates 需要 pair/coalition checks；
- whole fused callable 的结果只能归于 generated region；保持 fusion/layout 和其他 choices
  不变前，不能下放到 constituent source operator；
- injection 不是 repair-contributor screening 的必要条件，但缺 injection 时不得作充分性语言。

完整的数据隔离、multiplicity、interaction、intervention-integrity 与 claim boundary 见
`QWEN3_BIAS_CONTRIBUTOR_CONFIRMATION_GATE_V0_1_2026-07-20.md`。实例化入口依次为
`QWEN3_BIAS_CONTRIBUTOR_CANDIDATE_FREEZE_TEMPLATE_V0_1.json`、
`QWEN3_BIAS_CONTRIBUTOR_PRECISION_INPUT_SPEC_TEMPLATE_V0_1.json`、由 planner 生成的
`QWEN3_BIAS_CONTRIBUTOR_PRECISION_PLAN_TEMPLATE_V0_1.json` 和
`QWEN3_BIAS_CONTRIBUTOR_CONFIRMATION_MANIFEST_TEMPLATE_V0_1.json`。总体 B 的 variance 不得直接作为
repair contribution 的 variance；贡献精度需要独立 pilot variance 或外部固定的保守样本量依据。

## 9. 长期训练的角色

只对 heldout state bank 上通过方向贡献 gate，且具有 absolute-reduction/materiality evidence 或明确
科学优先级的少量
intervention 做长期验证。比较 eager、compiled、
compiled+repair 的多 seed training distributions，检查收敛速度、最终 quality、稳定性和训练
时间是否按局部预测变化。

长期实验是 impact validation，不是 matched-state local estimator。自由轨迹分离本身不能
反推某一步或某算子的 bias。

## 10. 执行顺序与交付物

### C0：合同冻结

交付物：Qwen3 query manifest，包含 `P_R/P_C`、U/T endpoints、随机耦合、
state weights、tolerance 来源、bootstrap unit、construction gates。

### C1：区分能力校准

交付物：五类 controls 的 machine-readable records 与 fail-closed verdict。任何概念混淆先修
Oracle，不进入真实 population claim。

### C2：真实 Qwen3 population calibration

交付物分两批：4-trajectory/96-state calibration bank 用于冻结分析与 prospective precision；
随后使用一次性冻结数量且至少 8 条新轨迹的 regularity-oriented confirmation bank。若声明
rare-regime coverage，trajectory count 必须由 `p_min/alpha` 单独决定。最终保存 raw paired
measurements、B/H/N/U、phase/trajectory conditional report 和 independent verdict。

### C3：bias contributor screening

前提：C2 确认 reproducible B。先冻结候选；再以独立 contributor pilot 或外部保守依据冻结贡献量自身的
精度设计；最后使用独立于 calibration、endpoint confirmation 和 contributor pilot 的 contributor bank。
交付物：精确区分 operator、generated kernel/region 与 intervention package 的 repair effect、baseline
transport、multiplicity-adjusted intervals、null control 和 interaction audit。不以 runtime-family
覆盖率冒充 bias attribution。

### C4：长期影响验证

前提：C3 得到稳定 contributor。交付物：多 seed 长期 endpoint distributions；结论限定为
impact/reproducibility，除非另有 correctness authority。

## 11. Stop / kill criteria

- U/T target B 在 independent trajectories 上落入 practical equivalence region：停止“寻找
  global-bias contributor”，转为 conditional impact 或其他 endpoint；
- B 的符号/大小主要由未声明的 state weighting 决定，且在 `P_R/P_C` 间不可运输：拒绝
  “operator 固有 bias”，只保留 distribution-specific claim；
- Oracle 无法在 controls 中区分 stable state effect 与 replay noise：停止真实归因；
- candidate region 在 calibration 有效但 heldout contribution 不复现：不进入长期验证；
- repair 改变 fusion/layout/realization identity：降级为 intervention-dependent region result；
- 局部 B 降低但长期 endpoint 无一致变化：保留 local bias contribution，拒绝 long-run harm；
- 没有 independent specification：任何结果都不能升级为 compiler correctness/bug claim。

## 12. 旧证据如何复用

- 20-state bank：复用为 selected-state semantic/heterogeneity pilot，不估计 target-population B；
- A/B/C transition snapshots：复用来验证完整 state replay、endpoint serialization 和 exact N
  controls，不作为 representative population；
- SiLU/final-norm/cast repairs：复用为 intervention mechanics、state dependence 和 null control；
  只有在 C2 state bank 重跑后才能进入 bias-contributor ledger；
- runtime/source-op census：保留为 denominator inventory，但下一阶段不追求所有 family repair；
- existing long free trajectories：只作 divergence context，不用于估计 local B。

现有 checkpoint inventory 不含可支持目标 early/middle/late population 的完整原训练 optimizer
states，因此新 calibration 选择完整重跑，而不是把 model-only checkpoint 配 fresh optimizer
伪装成自然中期 state。

现有 A/B/C update vectors 的 retrospective relative-U1 check 已完成：A exact-null，B/C 非零但
aligned sign 相反，且 discrepancy 几乎全为 orthogonal component。它证明 signed alignment 比 raw L2
多一层方向信息，也证明相对比例的 denominator 风险以及任何 U1 都不能替代 T1。primary aligned dot
可由已存 ratio 与 reference norm 重构，无需重跑。证据见
`QWEN3_UPDATE_ALIGNED_SHIFT_RETROSPECTIVE_FINDINGS_V0_1_2026-07-20.md`。

少 cluster coverage audit 已完成：trajectory-t 在 Gaussian/对称重尾 cases 表现接近 nominal，
但 centered-lognormal 与 5% rare-trajectory mixture 明显 under-cover；naive state-level interval
严重失效。因此 population claim 新增 tail gate。证据见
`BIAS_ORACLE_CLUSTER_COVERAGE_FINDINGS_V0_1_2026-07-20.md`。

T1 结构已经冻结为 heldout GRPO surrogate 与 heldout correct-answer NLL 两个独立 endpoints；greedy
reward 不作为唯一单步 functional。具体 rationale 见
`QWEN3_TASK_TRANSITION_ENDPOINT_RATIONALE_V0_1_2026-07-20.md`。

## 13. 当前实现与计划之间的缺口

`src/forkcert/oracle.py` 可以复用 raw paired grid、same-state paired N、reference/candidate self
variance、基础 mean field 和 repeat-noise correction，但不能单独产生本计划的 population verdict。
新增的 trajectory-aware record/aggregation/confirmation 路径已经补上其中多数结构；下表区分旧 core 与
当前整体路径：

| requirement | 当前状态 |
|---|---|
| trajectory/phase/block identity 与权重 | 旧 core 无；新 record 与单/多轨迹 aggregators 已实现并测试 |
| reference-update-aligned primary endpoint | denominator-free aligned dot extractor 已实现；relative ratio 仅作诊断；等待完整 calibration |
| calibration/confirmation 数据隔离 | confirmation evaluator 已 fail closed；manifest 尚未实例化 |
| trajectory-cluster small-sample uncertainty | trajectory-t、df 与 sign-flip sensitivity 已实现；regularity/tail claim 仍需冻结 |
| `P_R` 与 `P_C` 独立 query | 当前只执行 eager-anchored `P_R/Q-R`；`P_C` 尚未实例化，不能外推 |
| shift existence 与 materiality 两套 verdict | 新 population core 已分开；practical tolerance 尚未实例化 |
| prospective precision stopping rule | procedure 已实现并测试，包含 sign-flip p-value 分辨率；half-width、variance floor、multiplicity、tail scope 与 resource cap 尚未实例化 |
| rare trajectory regime coverage | tail ledger/formula 已实现；是否声明 prevalence coverage 仍是 confirmation 前选择 |
| Bias-contributor design gate | candidate-freeze → frozen precision input spec → reproducible planner output → final confirmation 已有模板/planner/validator 与 13 precision+design tests；C2 B 尚未确认 |
| Bias-contributor result estimator | 三臂配对形成 baseline/residual/contribution；C 的 B/H/N、overshoot、simultaneous absolute-reduction 与 veto-only sensitivity 已实现并通过 9 tests；真实 contributor study 未启动 |
| contributor pilot provenance | precision planner 已验证不读取 mean/sign，但真实三臂 records 到 hashed pilot summary 的 builder/validator 尚未实现；C3 前必须补齐，当前不阻塞 C2 |

当前 C0/C1 procedure 已冻结并完成 81 项定向回归；真实 GPU calibration-0 正在按同一合同逐 state
运行。立即顺序是：完成剩余 records → 单轨迹描述 → calibration-1/2/3 → 冻结并运行 independent
endpoint confirmation。只有 confirmation 支持某个 B，才实例化 contributor artifacts；81 tests 证明
procedure gates，没有替代真实 population evidence。
