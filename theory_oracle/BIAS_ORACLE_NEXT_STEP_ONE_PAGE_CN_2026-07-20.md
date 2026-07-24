# Bias Oracle 下一步：一页版

## 当前核心问题

在预先声明的 Qwen3 训练轨迹分布上，eager 和 compiled 从相同 state 出发时，存在哪些可复现的
implementation-relative effects：全局有向平均偏移、事前声明条件下的有向偏移、语义事件变化，或同状态运行变动？
这些 effect 可以同时存在，不是互斥选项；全局平均 B 不是唯一入口。state heterogeneity 是用来描述
effect 如何随 state 变化的结构，不能自动升级成某个事后挑选子群上的 Bias。

第一阶段不声称覆盖罕见训练异常，不判断 eager 正确，也不寻找贡献 runtime variance 的算子。

## 已经知道

- 少数 matched states 上 implementation difference 确实存在；
- 当前确定性协议下 same-state repeats exact；
- difference 可以传播到 gradient、parameter update 和 next state；
- local region repair effect 随 state 改变；
- 这些证据不足以证明训练分布上的 bias。

## 第一步实验

### 0. Construction smoke

只取一个预声明 state，验证：

1. 完整 pre-state 捕获不改变原 eager trajectory；
2. eager/compiled fresh-process arms 从完全相同 state 出发；
3. 两次 same-state repeat 可比较；
4. 能产生 update endpoint U 和 task-transition endpoint T；
5. 所有 state、RNG、compiler、artifact hashes 通过 fail-closed validator。

smoke 只判断测量系统是否有效，不读取 bias 结论。

当前进度：calibration-0 已自然完成 300 steps；24/24 个预声明 snapshots 的内容/history/identity batch
audit 全部通过。事前固定的 early/step-10 完整 record 也已通过：两组 paired transition repeats、U1/U2、
T1a/T1b、nested evaluator repeats 和 arm/pair artifact validation 均有效。

同时修正了两个概念风险：单 state 的 repeat-mean 只能叫 `state effect`，不能叫 B；task endpoint 的
transition repeats 与 common-evaluator repeats 是两个不同层次，必须分开估计。

T1a 还有第三个未识别层次：两次 bank generation 使用相同 frozen seed 并要求内容 hash 完全一致，
它们只是 reproducibility check，不是独立 rollout-bank samples。因此当前 T1a N 不单独识别 bank-sampling
variance，跨-state T1a H 同时包含 state 变化和 state-adaptive bank content 变化；T1a 是 local
policy-relative functional，endpoint class 为 `SIGNED_STATE_ADAPTIVE_TASK_ENDPOINT`。T1b 使用固定
correct-answer bank，class 为 `SIGNED_FIXED_BANK_TASK_ENDPOINT`，更接近固定外部 functional。

当前 N 还严格条件化于固定 implementation realization：同一 state、同一 arm 的 repeats 必须具有相同
compiler-config 与 graph-family digest；这两个已观测字段漂移时 record 直接 invalid。但当前 contract 未记录
最终 generated-kernel/autotuning variant identity，因此不能声称这类隐变量已被排除：若它们仍变化，当前
数据只能将其混在 residual N 中，无法与 GPU execution nondeterminism 分开。若要研究或排除它们，必须
扩展 realization instrumentation，或另立随机 realization factor/query。

step-10 进一步表明：稳定 update effect、paired N=0、无离散 fork、两个符号不同的 task effects 可以
同时出现。因此最终产出必须是 endpoint-indexed B/H/N/U profile，而不是单一 bias score 或 fork rate。

后续 early states 还暴露了另一个不能省略的轴：step-91 的 clip event 有单方向净变化，而 step-97
同时出现一个 off→on 与一个 on→off，导致 clip 总数差为 0，但 token-level disagreement 非零。因此
semantic event 必须并列报告 signed directional rate、两种单向 transition rate 与 disagreement rate；
后者是非负 semantic-impact profile，不得重新命名成 Bias，也不得进入只寻找 Bias contributor 的分析。
但经事前冻结并独立复现的 disagreement 可以进入另立的 semantic-impact contributor study。它可复用通用的
trajectory-level mean/interval machinery 并与其他 boundary endpoints 共享事前 multiplicity correction，但不能借用
signed-B identity、方向或结论名称。
四轨迹聚合使用独立字段 `calibration_profile_mean/conditional_profile_means` 保存这类非负量，避免底层
通用均值计算器的内部字段被误读成对外 Bias 结论。
record validator 还逐一证明 pair 中复制的 reference/candidate events 等于其链接 arm、event difference
确为 candidate-reference、clip count 与逐 token decisions 一致；否则整条 state evidence fail closed。

trajectory 汇总也已冻结两条防概念偷换规则：任何缺失/未定义 endpoint 都保留在 frozen bank 中并令该
endpoint fail closed，不能 complete-case 删除；U2 必须先逐参数坐标平均完整有符号 update-delta field，
其 mean-field norm 与 per-state L2 magnitude 分开报告，后者永远不叫 B。相关统计、record、realization
与单/多轨迹聚合、precision planning、confirmation isolation、few-trajectory sensitivity 和 contributor
nightly 环境补齐兼容的 pytest 8.x 后，完整测试套件当前 `413 passed`；其中本次
average/conditional/semantic 算子准入表述涉及的三组定向测试为 `30 passed`。
自动链条已完整产出并验证 step-10/18/57/62/78/91/96/97/105/107/108/126/140/151/153/166/204/222/224/227/236/264/273
二十三个 records；真实 calibration-0 汇总仍等待 step-275 最后一个 late-state record。

calibration acquisition 过程中 task evaluator 的 source hash 曾发生变化：前八个 records 使用旧版本，
step-105 使用加入显式 randomness-scope 的版本，当前版本又增加了 evaluator-repeat ID 配对检查。这里
不能把不同 source 假装成完全同一测量程序。兼容性审计会沿 provenance 重新读取四个 arm 的 task
artifact，以当前语义重算 T1a/T1b profile，并核对 bank identity；当前二十三个 records 均通过。旧记录只被
标记为 `LEGACY_IMPLICIT_SCOPE_NUMERICALLY_REVALIDATED`，此兼容只服务 calibration/design；正式
confirmation 必须使用单一、事前冻结的 acquisition code。

二十三个已完成 states 的只读 calibration diagnostic 也说明为什么现在不能提前宣布 Bias：same-state 两次
repeat 在已测 scalar endpoints 上完全一致，但 state effect 的符号普遍混合——training loss 为 7 正/14 负/2 零，
U1 aligned forcing 为 9 正/12 负/2 零，T1a 为 10 正/11 负/2 零，T1b 为 13 正/8 负/2 零。这首先证明了明显的
state-conditioned heterogeneity，同时表明一个有限样本均值可能由少数大 state effects 主导；它既不证明
总体 B 为零，也不证明总体 B 非零。clip 则进一步分开了净方向变化与无净方向的 disagreement。

四轨迹 campaign 控制器也已启动，但它不增加任何统计结论：当前只轮询 calibration-0 的 authoritative
record/ledger；达到 24/24 且完整有效后，才做单轨迹 scalar/U2 聚合并串行续接 calibration-1/2/3。
控制器禁止并发 GPU 进程、残缺 source 静默覆盖、低于 2 TiB 的继续写入以及 invalid evidence 后继续。
batch audit 还将 plan identity 绑定到 source 实际记录的 YAML/metadata；calibration-0 的 seed、data slice、
300-step horizon、24 个 target digests 与 eager-source 检查均已重新通过，不能只靠 plan 自我声明。

四轨迹层也已固定：必须先分别形成四个完整 trajectory estimates/有符号 U2 mean fields，再对四条
独立轨迹等权汇总，顶层自由度只有 3；96 states 不得冒充 96 个独立样本。四轨迹产出仍只用于
confirmation 规模设计，不允许触发 population-B 或 operator-contributor 结论。

confirmation freeze gate 的 procedure shell 也已补齐，但关键参数仍明确 `UNINSTANTIATED`：endpoint
family/multiplicity、每个 endpoint 的独立 desired half-width、variance floor、resource cap 和 tail scope。
每个 endpoint 必须分别冻结 `desired_half_width`、`variance_floor_sd` 和 `shift_existence_floor` 的
`threshold_sources`，并给出 description/selection rule，明确不使用 calibration candidate mean/sign。
三种 source role 不能互换：应用科学容忍度可以支持目标精度，却不能冒充数值 shift floor；exact-zero
null 也不能冒充运行方差下限。缺失来源或角色错配时 precision plan fail closed；这使“每个阈值为什么
这样取”成为可审计合同，而不是一个笼统 tolerance 标签。
具体的独立来源已整理为 `QWEN3_CONFIRMATION_THRESHOLD_SOURCE_PROTOCOL_V0_1_2026-07-20.md`：
reference/reference、candidate/candidate fresh-process control 与 identical-post-state evaluator control
分别约束 acquisition/runtime/evaluator 的 null envelope。它们只支持 measurement-level threshold，
不能冒充 practical harm、correctness 或长期训练阈值；所有实际数值仍为未实例化。
严格的单轨迹 null-control aggregator 也已加入：只读取同实现 repeat-2 minus repeat-1、同 post-state
evaluator-repeat contrasts、update artifact/next-state/event equality，并在不足 24 states 时 fail closed。
对当前二十三个完整 records 的只读预检全部可解析，所有这些 controls 均未观察到变化；该证据只能写成
`NO_OBSERVED_WITHIN-STATE_VARIATION_AT_R=2`，不能写成 runtime variance 不存在，也不能自动提供正的
variance floor。
四轨迹 null-control aggregator 也已接入 campaign：它要求 calibration-0..3 各有一个完整、哈希绑定且由
当前分析代码生成的单轨迹 summary，再汇总 trajectory-level signed/absolute contrasts、最大 state
contrast 与 artifact/event exactness。该产出仍标为 measurement-null description，不能自行实例化正
half-width、variance floor 或 practical tolerance。

这里需要一项理论修正：`B existence`、`interval precision` 和 `practical materiality` 不能被强制绑成
同一张票。没有独立 scientific/measurement precision target 时，为了让 planner 可运行而发明
desired half-width 反而不科学。`BIAS_ORACLE_CONFIRMATION_DECISION_MODE_V0_1_2026-07-20.md` 因此提出
两种合法模式：Mode E 事前冻结资源、trajectory bank、multiplicity 与 exact/null-control floor，只检验
有方向 average shift，完整报告实际区间但不虚构精度承诺；Mode P 只在存在独立 half-width/variance
floor 时做 precision-targeted planning。当前项目第一版推荐 Mode E；planner/evaluator 的 Mode E 核心
迁移已经完成并由定向测试覆盖，包括 fixed-resource 的 global 与预声明 phase-conditioned claim。实际
trajectory 数、endpoint family、multiplicity、tail scope 与 shift floor 仍未冻结，因此 confirmation 继续
保持 uninstantiated。

全局平均 B 不再是唯一分析入口。新增的 calibration-only boundary diagnostic 用 eager/reference 侧的
干预前 signed clipping margin 定义 near-boundary 区域，并逐 token 复核重建的 clip decision 与原始记录
完全一致。在当前十六个具有 eligible clipping decisions 的 states 上，全 eligible token 的 state-weighted mean margin shift 符号随 state 变化；
near-boundary 区域出现了 event change，但不同 tau 只有 0--8 个 states 有 exposure，证据远
不足以确认 conditional B。该诊断只用于选择并冻结候选 tau；独立 confirmation 前不允许触发 operator
attribution。summary 现已绑定冻结 capture plan、每个 transition/result/minibatch 的哈希和分析代码哈希；
权重也保持 trajectory→phase→state 层次，任一 phase 无 exposure 时 conditional estimand 明确未实例化，
不能用其他 phase 补齐。正式入口因此是：独立复现的 global B、预声明且 baseline-defined 的 conditional
B，signed semantic shift，或明确标为 non-B 的 nonnegative disagreement。三者可在同一事前冻结的
boundary multiplicity family 中共享保守的 family-wise error 控制，但必须使用不同的 estimand、verdict 和
attribution claim 名称；disagreement 仍不叫 B。tau confirmation 合同见
`QWEN3_BOUNDARY_CONDITIONAL_B_CONFIRMATION_CONTRACT_V0_1_2026-07-20.md`：只按 reference support 排除
不可定义的 tau，所有保留 tau 成套进入 multiplicity，不挑 observed effect 最大者。独立 boundary
confirmation evaluator 与 fail-closed spec template 已实现，但实际 support family、spec 与
confirmation bank 尚未实例化，因此目前仍不能触发 attribution。四轨迹完成后的 write-once support
freezer 已实现：它只读取 reference-side phase support，要求每个 trajectory×phase 至少两个 exposed states，
并保留全部满足条件的 tau；candidate shift/event 字段不参与选择。evaluator 以 reference repeat-1 固定
condition mask，但仅当两个 reference repeats 的 scorer log-probs 完全一致；否则正式 conditional claim
fail closed，并要求独立于 effect repeats 的额外 reference anchor，防止用同一份 runtime noise 选样本后
又估计 effect。当前 20/20 个完整 states 均满足 exact-reference-repeat 条件。evaluator 保留两个 paired
repeats 的 N，并对 continuous/signed-semantic endpoints 做独立 family
multiplicity；不允许与 global family 做未经调整的 joint claim。boundary endpoint 只能经过专用 bridge 与
record-integrity gate，不能进入旧的 global/phase 路径。disagreement 路径现已另外约束为关系量：baseline 是
`D(reference,candidate)`，repair residual 是 `D(reference,intervention)`，不允许把它伪装成普通 arm scalar。
该 conditional estimand 的权重也已冻结为
`EQUAL_TRAJECTORY_PHASE_EXPOSED_STATE_NEAR_TOKEN_V0_1`：trajectory、phase、exposed state 依次等权，
最后才在 state 内平均 near-boundary tokens。它不是 exposure-pooled token mean。family、confirmation、
bridge 与 contributor records 必须携带同一 ID；否则即使 tau 相同也视为不同 estimand 并 fail closed。
为防止该待完成项被绕过，新增 write-once boundary attribution bridge：它只能从 independently confirmed
且 operator-ready 的具体 tau endpoint 生成，冻结 exposed state census 与每个 state 的 reference-anchor
token-mask SHA-256。repair/injection 必须复用同一 mask，不能在 intervention 后重新选择分析成员。该 bridge
不选择 operator，也不让现有 global/phase contributor validator 自动接受 boundary endpoint。
bridge 后的 record-integrity gate 也已实现：它要求 intervention record bank 与 bridge 路径/hash、完整
exposed-state census、每个 state 的固定 mask、两个 repeats 及三个 arms 的 mask/cardinality 完全一致；
repair contribution 固定为 `candidate-intervention`，injection contribution 固定为
`intervention-reference`。任何额外 state、缺失 repeat、干预后 mask 重选或公式漂移均 fail closed。该 gate
只产出 same-bank mechanism-pilot paired effect records，本身不形成 operator-effect verdict。
这里进一步区分了 state-bank role：现有 bridge 复用 endpoint-confirmation states，只允许 same-bank
mechanism pilot，不能形成 population operator contribution。正式贡献研究必须先冻结另一组独立
contributor trajectories，在任何 operator outcome 产生前生成 reference masks，并排除 calibration 与
endpoint-confirmation 的 trajectory ID、seed 和 data slice。独立 bank freezer 已实现上述 disjointness、
完整 8×3 census、phase support、anchor exactness 与 pre-intervention 时序门；它通过后也只允许开始测量，
不提前宣告贡献。
独立 records validator 与 trajectory-level evaluator 也已接通。它们不会因为 repair response 在新 bank
稳定就直接宣布贡献：首先要求原 conditional endpoint 的 `candidate-reference` effect 在新 bank 上沿冻结
方向 transport；随后才检验 repair 的 `candidate-intervention` 或 injection 的
`intervention-reference`。两道量都必须通过事前 family multiplicity、trajectory-level interval、sign-flip
sensitivity 与 fixed-resource interval reporting。合成反例已验证：即使 intervention response 恒定，只要
baseline 在新 trajectories 上正负抵消，population contribution gate 仍关闭。
该 primary contribution 也不等于 explained fraction 或 absolute error reduction。repair evaluator 会另外
描述 repaired residual 与绝对 discrepancy reduction，但不让它们借用 directional-contribution verdict。
overshoot 反例已覆盖：从 `+1` repair 到 `-1` 时 directional contrast 为 `2`，绝对 discrepancy reduction
仍为 `0`；系统允许前一个 intervention-dependent claim，但明确禁止“解释 200% Bias”或“误差已减少”。
端到端反例测试已验证：当全体 states 的 global B 精确抵消而事前定义的 near-boundary conditional B 稳定
非零时，global attribution 保持关闭、conditional bridge 可以开启；非负 disagreement 不能冒充 signed B。
另有 outcome-independent resource planner 在 confirmation bank 冻结前只根据 boundary family 大小与
family alpha 计算 sign-flip p-value resolution 所需的最小 trajectory 数；它不读取 effect mean/sign/variance。
boundary spec 必须同时绑定该 resource plan 和最终 confirmation manifest，避免到结果出来后才发现 J 不够。
全局 B 与 phase-conditioned B 也已分开：核心 estimator 总是保留 early/middle/late 三个事前阶段的
方向 profile，但默认只作描述；若 precision spec 把某个 endpoint 放入
`phase_conditioned_endpoint_family`，三个 phase 必须成套进入 multiplicity、variance planning 和独立
confirmation，不能在看到全局抵消后只挑一个阶段。通过的条件 claim 使用
`endpoint::phase=<phase>` 身份进入独立 contributor study；其他任意 state 分组不能事后升级。
step-140 提供了一个直接反例：其全 eligible margin shift 为正，加入后 partial middle-phase mean 从负转正；
但该 state 没有 semantic flip，只在较宽 tau 增加无翻转 boundary exposures。因而有限样本 global/phase
mean 的方向故事可以被单个新 state 改写，而 conditional semantic profile 不随之改变；这支持把两类
estimand 分开，却仍不允许在 late phase 缺失时宣布任何 population effect。
step-151 又补上了一个不同反例：其全 eligible margin shift 为正，使当前 middle-phase margin mean 在删除
任一单 state 后都保持为正；但同一 state 的 T1a 一步任务 effect 为负，并产生两个 clip off→on。这两个
semantic changes 在 reference-defined `tau=0.05` 条件中可见，在 `tau<=0.01` 时该 state 没有 exposure。
加入 step-151 后，已观测 states 上 `tau=0.05` 的净 directional event mean 恰好抵消为零，但 semantic
disagreement 仍非零。因此真实数据已经同时给出“连续均值与一步任务 effect 方向不同”和“净事件变化
为零但 disagreement 非零”两种情形。它们仍只是 calibration-0 的 design evidence，不是 population claim。
step-153 进一步将这两层分开：其全 eligible margin shift 为正，但在所有事前 `tau<=0.05` 中都没有
reference-near-boundary exposure，也没有 semantic event change。它因而继续推高 middle 的 all-eligible continuous mean，
却完全不改变 boundary-conditioned semantic profile；同时 T1a 仍为负。这说明 all-token continuous mean、
boundary-conditioned effect 与 one-step task effect 不能互相代替。
step-166 给出了同一 state 内更直接的反例：其 all-eligible mean margin shift 为正，但在 reference-defined
`tau=0.05` 的四个 exposed tokens 上 mean margin shift 为负，且没有 semantic disagreement。也就是说，
即使不跨 state 聚合，全-token average 与 near-boundary conditional effect 也可能方向相反；二者必须作为
不同 estimand 报告，且无 event change 时不能把 conditional continuous shift 自动升级为 semantic effect。
step-204 则暴露了另一类不能被平均掩盖的情况：该 matched state 和四臂 record 完整有效，transition、
task、update 与 semantic endpoints 的 eager/compiled effect 均精确为零，但其 advantages 全为零，因而
eligible clipping decisions 数为零。all-eligible clipping-margin mean 在这个 state 上不是零，而是未实例化。
boundary aggregator 现保留该 state census、输出
`UNINSTANTIATED_NO_ELIGIBLE_DECISIONS` 与 null 值，并令原 state-balanced phase/global all-eligible estimand
fail closed；它不再产生 NaN，也不 complete-case 删除该 state。near-boundary estimand 继续按独立的
exposed-state 条件合同处理。这个 fail-closed 结论只针对事前冻结的“每个 sampled state 等权、再在 state
内平均 eligible decisions”合同；它不等于所有 global estimand 都不可定义。若改成 exposure-weighted token
population，或显式条件于 state 有 eligible decisions，必须另立并预先冻结权重合同，不能事后把当前缺失
state 删除后仍沿用原 global-B 名称。机器可读 ID 现固定为
`EQUAL_TRAJECTORY_PHASE_SAMPLED_STATE_ELIGIBLE_DECISION_V0_1`，与 boundary conditional 的
`EQUAL_TRAJECTORY_PHASE_EXPOSED_STATE_NEAR_TOKEN_V0_1` 分开验证。
四轨迹 boundary aggregator 现还要求每条 trajectory 的 24/24 reference scorer repeats 全部 exact，并把
96-state anchor audit 带到 multi summary；family freezer 再独立验证一次。任一 state anchor 不稳定时，
repeat-1 条件 mask 不能因其他 states support 足够而被正式冻结，必须改用独立 anchor protocol。
step-222 恢复了 512 个 eligible decisions，并给出负的 all-eligible margin shift；但 T1a 为正，T1b、
training loss 与 update-aligned effect 为负，而且所有预设 `tau<=0.05` 均无 reference-near-boundary exposure，
也无 semantic disagreement。它进一步证明：eligible-conditioned continuous mean、task endpoints、update
endpoint 与 boundary/semantic endpoints 是不同 estimands，方向不必一致。由于 step-204 的该 margin endpoint
未实例化，原“每个 frozen state 等权”的 late/global eligible-conditioned mean 继续 fail closed；不能只用
step-222 等有值 states 做 complete-case average。其他在两个 states 上都定义的 core endpoints 不受此限制。
step-224 同样有 512 个 eligible decisions，all-eligible margin shift 为负，并首次在 late phase 的预声明
`tau=0.05` 中产生三个 reference-near-boundary exposures；其 conditional margin shift 为负但没有 event
change。late 目前只有这一个 exposed state，仍不能实例化 phase conditional estimand。该 state 的 loss 为正、
clipped-gradient alignment 为负、update alignment 为正、T1a 为负而 T1b 为正，且 next update 确定性不同。
因此连续、语义和一步转移是并列但可关联的 ledgers，不是“只有跨过事件边界才会产生 update effect”的必然链。
step-227 则给出更直接的 global-vs-conditional 反例：all-eligible margin shift 为正，但 `tau=0.05` 的五个
reference-near-boundary decisions 上 conditional shift 为负，并确定性地产生一个 `on→off` clipping event；
loss 与 clipped-gradient alignment 为负，update alignment、T1a 与 T1b 为正。加入该 state 后，calibration-0
的 `tau=0.05` 在 early/middle/late 每个 phase 均至少有两个 exposed states，因而这条 trajectory 内的
conditional estimand 已定义；原 state-balanced all-eligible global estimand 仍因 step-204 未实例化。这只
授权 calibration description，不是 independent confirmation 或 operator attribution。
step-236 又给出方向相反但同样关键的实证：all-eligible margin shift 为正；reference-defined
`tau=0.01/0.05` 各只有一个 exposed decision，其 conditional margin shift 为正，并确定性地产生一个
`off→on` clipping event。两边各自的 gradient/update artifacts、loss、events 与 next-state repeats 均精确，
所以当前协议下未观察到 N；loss、T1a、T1b 为正而 clipped-gradient alignment 为负，next state 不同。
加入该 state 后，`tau=0.05` 的 phase-balanced conditional continuous shift、signed semantic shift 与
nonnegative disagreement 在当前 trajectory diagnostic 中均为正；`tau=0.01` 的 late phase 仍只有一个
exposed state，故完整 conditional estimand继续未实例化。与此同时，原 state-balanced all-eligible global
margin estimand仍因 step-204 的零 eligible decisions而 fail closed。这个结果直接证明：可定义的
boundary-conditional/semantic estimand不能等待另一个权重合同下的 global B，也不能把单 token event
当作 population effect；它仍只用于 calibration/design。
step-264 则把 all-eligible continuous mean 与 boundary family 再次分开：其 512 个 eligible decisions 上
all-eligible margin shift 为正，但所有预声明 `tau<=0.05` 都没有 reference-near-boundary exposure，也没有
clipping disagreement；因此它改变 all-token continuous description，却完全不改变当前 boundary-conditional
或 semantic aggregate。该 state 的 loss、U1 aligned forcing 为负，T1a/T1b 为正，U2 magnitude 非零且
next state 不同，两边 self repeats 仍精确。它既不能用于支持“全局连续 shift 必然产生语义事件”，也不能
因无 event 而从一步 update/task ledger 中删除。
step-273 是第二个零 eligible-decision state，也是更强的 exact-null control：eager/compiled 的 loss、完整
clipped-gradient field、完整 parameter-update field、semantic events、next-state selected components、T1a
与 T1b 全部精确相同，两个 repeats 与 evaluator repeats 也精确。all-eligible margin endpoint 因零 eligible
decisions 继续是 `UNINSTANTIATED` 而不是数值零，所有 boundary taus 则都无 exposure。该 state 对有定义的
core endpoints 贡献真实零 effect，但不能被用来填补另一个未定义的 conditional margin estimand；这进一步
证明“zero effect”和“endpoint 在该 state 未实例化”必须分开。
通用 leave-one-state-out influence diagnostic 随后验证：当前 early 的 all-eligible phase mean 对单 state 不稳，而 middle mean
在新增 step-151 后暂时对单 state 删除保持正向。该诊断逐 phase、逐
tau 分开 continuous/directional-event/disagreement，绝不池化缺失 late phase，也不形成 population claim。
若某 conditional profile 少于三个 exposed states，LOO 后只剩一个 observation；此时即使机械符号不变也
标记为 `stability_interpretable=false`，不能包装成 conditional stability。
同一套 partial influence 现已扩到全部声明的 scalar/event endpoints，并继续保留 endpoint class。23-state
结果不是“一切均值都不稳”：training loss 的 early 仍不稳而 middle 在完成八个 states 后暂时稳定为负；U1 的 early 不稳而 middle
暂时稳定为负；T1b early/middle 暂时均稳定为正；更关键的是 T1a early 的 LOO means 全为正、middle 全为负。这给出一个真实的
phase-conditional cancellation pattern：若只看跨 phase global mean，相反方向可能抵消；若事前把三个 phase
成套纳入 multiplicity，则它们是不同的 conditional estimands。当前没有 late states、只有 calibration-0
一条 trajectory，因此这仍只是 design evidence，不能写成已确认的 phase Bias。
contributor gate 已能解析该身份并从嵌套 confirmation claim 冻结正确符号；repair profile 必须把 records
限制到相同 phase，显式估计 `P(state | phase)` 上的贡献。manifest 同时绑定
`target_state_condition` 与 `primary_estimand.target_phase`，防止确认 early-B 后却用全局 states 做算子归因。
H/N 估计也不再把方差分量逐组截零后当成正式证据：输出保留可能为负的未约束 method-of-moments
值（负值表示当前 repeats 无法分辨该分量）、非负截断描述值，以及以 independent trajectory 为单位的
不确定性。未经单独确认的 H 点估计不会把 B verdict 改写成“state-conditional bias”。
N 也严格沿 trajectory→phase→state 等权聚合，不直接平均全部 state；因此即便各 phase 的 state 数
不同，也不会悄悄改变目标分布。当前 Qwen3 的 8×3 平衡设计不受数值影响，但通用定义不再依赖这一巧合。
每份 estimate 现在还自带 identification assumptions：same-state repeats 的 exchangeability/条件零均值、
independent trajectories、balanced-repeat H correction，以及 paired N 保留 reference/candidate noise
covariance。固定 reduction tree、reassociation、cast placement 属于 `m(state)`；浮点、reduction 或 compiler
optimization 不被预先贴成 bias/variance mechanism。
由于 H 语义、条件证据结构和 materiality verdict 已发生实质变化，population estimator/output 已提升为
`forkcert.bias-oracle-population.v0.2`，并更新 aggregator、confirmation 与 contributor 的依赖哈希；
旧 v0.1 结果不能在同一 schema 名下与新结果混用。
样本量规划只读取四条 calibration trajectory 的 dispersion，不读取 mean/sign；小 pilot variance 先取
单侧上界。U2 不对数亿 coordinates 直接做普通 t tests，推荐用 calibration mean field 冻结一个方向，
再在全新 trajectories 上做 scalar directional replication；这不冒充 full-vector omnibus claim。
方向规划不能直接使用同四条 calibration 在其自身 full-mean direction 上的 in-sample variance；新诊断
对每条 trajectory 用另外三条定方向，再投影 held-out trajectory，使用四个 cross-fitted projections 的
dispersion 加小样本上界与独立 variance floor。full norm 或 LOO stability 未过事前门槛时保持
`UNINSTANTIATED_DIRECTION`，不能从 confirmation 重新选方向。
该路径现已落到同一个 confirmation estimator：direction freezer 固定完整 shard hashes 与 normalization；
每个全新 paired update artifact 计算 `<delta_U, v_cal>`，再按 trajectory→phase→state→repeat 进入普通
B/H/N/U。precision planner 使用 cross-fitted projection dispersion，而非 L2 magnitude 或同向 in-sample
dispersion。门槛未实例化/未通过时 fail closed，不会把 U2 静默移出 endpoint family。

confirmation evaluator shell 已要求：trajectory count 与 precision plan 完全一致；所有 seed、data slice、
capture plan 和 results root 唯一；四条 calibration trajectories 明确排除；adjusted interval alpha 真正
传入最终 trajectory-t interval；calibration 不进入 confirmation df。任一身份或 completeness 漂移均
`INVALID`，不能用部分新轨迹给总体结论。
evaluator 还会从 precision plan 中哈希绑定的 calibration aggregate 与 frozen spec 重新运行 planner，
并要求除 input links 外逐字段完全相等；手写 `valid=true`、缩小 J、改 variance floor/multiplicity 或替换
U2 direction 都不能通过。
四轨迹 calibration aggregate 现在也记录 multi-trajectory aggregator、record loader、population estimator
与 record validator 的独立文件哈希；precision planner 必须验证它们等于当前冻结实现。因而旧算法生成
的 calibration summary 不能在不重新聚合 raw records 的情况下为新版 confirmation 规划样本量。
multi-trajectory aggregate 还必须重新验证并哈希每条 trajectory 的 capture-batch audit、source
config/metadata 和完整 24-state audit census；只有 pair records 有效、但自然 source/history 证据未绑定
时，calibration construction 仍为 invalid。
manifest 同时冻结 planner、population estimator、trajectory sign-flip、record loader 和 U2 projector 的
独立文件哈希；仅保持 evaluator 主文件不变而修改 imported dependency 也会令 confirmation 无效。
confirmation campaign 也只接受通过上述完整 preflight 的 manifest，随后严格串行执行 source→source-binding
audit→24 records→final evaluator；未实例化 manifest、残缺 source、GPU 冲突或低于存储安全线均在继续
采集前 fail closed。最终 evaluator 之前不允许 population-B claim，之后也不会自动启动算子实验。

primary adjusted trajectory-t 若建立 shift，还必须通过冻结的 trajectory-level studentized
Rademacher sign-flip sensitivity；冲突时降级为 `INDETERMINATE_METHOD_SENSITIVITY`。该 sensitivity
只有否决权，不能把 primary 未建立的 shift 重新判成存在，也不增加顶层自由度。
最终 confirmation 还单独检查实际区间半宽是否达到事前冻结的 `desired_half_width`。规划时预计能达到
不等于实际数据已经达到；若新轨迹方差更大，保留 shift-existence 结果，但精度票记为
`INDETERMINATE_REALIZED_PRECISION`，并阻止该 endpoint 进入 operator attribution。practical materiality
没有独立 tolerance 时继续是 `UNINSTANTIATED_MATERIALITY`；检测到 B 不自动等于有害。
每个 global/phase endpoint 的 decision matrix 还显式携带
`correctness=UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY` 与
`long_run_training_impact=UNINSTANTIATED_ONE_STEP_ORACLE_ONLY`；下游不能只取 shift verdict 而丢掉
这两个 claim boundary。
precision planning 现在还检查 sign-flip 的离散 p-value 分辨率：8 条轨迹时最小双侧 exact p 为
`2/256`；若 multiplicity-adjusted alpha 更小，计划必须自动增加 trajectories，否则 fail closed。

confirmation 的 trajectory 抽样也已在完整 calibration 结果出现前冻结：从 112 个声明的非重叠
64-prompt blocks 中排除 4 个 calibration blocks，按固定 SHA-256 rank 得到 108-block reservoir；未来
precision plan 只提供 `J`，builder 只能取 reservoir 前 `J` 个并确定性生成不重复 seed 与每 phase 8 个
steps，不能读取 calibration mean/sign。另一个 freezer 会把 precision、trajectory bank、source configs、
capture plans 和 evaluator hashes 一次性写入 manifest，并在任何 confirmation outcome 已存在时拒绝冻结。

容量测量显示每个完整 matched state 约 27 GiB、每条 24-state trajectory 约 818 GiB。retention policy
因此也在完整 calibration 结果前冻结：每条 calibration trajectory 每 phase 用独立 SHA rank 保留一个
full replay snapshot；其余大 tensors 只有在 24 records、scalar summary、有符号 U2 summary、下游 aggregate
和 tombstone hash ledger 全部通过后才可能进入删除候选。policy 本身不执行删除，当前没有删除原始证据。

### 1. Calibration bank

- 4 条完整 300-step eager-anchored trajectories；
- 每条在 early/middle/late 各预选 8 个 states，共 24 states；
- 总计 96 matched states；
- 每个 state/implementation 两次 repeats；
- states 不按 fork、raw delta、loss 或 operator effect 筛选。

4 条轨迹只用于估计尺度、检查分布形状和确定 confirmation 需要多少新轨迹，不产生最终
population claim。

当前 primary query 是 eager-anchored `P_R/Q-R`。它回答“baseline 会访问的 states 上切换实现”的
relative B；即使独立 confirmation 通过，也不能自动代表 compiled-anchored `P_C`。之后必须对已确认
endpoint 单独运行 Q-C transport，或明确把 operator/impact claim 限定在 `P_R`。

### 2. Independent confirmation

使用 calibration 未出现过的新 trajectory/data/RNG seeds。轨迹数量由 calibration 的
trajectory-level variability 和预声明精度决定，并在看到 confirmation effect 前冻结。

同一轨迹中的更多 states、tokens 或 parameter coordinates不能冒充更多独立 trajectories。

## 每个 state 测什么

### U：一步 update

- compiled-reference parameter update field；
- 无 state-specific 分母的 signed aligned dot `<compiled-reference, reference update>` 作为 primary；
- 原相对投影比例只作 denominator-sensitive 描述量；
- orthogonal magnitude；
- gradient、clip、AMP、optimizer 和 next-state propagation。

U1/U2 都依赖冻结参数坐标与 optimizer scale，不能跨重参数化或训练配置称为 compiler 固有 bias；T1
提供同一 matched step 上较少依赖参数坐标的任务 functional，但仍不是长期训练结论。

### T：一步任务影响

- 独立 frozen rollout bank 上的 post-update GRPO surrogate difference；
- 固定 correct-answer bank 上的 post-update NLL difference；
- 两个 post states 使用同一个 common evaluator，避免评价器实现混淆。

## Oracle 输出

对每个预声明 U/T endpoint 分开报告：

- `B`：trajectory distribution 上的 average shift；
- `H`：effect 随 trajectory/phase/state 的变化；
- `N`：same-state repeat variability；
- `U`：有限 independent trajectories 带来的估计不确定性。

对 clip/skip 等离散 endpoint 还要分开报告 directional shift 与 disagreement。前者是有符号的事件概率
变化，可以在预注册后进入 signed semantic confirmation；后者及两个单向 transition rate 描述语义不一致结构，
即使其平均值很大也不能作为 signed-B endpoint。boundary confirmation 现可将 disagreement 作为
`NONNEGATIVE_SEMANTIC_IMPACT_NOT_B` 独立角色事前冻结，并给出不含 B 字段的 population mean、H/N/U 描述与
semantic-impact verdict。它可进入独立 intervention study，但不得借用 signed-B 的方向或结论名称。

当前 core clip rate 的 population unit 是 state：每个 state 内先除以该 state 的 recorded completion-token
clip-decision positions，再按冻结的 trajectory→phase→state 等权规则聚合；它是 unconditional event-rate
estimand，不是 eligibility-conditioned rate，也不是把所有 token 混合后的 exposure-pooled rate。历史字段
`clip_decision_exposure_count` 实际记录的是 decision-position denominator，名称不得被解释为
`advantages != 0`。与 clipping boundary 对齐的 eligibility-conditioned event rate 属于另立的
reference-anchored boundary family；当某 state 没有 eligible decisions 时该 conditional endpoint 未实例化，
不能用 core unconditional zero 代替。若未来 batch/token 数变化，必须保持各自冻结的分母与层级权重，不能
在看结果后切换 query。

precision template 还显式列出 endpoint role catalog：U/T 是 core signed candidates；training loss、旧的
denominator-sensitive U1 ratio 与 raw clip-count shift 是可选 signed numerical candidates；clip/gradient-clip/
optimizer-skip directional rates 是可选 signed semantic candidates；disagreement 只能进入事前冻结的 nonnegative
semantic-impact role；one-sided rate 和其他 magnitude 默认仍只作描述。planner 要求 catalog 完整且未漂移。可以用 calibration 做 availability/design
discovery，但必须在任何 independent confirmation outcome 之前一次性冻结最终 signed family、phase family
与 multiplicity，不能看结果后增删。

这里的 `B` 只表示目标 matched-state distribution 上的平均局部 implementation shift，不把“导致长期
训练差异的所有机制”重新命名成 Bias。`B != 0` 不保证该方向会在自由训练中累积：后续 dynamics 可能
衰减、旋转或抵消它；`B = 0` 也不保证长期无影响：state-conditioned effects、非线性传播和访问分布改变
仍可能产生长期差异。U/T endpoint 越接近实际 update 或一步任务影响，越能减少纯数值量与训练意义之间
的距离，但仍不是长期因果结论。长期自由训练因此是单独的 validation endpoint；若要从一步转移推导
长期结果，还需要额外的稳定性、状态分布 transport 和扰动传播假设，当前 Oracle 不默认这些假设成立。

可能结论：

1. `REPRODUCIBLE_AVERAGE_SHIFT`：可以进入该 endpoint 的 bias-contributor 分析；
2. `NO_STABLE_AVERAGE_DETECTED` 但 H 较大：只说明 effect 异质；未约束 H、非负截断值和不确定性
   必须并列。只有预注册并独立确认的 `endpoint::phase=<phase>` 才能进入条件 bias attribution；
3. N 较大或其不确定性太宽：Oracle 需要增加 repeats，暂不做 bias attribution；
4. `PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT`：停止该 endpoint 的 global-B contributor 搜索；它不
   声称逐 state、tail event 或两个实现整体等价；
5. `INDETERMINATE/INVALID`：增加独立轨迹或修复 construction，不能挑算子解释。

在 global/phase-conditioned average-shift family 内，最终面向使用者的组合 disposition 不只看“显著/不显著”：shift 通过且实际区间达到冻结精度，才是
`CONFIRMED_IMPLEMENTATION_RELATIVE_AVERAGE_SHIFT`；shift 通过但区间过宽，标为“检测到 shift、但
目标精度未达到”；未检出且精度达标，只能说“在该 floor 与目标精度下未检出”，不能说等价或零效应；
未检出且精度不达标则明确是 `INDETERMINATE`。只有第一种允许进入该 average-shift family 的 contributor
study；它不否决由各自独立 gate 确认的 boundary-conditioned 或 semantic effect。

Practical materiality 是另一条可选轴：只有提供独立于 calibration/confirmation 结果的外部应用容忍阈值，
并在 confirmation 前冻结来源，才允许输出 `MATERIAL_AVERAGE_SHIFT`、
`PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT` 或 materiality-indeterminate。没有这种阈值就保持
`UNINSTANTIATED_MATERIALITY`；测量分辨率、负控制包络不能冒充应用容忍度，materiality 也不能冒充
correctness 或长期训练影响。

## 什么时候进入算子分析

全局平均 B 不是算子分析的总门槛。只有一个**命名且事前冻结的 effect estimand**在 independent
confirmation 上通过它自己的门槛后，才冻结与该 estimand 对应的候选集合。当前允许的入口分为：

1. global signed B；
2. 事前声明的 phase-conditioned 或 reference-boundary-conditioned signed B；
3. signed semantic-event shift；
4. 明确标为 non-B 的 nonnegative semantic disagreement/impact。

四类入口不得共享未调整的检验、方向或 claim 名称。global B 未确认只关闭 global-B contributor study，
不能关闭已经独立确认的 conditional 或 semantic study；反过来也一样。每条路径都先用独立 contributor
pilot（或外部保守依据）冻结贡献量自身的精度，再在最终独立的 contributor-confirmation trajectories 上
做完全相同 effect family、condition mask、denominator 与层级权重下的 repair/injection。

对 signed B 路径，repair 的 primary contrast 可写成：

`bias contribution = full compiled B - repaired compiled B`。

对 signed semantic 路径，primary contrast 是相同事件方向率的 candidate-repair；对 disagreement 路径，
primary target 是 repair 后 reference-candidate disagreement 的减少，必须继续称 semantic-impact
contribution，不能重新命名成 Bias。boundary 路径必须复用 independent confirmation 前冻结的
reference-anchor mask，不能在 repair/injection 后重新选择 near-boundary states 或 tokens。

Injection 是独立 family，不用 repair 结果冒充：在同一冻结 endpoint/方向上同时重测
`full candidate - reference` 的 transport，并估计 `reference injection - reference`。该 profile 与 repair
各自做 trajectory interval 和 veto-only sign-flip sensitivity；二者不要求对称。repair effect 不是必要性，
reference-context injection effect 也不是充分性，均只是在声明 intervention/context 下的 attribution。
Repair contributor precision 的 desired width、directional contribution floor 与 variance floor 同样必须
各自带可审计 `threshold_sources`，并明确不使用 contributor-pilot mean/sign；角色错配或缺失来源时
planner fail closed。当前 precision plan 只覆盖 primary repair
family，injection 不得借用其 alpha、variance 或 trajectory count，必须另冻计划。
Operator candidate-freeze 还必须哈希绑定完整 intervention-unit universe。只有
`EXHAUSTIVE_ELIGIBLE_UNIVERSE` 且全部 eligible IDs 都进入 family 时，才能声称覆盖冻结全集；
`PREDECLARED_SUBSET_OF_UNIVERSE` 的结论强制限制为 `SELECTED_CANDIDATES_ONLY`。source operator、
generated kernel、region 仍是不同 unit types，不能因 census 完整就跨类型改名。universe 还必须声明
scope kind，并精确哈希绑定产生 endpoint-confirmation 结论的冻结 state-bank evaluation；scope kind 与
每个 intervention unit type 不一致、或换成另一个 state bank 时，validator 均 fail closed。这里的绑定只
证明“归因所用总体没有被静默替换”。exhaustive claim 现在还强制链接独立 coverage evidence：逐个
confirmation state 的 census 必须标为 complete，state 顺序/集合等于 confirmation evidence，所有 state
observed-unit union 等于 universe IDs，raw census artifacts 逐一过 hash。允许的表述仅是“冻结 state
bank 中观察到的全部 eligible units”，不是所有可能程序、模型或运行状态；否则仍限制为 selected candidates。
每个 selected operator/kernel/region candidate 还必须绑定 unit-specific structural identity 与完整
state-realization map。map 的 state IDs/order 等于 confirmation bank；每个 state 要么列出 exact call IDs、
count 和 call-order digest，要么显式证明该 unit 未出现。一个 operator family 名、kernel 名或 region 的
constituent 不能借用同一 candidate ID 冒充 exact operator intervention。

在 signed-B contributor study 中，我们只找贡献该 B 的算子，不找贡献 N 的算子；conditional/semantic
study 则只归因其精确定义的 conditional 或 semantic effect，不能借用 global-B 结论。总体 B 的 variance 不能替代 repair contribution 的
variance；候选发现、endpoint Bias confirmation、contributor precision pilot 和 contributor confirmation
不能共用最终确认数据。operator intervention、generated kernel/region intervention 和改变周边
编译选择的 intervention package 必须分别命名；repair effect 仍不自动等于 root cause、必要性或充分性。
完整门槛见 `QWEN3_BIAS_CONTRIBUTOR_CONFIRMATION_GATE_V0_1_2026-07-20.md`。
该门槛已有 fail-closed validator：未确认 B、旧轨迹复用、候选/claim-unit 漂移、precision 或
multiplicity 漂移都会阻止 contributor 实验启动。
contributor precision 已拆成事前 input spec 与 planner 生成的 frozen plan；validator 会从 spec、candidate
artifact 和 pilot summary 重新计算关键字段，拒绝手工伪造或采集后改写的 `valid=true` plan。
当前仍明确保留一个 C3 缺口：真实三臂 raw records 到 hashed contributor-pilot summary 的 provenance
builder/validator 尚未实现；在总体 B 尚未确认的 C2 阶段不提前采集该数据，也不把 synthetic planner tests
冒充真实 contributor pipeline。
结果估计器也已固定为同 state/repeat 的三臂配对：baseline=`candidate-reference`、
residual=`repair-reference`、contribution=`candidate-repair`。它分别报告 contribution 的 B/H/N，并明确
保留 paired covariance；仅有 primary interval 时保持 sensitivity/integrity pending，不能直接给最终因果 verdict。
contributor sensitivity 同样只有否决权；即使通过，仍保持 intervention-integrity pending。

## 当前执行顺序

`calibration-0 source audit → 24 frozen records → signed scalar/vector trajectory summary → calibration-1/2/3 → 4-trajectory calibration → freeze and run independent confirmation → bias contributor → long-run validation`

在 calibration 之前继续扩大 operator repair coverage，不服务于当前核心问题。
