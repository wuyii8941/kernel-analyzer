# Qwen3 Bias Oracle confirmation freeze gate（草案 v0.1）

状态：**procedure shell only，尚未实例化**。本文在 calibration-0 完成前写入，用来限制
calibration 后能做什么；它本身不选择 half-width、trajectory count、tail claim 或 U2 方向，因而
不能据此启动 confirmation。

## 1. 为什么还需要这一层

四条 calibration trajectories 只给出 effect scale、trajectory-to-trajectory variation、H/N 和
construction failure rate。它们不自动给出：

- 多宽的 interval 才算回答了问题；
- 多个 endpoint 是否形成一个联合 claim family；
- 高维 U2 mean field 如何在新数据上确认；
- 是否要声称覆盖 rare trajectory regimes；
- 资源不够时应缩小 claim，还是接受低精度。

这些选择若在看过 confirmation sign 或 significance 后再改，就不再是 confirmation。

## 2. Calibration 后、confirmation 前必须一次性冻结的对象

1. **Scalar endpoint family**：U1、T1a、T1b 中哪些保持 confirmatory；不得因为 calibration
   mean 接近零或符号不好看而删除。endpoint 只能因事前 construction rule（例如大量
   `UNINSTANTIATED`）降级，并保留降级记录。
2. **Multiplicity scope**：
   - 若要作“至少一个 endpoint 有 shift”的联合结论，必须冻结 familywise alpha 和 simultaneous
   interval procedure；
   - 若只作彼此独立、明确命名的 endpoint claims，也必须禁止事后只报告显著 endpoint。
   - global B 与 phase-conditioned B 是不同 claims。若声明 phase family，某个 endpoint 的
     early/middle/late 三个条件 claims 必须成套进入 frozen family 和样本量规划；未声明时三个阶段只作
     descriptive profile，不能看到 global cancellation 后再挑一个阶段进入 operator analysis。
3. **Threshold provenance**：每个 scalar endpoint 的 `desired_half_width`、`variance_floor_sd` 和
   `shift_existence_floor` 必须分别记录 source、description、selection rule 和“不使用 calibration
   candidate mean/sign”的声明。目标宽度可来自独立测量分辨率、负控制或外部科学精度需求；方差下限
   只能来自测量/负控制或外部保守 variance floor；shift floor 只能来自 exact-zero null、测量分辨率或
   负控制。三种角色不能互换，也不能取 observed calibration mean 的方便比例；缺失或错配不得生成
   有效 precision plan。这些 source 只证明测量/规划阈值的来源，不自动定义实际危害。
4. **Optional practical tolerance**：若要作 materiality 或 practical-equivalence claim，必须另给正的
   `practical_tolerance`，并哈希冻结 `EXTERNAL_SCIENTIFIC_TOLERANCE` 来源、description、selection
   rule 和“不使用 calibration mean/sign”的声明。measurement resolution 或 negative-control envelope
   不能冒充应用容忍度；没有独立应用阈值时 materiality 保持 `UNINSTANTIATED`。
5. **Variance planning rule**：只使用四条 calibration trajectory estimates 的 dispersion，不使用
   calibration mean/sign 来降低样本量。小 pilot 的 variance uncertainty 必须进入规划，不能直接把
   sample SD 当已知真值。
6. **Trajectory count 与 resource cap**：一次性冻结；confirmation 中途不得按当前 interval/sign
   无惩罚扩样。
7. **Tail scope**：冻结 `p_min/alpha_tail` 或明确选择 `REGULARITY_CONDITIONAL_ONLY`。普通 mean CI
   不替代 rare-regime coverage。
8. **Seeds/data blocks/state rule**：全部与 calibration 隔离，仍是每条轨迹 3 phases × 8 states，保持
  同一 trajectory-estimate precision regime。

seed/data/state 不能在得到 `J` 后手工挑选。v0.1 已在完整 calibration 结果出现前冻结一个确定性
trajectory-bank design：对 eligible non-calibration blocks 做固定 SHA-256 ranking，precision plan 只提供
`J`，materializer 只取前 `J` 项并按固定 hash rule 生成 seed 与 8×3 steps。该规则不读取 calibration
mean/sign；其机器可读设计见 `QWEN3_BIAS_ORACLE_CONFIRMATION_BANK_DESIGN_V0_1.json`。

precision planner 还必须验证四轨迹 calibration aggregate 中冻结的 multi-trajectory aggregator、record
loader、population estimator 与 record validator hashes。仅哈希 calibration JSON 本身不足以证明其
统计语义；analysis hash 漂移时必须从 raw frozen records 重新聚合，而不是沿用旧 summary。

## 3. 推荐的 scalar precision planning

对每个 endpoint，从四条 calibration trajectory estimates 得到 sample variance `s²_cal`。由于只有
4 条轨迹，不直接把它当真实 variance；先在声明的近似正态/regularity 条件下构造 variance 的单侧
上界 `sigma²_upper`，并与独立 measurement/variance floor 取较大者。然后寻找最小 `J`，使冻结的
simultaneous t interval 计划半宽满足：

`critical(J, alpha_family) * sqrt(sigma²_plan / J) <= w*`。

额外规则：

- `J >= 8` 只是下限，不是充分性保证；
- 多 endpoint 联合覆盖时，planning critical value 必须包含 multiplicity；
- 若所需 `J` 超过 resource cap，返回 `INFEASIBLE_AT_DECLARED_PRECISION`，不能悄悄放宽 `w*`；
- pilot variance 为 0 且没有独立 variance floor 时返回 `UNINSTANTIATED_ZERO_SCALE`，不能自动取 8；
- confirmation 实际 interval 仍只用新 confirmation trajectories，不把 calibration 混入顶层 df；
- 实际 interval 超过 `w*` 时精度票是 `INDETERMINATE_REALIZED_PRECISION`，不是继续采到显著；
  shift-existence、realized precision 与 practical materiality 三张票必须分开，只有前两张都通过才可进入
  针对该 B endpoint 的 operator attribution，而这仍不代表有害或 correctness error。

这只是 regularity-conditional precision design；4 条 calibration trajectories 无法经验保证 heavy-tail
或 rare-mixture coverage。

## 4. U2 的 confirmatory 处理

U2 的完整对象是固定参数坐标中的 mean update-delta field。直接对数亿 coordinates 做普通 t tests，
或检验 observed norm 是否非零，都容易产生维数、selection 和 norm-bias 问题。

推荐的第一版确认方式是 **independent directional replication**：

1. calibration 四轨迹先得到有符号 mean field `M_cal`；
2. 若 `||M_cal||` 未高于冻结的 vector measurement floor，U2 confirmation direction 为
   `UNINSTANTIATED_DIRECTION`；
3. 否则在 confirmation 解盲前冻结 `v_cal = M_cal / ||M_cal||` 及其 artifact hashes；
4. 每个新 state 的 scalar U2 observation 定义为 `<delta_U(state), v_cal>`；
5. 之后按同一 trajectory→phase→state 权重和 trajectory-level inference 确认其 average shift。

这样回答的是“calibration 中发现的固定坐标方向是否在独立 trajectories 上重现”，不是证明完整
高维 mean field 的任意方向都非零。完整 field、norm 和 orthogonal residual 继续描述性报告。若未来要
作 full-vector omnibus claim，必须另行冻结并通过 high-dimensional null/rotating-shift simulation，
不能把 directional replication 冒充 omnibus test。

## 5. Confirmation 分析时禁止的操作

- 根据 confirmation sign 选择单侧方向；
- 只保留显著 endpoint；
- 用更多 states/tokens/coordinates 增加顶层 df；
- 把 calibration 与 confirmation trajectories 合并后称独立复现；
- interval 太宽时无版本地继续追加 trajectories；
- U2 projection 不复现时重新从 confirmation mean 选择方向；
- tail coverage 不足时仍声称 compiler effect 对全部训练 regimes 普遍稳定。

## 6. Gate 输出

冻结后的 machine-readable manifest 至少包含：endpoint family、alpha/multiplicity、每 endpoint `w*`、
variance upper-bound confidence level、variance floor、planned `J`、resource cap、tail scope、U2 direction
artifacts、prospective trajectory-bank hash、confirmation source-config/capture-plan hashes 和所有 evaluator
hashes。freezer 在任一 confirmation source metadata、capture audit 或 state record 已存在时必须拒绝冻结。

只有该 manifest 自洽且在读取任何 confirmation outcome 前写入，confirmation 才能启动。即使随后
确认 average shift，它仍是 implementation-relative semantic/update impact，不是 correctness error。

## 7. 方法依据与适用边界

- Browne, *On the use of a pilot sample for sample size determination*（Statistics in Medicine,
  1995，[DOI](https://doi.org/10.1002/sim.4780141709)）直接讨论了小 pilot 的 sample SD 被当成已知
  sigma 时会低估主研究样本量，并提出使用 sigma 的单侧上置信限。我们借用的是“把 pilot variance
  uncertainty 纳入规划”这一原则；其原文是 t-test power setting，不直接证明我们的 clustered
  trajectory precision formula。
- Liu, Yu and Li, *Multiple-Splitting Projection Test for High-Dimensional Mean Vectors*（JMLR 2022，
  [paper](https://www.jmlr.org/papers/v23/20-1103.html)）说明高维 mean testing 中，用独立 data split
  学习 projection direction 再做低维确认是有正式统计先例的。我们当前建议只是更简单的单次
  calibration/confirmation directional replication，不声称继承该文 multiple-split exact test 的
  guarantees。
- trajectory-level t、few-cluster sensitivity 和 nonstationary block-bootstrap 限制的依据另见
  `BIAS_ORACLE_CLUSTER_INFERENCE_RATIONALE_V0_1_2026-07-20.md`。这些统计文献支持分析结构，不证明
  compiler discrepancy 的机制或危害。
