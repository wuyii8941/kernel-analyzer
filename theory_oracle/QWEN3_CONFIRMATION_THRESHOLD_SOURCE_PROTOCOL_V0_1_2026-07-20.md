# Qwen3 confirmation 阈值来源协议 v0.1

状态：**设计协议，不含数值阈值，不能据此启动 confirmation**。

## 1. 为什么必须单独定义

Oracle 至少需要区分三件不同的事：

1. `shift_existence_floor`：多小的 implementation-relative shift 仍被当前测量系统视为可分辨；
2. `desired_half_width`：独立 confirmation 的区间必须窄到什么程度，才算真正回答了该 endpoint；
3. `variance_floor_sd`：四条 calibration trajectory 偶然给出很小 dispersion 时，用什么保守下限避免
   严重低估确认样本量。

它们不是 practical harm threshold。没有独立应用依据时，`practical_tolerance` 必须继续保持
`UNINSTANTIATED`。统计上可分辨不等于训练上有害，也不等于 correctness violation。

## 2. 禁止的数据依赖

三个阈值及其 selection rule 都不得读取：

- calibration 或 confirmation 中 candidate-reference effect 的均值、符号或显著性；
- 哪个 endpoint 看起来最容易得到非零结论；
- 哪个 phase、state 或 operator repair 看起来最有利；
- 为满足现有 resource cap 而事后放宽的数值。

calibration candidate effects只用于估计 trajectory dispersion、endpoint availability、tail/regularity
诊断和 confirmation 资源可行性；不能反过来定义“值得检出的 effect 有多大”。

## 3. 独立测量负控制

在不读取 candidate-reference outcome 的前提下，对冻结 state bank 建立以下控制：

### C-N1：reference/reference fresh-process control

从同一 matched state 启动两个相互独立的 eager/reference fresh processes，执行与主测量相同的
transition、artifact serialization、post-state reload 和 common evaluator。二者的真实
implementation effect 按 construction 为零；观察到的差异只可用于估计 acquisition/evaluation
pipeline 的 null envelope。

### C-N2：candidate/candidate fresh-process control

对 compiled implementation 做同样的 fresh-process 配对。它检查 compiled runtime realization、cache、
autotuning 或 nondeterministic execution 是否让“同一固定实现”的重跑产生差异。若 compiler-config、
graph-family 或其他已记录 realization identity 漂移，control invalid，而不是把漂移自动解释成 GPU
noise。

### C-N3：identical-post-state evaluator control

同一 post-state artifact 在完全相同的 frozen T1a/T1b bank 上重复进入 common evaluator。该 control
只估计 evaluator 层的分辨率/重复性，不能估计 transition runtime variability，也不能估计 T1a 的
bank-sampling variance。

三个 control 都必须沿 trajectory→phase→state 的同一权重形成 trajectory-level null estimates。只在
一个 state 上为零不能支持总体 measurement floor 为零。

当前 calibration contract 已经为每个 state 采集 eager repeat-1/repeat-2、compiled
repeat-1/repeat-2，并在每个 transition repeat 内采集 common-evaluator repeats。因此 C-N1/C-N2/C-N3
原则上可由这些**同实现配对**直接形成，不需要把 candidate-reference difference 重新包装成负控制，
也不必另跑一套同目的 GPU arms。四条 calibration 完成后应从 raw arm/evaluator records 单独生成
null-control artifact；它只读取 within-implementation/evaluator-repeat contrasts，不读取跨实现 mean/sign。

两次 repeat 只能描述该冻结协议下“在一次可比较重跑中是否观察到变化”，不能给罕见 runtime event
很强的发生率上界。若最终仍全为零，报告应写成 `NO_OBSERVED_WITHIN-STATE_VARIATION_AT_R=2`，不能写成
“runtime variance 不存在”。

## 4. 三种阈值各自如何取值

### shift-existence floor

对每个 endpoint 单独从 C-N1/C-N2/C-N3 中与其测量链相关的 null-control effect 构造双侧保守 envelope，
再与该 endpoint 的可审计数值分辨率下限取较大者。数值分辨率必须由 reference-only quantities、dtype、
序列化精度和 evaluator reduction contract 推导；不能使用 candidate-reference mean/sign。

若所有 null observations 都恰为零，但没有正的解析分辨率下限，则不得把“观察到零”直接变成正的
经验分辨率，也不得假装无限精确。可选择 `EXACT_ZERO_NULL`，但此时它只定义零假设；precision
planning 仍需要独立的正 `desired_half_width` 与 `variance_floor_sd`。

### desired half-width

第一版只承诺 measurement-level shift existence，不承诺应用危害。因而每个 endpoint 的目标半宽应由
上述独立 null envelope/解析分辨率的预先固定倍数给出，使区间能分开“pipeline 自身不可分辨区域”和
可分辨 implementation shift。倍数必须在查看 candidate effects 前写入 control protocol。

如果所得半宽需要超过 resource cap 的 confirmation trajectories，正确输出是
`INFEASIBLE_AT_DECLARED_PRECISION`；不能改用 calibration effect 的某个比例让实验变得可做。

### variance floor

它用于样本量规划，不用于宣称 effect 存在。对每个 endpoint 取以下量的最大者：

- 相关 null controls 的 trajectory-level SD 单侧上界；
- 基于 reference-only endpoint scale 的外部保守 floor；
- 四条 calibration trajectory dispersion 的单侧上界由 planner 另行计算，不在这里冒充独立 floor。

若前两项都无法给出正值且 calibration dispersion 又为零，endpoint 保持
`UNINSTANTIATED_ZERO_SCALE`，不能默认最少八条 trajectory 已经足够。

## 5. endpoint-specific 边界

- `U1_reference_aligned_dot`：控制链必须覆盖 transition、reference-update artifact 和 dot-product
  evaluator；其量纲依赖参数坐标与 optimizer scale，阈值不能跨模型/optimizer 配置复用。
- `T1a_heldout_grpo_shift`：必须固定同一个 state-adaptive bank；当前一个 bank/state 的设计不识别
  bank-sampling variance，因而阈值只适用于 conditional-on-frozen-bank query。
- `T1b_correct_answer_nll_shift`：使用固定 correct-answer bank，可直接建立 evaluator null control，
  但仍不等于长期任务精度 tolerance。
- `U2_calibration_direction_shift`：只有 direction norm/stability gate 通过后才实例化；null control 必须
  投影到同一 frozen direction，不能为 null data 重新选方向。
- semantic directional endpoints：null control 必须保留逐 decision 配对；无 fork 只给出有限 exposure
  下的零观察，不能证明任意 state 上事件概率完全相同。

## 6. positive controls 的位置

注入已知 wrong program、放大数值扰动或强制 decision flip 可以检查 Oracle 是否有能力拒绝明显异常，
但不能用来设定 negative measurement floor、自然 workload prevalence 或 practical tolerance。positive
control 的 effect 大小由注入机制决定，不代表真实 compiler discrepancy 的最小有意义尺度。

## 7. 冻结顺序

1. 在查看任何 confirmation outcome 前冻结 control state-selection、endpoint evaluator、倍数规则和
   artifact hashes；
2. 运行 C-N1/C-N2/C-N3，生成只含 null-control quantities 的 threshold-source artifact；
3. 四条 candidate-reference calibration 只向 precision planner提供 trajectory dispersion，不提供
   threshold 数值；
4. 将 control-derived thresholds、endpoint family、multiplicity、tail scope 和 resource cap 一次性写入
   confirmation precision spec；
5. 若任一 endpoint 缺少可审计的正 half-width/variance floor，则该 endpoint 不得启动 confirmation，
   但必须保留为 `UNINSTANTIATED`，不能静默删除后缩小 multiplicity family。

## 8. 本协议能支持与不能支持的结论

它能支持：在声明的 matched-state distribution、implementation realization 与 measurement pipeline 下，
某个平均 implementation-relative shift 是否超过独立测量系统的不可分辨区域，并以预先要求的精度复现。

它不能支持：该 shift 是否违反数学规范、是否会让长期训练失败、是否具有应用上重要的幅度，或某个
operator 是否为 root cause。这些分别需要 specification/high-precision authority、long-run validation、
external practical tolerance 和独立 repair/injection attribution。
