# Matched-State B/H/N Oracle 定义 v2

## 1. 一句话

在预先声明的 state distribution、reference/candidate、observable、重复执行与
随机耦合协议下，测量 implementation-relative discrepancy，并分开报告：

- `B`：跨 states 的平均效应；
- `H`：条件均值效应随 state 的变化；
- `N`：同一 state 重复执行时，配对差值的运行变化；
- `U`：只抽取有限 states 带来的 sampling uncertainty。

这四项构成 discrepancy profile。它们本身不等于 correctness、安全性或长期训练结论。

## 2. B 不是未经限定的“编译器 bias”

令 `D(s,r)=Y_C(s,r)-Y_R(s,r)`。对给定 state `s`，
`m(s)=E_r[D(s,r)]`；目标分布上的平均效应为 `E_s[m(s)]`。

只有 reference 是独立真值时，才能把它称为 numerical bias。通常 eager 只是
baseline，因此更准确的名称是 average implementation-relative shift。

对张量输出，不能先对元素做有符号平均。确定性差值 `[+1,-1]` 的元素均值为零，
但并不是零效应。v2 至少保留完整 elementwise mean，并用其 RMS/norm 作为默认
无方向摘要。只有预先声明了 decision margin、loss projection 或其他标量 functional
时，才报告有方向的 scalar B。

## 3. H、N 和 U 必须分开

- `H = Var_s(m(s))`：state-conditioned effect heterogeneity。
- `N = E_s Var_r(D(s,r))`：给定 pairing/coupling 下的差值运行方差。
- `U`：对 B/H/N 的有限 state-sample 估计误差。

有限 repeats 会把 `N/R` 混入观测到的 state-mean variance，因此 H 需要噪声修正。
同时应报告 reference 与 candidate 各自的 repeat variance；只报差值方差会隐藏两边
相关噪声。没有至少两次 same-state repeats 时，N 未被识别，不能填写为零。

“浮点误差是 variance”“reduction 是 bias”都不是分类规则。固定 reduction tree、
cast placement 和 reassociation 通常产生确定性的 state-conditioned effect，可能贡献
B 或 H；只有执行本身重复变化时才贡献 N。

## 4. 四层 Oracle

### 测量层

保存原始配对输出、完整 state identity、implementation/execution identity、RNG 与
coupling protocol。shape、调用次数、fallback 或 state 配对失败返回 `INVALID`。

### Profile 层

对每个预声明 observable 产生 B/H/N/U。一个 operator 可以有多个 observable：
输出张量 norm、decision margin、loss contribution、gradient projection 等。不存在一个
对所有训练语义都充分的 operator scalar。

### Transition 层

从同一完整 pre-step state 分别执行 reference/candidate，测量 loss、gradient、离散
事件、optimizer/scaler/scheduler 和 parameter update。只恢复模型、使用 fresh SGD
或遗漏 compiler history 都不是 natural transition。

### Verdict 层

五种结果保持分开：`ACCEPT / REJECT / INDETERMINATE / INVALID /
UNINSTANTIATED`。ACCEPT 是相对于预声明阈值的等价性结论：置信上界必须位于阈值内。
“点估计没超过阈值”或“未显著拒绝零”都不足以 ACCEPT。

correctness、semantic impact、transition impact 和 attribution 使用不同 ledger；
不能用 B/H/N 的一个 verdict 代替全部。

## 5. 多模块与跨步

多模块 profile 必须在同一批 matched states 上收集。下游模块的差异同时包含误差
生成与传播，不能仅凭 hook 先后顺序称某个模块为 root cause。operator causality 仍需
保持 realization identity 的 repair/injection 或更强干预。

跨步有两种不同对象：

1. 从训练轨迹中抽取多个冻结 pre-step states，在每个 state 上做 paired one-step probe；
   这些 states 可以估计目标 state distribution 上的 B/H/N/U。
2. 让 reference/candidate 自由运行；第 1 步后状态不同，因此只能研究 trajectory
   divergence，不能继续把 hook difference 当 matched-state operator B/H/N。

参数距离看似 `t` 或 `sqrt(t)` 增长不能单独识别 bias/noise。非线性 Jacobian、学习率、
梯度方向相关、边界事件和反馈都可改变增长指数。长期训练属于 validation 或独立动力学
问题。

## 6. 当前实现边界

`src/forkcert/oracle.py` 当前提供：

- balanced input×repeat grid 检查；
- tensor sign-cancellation-safe 的 B norm；
- repeat-noise-corrected H；
- paired-difference、reference、candidate runtime variance；
- state-sampling interval 与等价性式 verdict；
- matched-state `TrainingOracle`，联合汇总多个 operator 与 step states；
- `TwinTrajectoryMonitor`，明确只做自由轨迹描述。

置信区间目前使用 state-cluster normal approximation，是初始估计方法，不是所有高维、
重尾或相关 state population 下的最终统计方案。真实研究结果应使用与 sampling design
相符的 cluster/bootstrap 或其他预声明方法。

## 7. 当前真实实验

Qwen held-out transport bank 使用三个事先抽取的 prompt/RNG 轨迹，每条包含十个
grad-enabled matched states、每实现两次重复和 tracked compiler identity。它用于检验
真实 eager/compiled B/H/N/U 的 transport，不用于证明 eager 正确。

B 轨迹 step 29 还会捕获 optimizer、scheduler、AMP scaler、Python/NumPy/Torch RNG、
实际 minibatch 与 compiler-history inputs。只有 snapshot audit 通过后才运行自然一步
transition comparison。

## 8. Kill criteria

- B 仅因 state/element 加权改变符号或消失；
- H 与 N 无法在 repeats 下区分；
- ACCEPT 只来自点估计或任意阈值；
- 多模块排序与 raw delta 完全相同且没有 transition/semantic 增量信息；
- matched probes 无法跨独立轨迹复现；
- operator profile 无法解释或预测同状态 transition effect；
- repair/injection 改变 fusion/layout，导致 attribution 只依赖 intervention；
- 没有独立 specification，却仍声称 correctness 或 bug；
- 自由训练轨迹被用来反推局部 compiler effect；
- 所谓 benign noise 在合法 negative control 中造成事件或 update drift。

