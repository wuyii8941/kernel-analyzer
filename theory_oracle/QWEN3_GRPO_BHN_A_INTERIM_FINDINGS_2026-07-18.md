# Qwen3 GRPO B/H/N：轨迹 A 中期结论

状态：`INTERIM`。这里只分析预声明轨迹 A；不能替代 A/B/C 联合结果。

## 1. 这次实例化在问什么

在同一个 rollout state、相同 token field 和配对 RNG 协议下，比较 eager scorer 与 grad-enabled Inductor scorer 的 512 个 current-token log-probability。每个实现同状态执行两次。

这里的“error”是 implementation-relative discrepancy，不是相对数学真值的 correctness error。B/H/N 分别回答：

- B：跨所采 states 保留下来的平均实现差异有多大；对向量 observable 使用 elementwise mean effect 的范数，避免正负元素相互抵消。
- H：逐 state 的平均实现效应变化有多大，并扣除可识别的 repeat noise。
- N：相同 state 重复执行时，配对实现差异自身有多大波动。
- U：有限 state clusters 造成的估计不确定性；它不是 N，也不是 H。

## 2. A 观察到了什么

- 10 个 state clusters，2 次同状态重复，observable shape 为 512。
- signed global mean shift 约为 `1.13e-5`，但 sign-safe `B_norm` 约为 `1.11e-3`。
- 各 state 的 signed mean effect 横跨负值和正值，范围约为 `[-1.62e-4, 3.34e-4]`。
- repeat-corrected H 为非零；在当前冻结协议下，ref、candidate 和 paired-difference 的观测 N 都为零。
- B 的近似 state-cluster 区间仍较宽；只有 A，不能稳定估计目标 state distribution 上的总体量。

## 3. 对 Oracle 定义的直接意义

轨迹 A 已经否定“把所有元素和 states 的 signed mean 当成唯一 bias”的简单定义：这个标量接近零，不代表实现效应消失；它可能只是元素方向和 state 方向抵消。

因此 operator/observable profile 至少需要同时保留：

1. 有方向的信息，例如 signed summaries 或 mean-effect vector；
2. 不被符号抵消的 B magnitude；
3. state-conditioned effect 及 H；
4. 同状态 repeats 所识别的 N；
5. state-cluster sampling uncertainty U。

这也说明 B/H/N 不是给整个训练过程贴一个数字，而是对预声明 observable 的 discrepancy profile。loss、gradient、update 和语义事件必须分别实例化，再研究它们之间的传播关系。

## 4. 目前绝对不能推出什么

- N 为零不表示所有硬件、kernel、autotuning 或 stochastic protocol 下都没有 runtime variability。
- signed mean 很小不表示“无 bias”，B 非零也不表示会线性累积。
- H 非零只说明实现效应依赖 state；尚未定位到具体 operator，也没有证明哪些 state 是风险 state。
- 这不是 ACCEPT 或 REJECT。由于没有外部可接受边界，verdict 必须是 `UNINSTANTIATED`。
- eager 不是独立真值，因此不能作 correctness 或 compiler bug 结论。
- scorer-level B/H/N 不能证明训练不收敛、精度下降或训练变长；这些属于 transition/long-run validation。

## 5. 下一道证据门槛

轨迹 B/C 用于检查上述结构是否跨独立轨迹保持。B 的预声明 step-29 snapshot 用于在完全相同起始训练状态上比较 loss、gradient 和 parameter update；自由运行的双轨迹差异不能替代这一 matched-state transition comparison。

原始结果：`results/training_step_oracle/qwen3_grpo_heldout_transport_v0_1/a_bhn_v0_2_interim.json`。
