# Operator Oracle v0.1：0–1 判定过程

> 本文把已有 contract 定义压缩为一个可审计的判定过程。这里的“0–1”不是强迫所有观测都给二元答案；它指的是：**只有在 subject、执行身份和合法行为集合都成立时，才允许产生 0/1；否则必须拒绝产生这个 bit。**

## 1. 二元核心

对 operator `o` 的一个已声明输入 `x`，设：

- `Z_C(x,r)`：candidate/compiled 在 runtime 条件 `r` 下的可观察结果；
- `S_o(x)`：外部规范、数学关系或独立 reference 对该输入允许的结果集合、关系或概率律；
- `G(x,r)`：输入、执行路径、配置和 operator 身份证据全部有效。

当且仅当 `G=1` 且 `S_o(x)` 已实例化时，定义：

```text
fail_o(x,r) = 1   当 Z_C(x,r) 不属于 S_o(x)
fail_o(x,r) = 0   当 Z_C(x,r) 属于 S_o(x)
```

若 `G=0` 或 `S_o(x)` 缺失，`fail_o` **没有定义**，不能用 0 代替。

这就是 Oracle 的最小 0–1 核心。它判断的是“是否违反已声明合同”，不是“是否与 eager 数值相同”。

## 2. 产生 bit 之前的四道门

按顺序检查；前一道失败时不继续伪造后一道结论。

| 门 | 必须回答的问题 | 失败输出 |
|---|---|---|
| subject | 判断的是哪个 operator instance、dtype、shape、layout、option 和状态字段？ | `UNINSTANTIATED` |
| identity | 观测确实来自所声称的 compiled operator/region，而非 fallback、另一条路径或观测扰动吗？ | `INVALID`/`INAPPLICABLE` |
| semantics | 合法集合 `S_o(x)` 的独立来源是什么？ | `UNINSTANTIATED` |
| evidence | 当前证据是否足以判断 membership、概率律或总体约束？ | `INDETERMINATE` |

因此，一个可靠 Oracle 必须允许三种非二元结果：没定义、证据无效、证据不足。这不是软弱，而是防止把“没测到错误”偷换成“正确”。

## 3. 四类合同如何产生 0/1

### 3.1 Exact / structural

shape、stride、index、option、dtype conversion、metadata、完整状态字段等有明确关系时，直接检查关系。一个有效反例即可产生 `fail=1`。

### 3.2 Numerical

先由指定精度、舍入模型、高精度 reference 或独立应用误差预算给出 `S_o(x)`，再判断 candidate 是否在包络内。不能先观察 eager/compiled delta，再调 `rtol/atol` 来决定合法范围。

### 3.3 Distributional

`Z_C` 的对象是概率律，不是一次随机 token。0/1 来自“候选分布是否属于允许的分布集合”；有限样本若无法把置信集合放到边界一侧，就输出 `INDETERMINATE`。

### 3.4 Transition

检查声明的完整下一状态关系，例如参数、optimizer state、loss scale、step counter 和 autograd metadata。只检查一个 parameter norm 不能替代完整 transition contract。

## 4. 单点 bit 与总体 verdict 不能混为一谈

单个有效 exact witness 的 `fail=1` 足以拒绝一个 universal claim。反过来，有限样本全是 `fail=0` 只说明“覆盖范围内未见违反”，不能证明所有输入正确。

对 workload/population 合同，先声明输入分布、尾部风险和允许集合 `A_o`，再用完整估计量的同时置信集合 `C_hat_o` 判定：

```text
ACCEPT         C_hat_o 完全落在 A_o 内
REJECT         C_hat_o 与 A_o 不相交，或已有有效 exact witness
INDETERMINATE  C_hat_o 跨过边界
```

`ACCEPT` 必须附带覆盖范围；它不等于形式证明。

## 5. 合同优先级

检查顺序是：

```text
exact structural/option/index/state obligations
    → domain/special-value/exception obligations
    → numerical or distributional envelope
    → workload/risk compatibility
    → downstream impact
```

例如，wrong stride、忽略 `alpha` 或丢失 `requires_grad` 已经违反 exact relation，不能被宽松浮点 tolerance 原谅。反之，一个允许的 top-k tie alternative 即使 raw index delta 很大，也不能判错。

## 6. bias、variance 在判定后的角色

当有多组真实输入和 repeats 时，另行解释 discrepancy：

- average relative bias：在声明输入总体上 candidate 相对 baseline 的平均方向；
- input/state heterogeneity：确定性 effect 随输入或状态改变；
- runtime variability：完全相同输入和配置重复执行仍改变；
- sampling uncertainty：有限输入/repeat 使总体估计不精确。

这些字段解释错误形态和风险集中在哪里。它们不定义 `S_o(x)`，也不能把 `fail=0` 或 `fail=1` 推出来。固定 cast elision、reassociation 或 reduction tree 可以有零 runtime variability，却仍可能是确定性 violation 或合法数值差异。

## 7. 当前真实结果卡

### Card A — H1 显式 bf16 cast 被消除

```text
Subject:     matmul → fp32/bf16/fp32 round-trip → SiLU-like op → sum composite
Contract:    显式 dtype conversion 是可观察语义，不能按 no-op 删除
Identity:    compiled composite 已执行；不能据此唯一归因某个 fusion pass
Evidence:    PyTorch 2.11 与 2026-06-09 nightly 的 compiled 都接近 no-cast 程序，
             与保留 cast 的结果相差约 1.09e-1
Bit:         fail=1
Verdict:     REJECT exact representation-conversion relation
Scope:       covered composite/input；fixed negative 尚缺失
```

判定依据是显式 cast 合同，不是“delta 大所以错”。

### Card B — H3 forward 很接近，但 backward 错

```text
Subject:     slice_scatter(...).sum(same_dim) backward relation
Contract:    完整 backward/scatter placement relation
Broken:      forward max delta 4.77e-7，compiled x.grad 全零 → fail=1 / REJECT
Fixed:       forward max delta反而为 7.15e-7，grad exact match → covered fail=0 / ACCEPT
```

该例证明 raw forward delta 的大小和真正 endpoint 的 0/1 可以反向排序。

### Card C — H6 eager 错而 XLA 对

```text
Subject:     小负数输入的 SELU/ELU
Contract:    稳定高精度 expm1 数学关系
Eager:       小量被消成 0 → fail=1 / REJECT
XLA:         跟随稳定数学结果 → covered fail=0 / ACCEPT
```

该例证明 eager 只能是 baseline；implementation disagreement 本身不能决定哪边错。

### Card D — H4 在当前 T4 上拒绝给 bit

```text
Subject:     bf16 hardtanh boundary backward
Observed:    eager/compiled gradient 都为 1
Identity:    两个 PyTorch 环境都警告 T4 不支持原生 bf16 compilation，并跳过该路径
Bit:         undefined
Verdict:     INAPPLICABLE for intended compiled-bf16 realization
```

结果相同不是通过证据，因为目标 realization 没有建立。

### Card E — eager 与 graph 完全相同但共同错误

```text
Subject:     TensorFlow float32 SELU(-1e-10)
Baseline:    eager = 0
Candidate:   non-XLA tf.function graph = 0
Relative:    eager/candidate delta = 0，relative bias = 0
Contract:    stable expm1 mathematical relation ≈ -1.758099e-10
Bit:         fail=1 for both covered implementations
```

该例证明 bias/variance 分解只能解释实现间差异，不能替代 correctness Oracle。

### 同一 operator/input 上的顺序反转

以 eager 为 baseline，在同一个 `SELU(-1e-10)` 输入上：

```text
candidate A: non-XLA graph，eager delta = 0，truth verdict = REJECT
candidate B: XLA，eager delta ≈ 1.758e-10，truth verdict = covered ACCEPT
```

因此，“eager delta 更小”的候选反而更错。任何只按 eager discrepancy 单调排序的规则都无法复现这个 correctness 顺序；加入 bias、heterogeneity 或 runtime variance 也无法修复，因为 candidate A 的 relative discrepancy profile 可以全部为零。缺少的是独立语义，不是更复杂的差异统计。

### Card F — 真实 large-delta 但合法的 reduction

```text
Subject:     CUDA float32 sum([2^25, 2, -2^25])
Eager:       2
Compiled:    0
Raw:         delta=2，default allclose=false
Contract:    gamma_2 * sum(abs(x)) ≈ 8.000001 的独立分析包络
Bit:         eager fail=0，compiled fail=0
Verdict:     covered ACCEPT
```

该输入和包络在执行前冻结。它证明 equality/allclose 会拒绝一个真实但合法的 compiled 结果。

### Card G — 样本不足时不产生 pass bit

```text
Subject:     compiled multinomial([1,1])，100 draws
Target:      p=(0.5,0.5)，允许 TV <= 0.01
Observed:    53/47
95% CI:      [0.3942, 0.6658]
Allowed:     [0.49, 0.51]
Bit:         undefined at population-law level
Verdict:     INDETERMINATE
```

“53/47 看起来合理”不是等价证明；置信集合跨过边界时，Oracle 必须拒绝输出 0。

## 8. 最小输出

每张最终结果卡至少包含：

```text
subject/signature
input population and coverage
contract source and allowed set S_o(x)
candidate realization identity
validity/applicability
0/1 membership result（若有定义）
population verdict and uncertainty
correctness / compatibility / impact claim level
bias / heterogeneity / runtime variability explanation（若测量）
```

## 9. 当前完成边界

这个过程已经定义了“何时能给 0/1、0/1 代表什么、何时必须拒判”。已有真实案例覆盖 `REJECT`、covered `ACCEPT`、baseline-wrong、shared-wrong、real large-but-conforming、stochastic `INDETERMINATE` 和 identity refusal；规范独立的 synthetic controls 还覆盖 set-valued semantics。

这足以建立和验证 **0–1 判定机制的核心**，但还不是 validated general-purpose operator suite：自然发生的 set-valued candidate pair、更多 floating envelopes、统一 immutable held-out manifest 与跨 family coverage 仍未完成。未完成部分属于适用范围和外部验证，不再是“0/1 代表什么”的定义缺口。
