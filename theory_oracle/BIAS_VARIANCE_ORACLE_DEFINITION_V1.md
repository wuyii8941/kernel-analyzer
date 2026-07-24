# Bias-Variance Oracle 定义 v1.0（已被 v2 取代）

> 本文保留为历史审计材料。它把零均值差异称为安全噪声、把非零单步均值
> 直接解释为长期线性累积，并把自由运行 twin trajectory 的时间波动称为
> matched-state H/N；这些结论缺少必要假设。当前规范见
> [v2](BIAS_VARIANCE_ORACLE_DEFINITION_V2.md)。

> 用偏差（B）、异质性（H）、运行方差（N）三个统计量判定算子替换是否会导致训练分叉。

## 1. 问题

在 AI compiler 优化中，我们经常需要替换算子实现（不同 kernel、不同引擎、不同 fusion 策略）。替换后需要回答：**训练会不会因此分叉？**

Raw diff（`allclose`、`max |delta|`）无法回答这个问题，因为：

- 大 diff 可能是良性的（对称噪声，如 reduction 重排序）
- 小 diff 可能是致命的（系统性偏移，每步累积）
- 单次 diff 是标量快照，丢失差异的结构信息

## 2. 三分量分解

对算子 `o`，设 reference 实现为 `Z_R`，candidate 实现为 `Z_C`。在真实训练输入 `x ~ Q` 和运行随机性 `r` 下，定义差异：

```
D(x, r) = Z_C(x, r) - Z_R(x, r)
```

将 `D` 分解为三个独立的统计量：

### B — 偏差（Bias）

```
m(x) = E_r[D(x, r)]          每个输入的条件均值差异
B    = E_x[m(x)]              总体平均偏差
```

B 是**确定性方向偏移**的度量。如果 B ≠ 0，说明 candidate 实现在所有输入上都倾向于往某个方向偏。这是最危险的差异类型，因为它会跨训练步累积：

- 每步产生的 gradient bias ≈ ∂L/∂z · B
- 经过 T 步训练，累积参数偏差 ≈ T · lr · gradient_bias
- 参数偏差线性增长 → 训练分叉

### H — 异质性（Heterogeneity）

```
H = Var_x[m(x)]              偏差随输入变化的方差
```

H 度量**偏差的输入依赖性**。可能出现：

- H ≈ 0, B ≠ 0：所有输入的偏差相同（如固定的 bf16 rounding bias）
- H > 0, B ≈ 0：不同输入的偏差方向不同且近似对称（可能安全）
- H > 0, B ≠ 0：偏差既有系统方向又随输入变化（最需要关注的区域）

### N — 运行方差（Runtime Variance）

```
N = E_x[Var_r(D(x, r))]      相同输入重复执行的差异方差
```

N 度量**纯粹的执行随机性**。确定性算子（fixed reduction tree、fixed cast placement）的 N = 0，即使它有非零 bias。只有涉及真正随机机制（CUDA kernel 调度不确定性、stochastic rounding 等）时 N > 0。

N 通常可以被 SGD 本身的噪声吸收。

### 三个量为什么不能混为一谈

| 场景 | B | H | N | Raw diff | 训练影响 |
|------|---|---|---|----------|----------|
| bf16 rounding bias | ✗ 非零 | ≈ 0 | 0 | 小 | 危险，线性累积 |
| reduction 重排序 | ≈ 0 | > 0 | 0 | 可能很大 | 安全，差异对称 |
| CUDA 非确定性 | ≈ 0 | ≈ 0 | > 0 | 小 | 安全，被 SGD 吸收 |
| 前三者的混合 | ? | ? | ? | ? | 需要分解才能判断 |

Raw diff 把这四种场景混在一起。Oracle 把它们分开。

## 3. 四层架构

### 第一层：测量（Measurement）

对 reference 和 candidate 在 n 个输入上各执行 k 次（k 次用于估计 N），收集原始配对观测 `{(Z_R(x_i, r_j), Z_C(x_i, r_j))}`.

### 第二层：统计特征（Profile）

从原始观测计算 B、H、N。输出一个 `OperatorProfile`：

```
OperatorProfile:
  name:            算子标识
  bias (B):        总体平均偏差
  relative_bias:   B / output_scale
  bias_std_err:    bias 估计的标准误（用于统计显著性）
  heterogeneity:   H (偏差随输入变化的方差)
  runtime_var:     N (运行方差)
  output_scale:    reference 输出的平均绝对值
  n_inputs:        输入样本数
  n_repeats:       重复执行次数
```

### 第三层：传播（Propagation）

算子级别的 B/H/N 在训练步中的传播：

```
forward: operator 输出差异 → loss 差异
backward: loss 差异 → gradient 差异
optimizer: gradient 差异 → parameter update 差异
multi-step: parameter update 差异的累积
```

在整个训练过程中持续追踪（`TrainingOracle`），监控：

- 每个模块的 B/H/N 时间序列
- 参数分歧轨迹（线性增长 = bias 累积，√t 增长 = 随机游走）
- Loss 分歧轨迹

### 第四层：判定（Verdict）

五种判定结果：

```
ACCEPT:          所有指标落在预声明可接受范围内
REJECT:          至少一个指标超出范围且统计显著
INDETERMINATE:   超出范围但样本不足以确认（统计不显著）
INVALID:         测量不满足前提条件（如输入数不足）
UNINSTANTIATED:  没有预声明可接受范围
```

## 4. 可接受范围（Acceptance Criteria）

可接受范围**必须在观测数据之前声明**，不能从被判定的数据中反推。

### 算子级别

```
|relative_bias| ≤ max_relative_bias     偏差占输出的比例上限
heterogeneity_cv ≤ max_heterogeneity_cv 异质性变异系数上限
runtime_cv ≤ max_runtime_cv             运行方差变异系数上限
```

### 训练步级别

```
|loss_bias_relative| ≤ max_step_loss_bias      单步 loss 偏差上限
|param_update_bias_relative| ≤ max_step_param_bias  单步参数更新偏差上限
```

### 统计显著性

当 `|relative_bias| > max_relative_bias` 时，还需检查统计显著性：

```
significance = |B| / std_err(B)
```

- significance > 2.0：bias 是统计显著的 → REJECT
- significance ≤ 2.0：可能是有限样本的偶然结果 → INDETERMINATE

这防止了把"样本均值恰好非零"误判为"系统性偏差"。

## 5. TrainingOracle：跨步监控

TrainingOracle 在整个训练过程中运行，而不是单次快照。

### 每步操作

1. `begin_step()`：清空当前步的捕获缓存
2. 执行 reference 和 candidate 的 forward + backward + optimizer step
3. `end_step(step, ref_loss, cand_loss)`：
   - 计算每个目标模块的输出差异
   - 更新该模块的在线 B/H 估计（Welford 算法）
   - 计算当前参数分歧 `||θ_cand - θ_ref||`
   - 检查 acceptance criteria
   - 返回本步快照

### 参数分歧增长分析

参数分歧 `d(t) = ||θ_cand(t) - θ_ref(t)||` 的增长模式直接反映差异性质：

- **d(t) ∝ t**（线性增长）：存在系统性 bias，每步累积相同方向的偏差
- **d(t) ∝ √t**（平方根增长）：差异是随机游走，零均值噪声
- **d(t) 减速**：差异被训练动力学（learning rate decay 等）压制

Oracle 通过 log-log 回归估计增长指数：

```
log d(t) = α · log t + c
α ≈ 1.0 → 线性增长（bias）
α ≈ 0.5 → √t 增长（noise）
```

## 6. 实验验证

在 MLP + SGD 分类任务上，三个场景：

| 场景 | 扰动 | Oracle 判定 | 参数分歧增长 |
|------|------|------------|-------------|
| A: 相同实现 | 无 | ACCEPT | 0 |
| B: fc2 加 +0.005 偏差 | 系统性 bias | REJECT (step 4) | α=0.70 (线性) |
| C: fc2 加 std=0.005 噪声 | 对称噪声 | ACCEPT | α=0.57 (√t) |

B 和 C 产生相似量级的 raw diff，但：
- B 的 fc2 relative_bias = 2.4%，统计高度显著 → REJECT
- C 的 fc2 relative_bias = 0.036%，不显著 → ACCEPT

这证明 Oracle 能区分 raw diff 无法区分的情况。

## 7. 与 codex v0.1 Oracle 的关系

codex v0.1 定义了一个形式化的合同检验框架（四类合同、五种 verdict、四道前置门）。本定义保留了正确的设计原则：

- 预声明可接受范围（不能从被判定数据反推）
- Fail-closed（证据不足 → INDETERMINATE，不是 ACCEPT）
- 五种 verdict（ACCEPT/REJECT/INDETERMINATE/INVALID/UNINSTANTIATED）
- 统计显著性检查

但做了两个重要改变：

1. **以 B/H/N 分解作为核心判定逻辑**，而不是通用的合同匹配。这更直接地回答"训练会不会分叉"。
2. **增加了训练过程中的持续监控**（TrainingOracle），而不是只做单次测量。

## 8. 实现

核心实现在 `src/forkcert/oracle.py`，主要组件：

```
Oracle                    单次 B/H/N 测量和判定
TrainingOracle            跨步训练监控
OperatorProfile           算子级别的 B/H/N 分解结果
StepProfile               训练步级别的 B/H/N 分解结果
AcceptanceCriteria         预声明可接受范围
VerdictResult             判定结果
```

测试在 `tests/test_oracle.py`（23 个测试）。
实验在 `scripts/training_oracle_experiment.py`。
