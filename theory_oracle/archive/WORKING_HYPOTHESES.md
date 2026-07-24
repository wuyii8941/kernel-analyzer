# Working Hypotheses

以下假设用于组织实验，不是项目已经证明的事实。

## H1：固定 compiler choice 主要产生确定性的状态条件差异

### 直觉

固定的 reduction tree、reassociation、cast placement 或 kernel 实现，在完全相同 state 上可能稳定复现；但 discrepancy 的大小和方向会随 state 改变。

### 支持它的观察

- 同一 state 重复执行时，implementation discrepancy 基本不变；
- 不同 states 上 discrepancy 明显不同；
- states 的某些特征能够解释这种差异。

### 推翻或削弱它的观察

- 同 state 重复执行的波动与跨 state 差异同量级；
- state 特征和 discrepancy 之间没有稳定关系；
- 换轨迹后条件结构完全消失。

## H2：全局 numerical mean shift 不是 semantic drift 的充分预测量

### 直觉

全局正负 discrepancy 可能互相抵消，但靠近 decision boundary 的 discrepancy 方向可能高度一致；反过来，大的平均 shift 也可能全部发生在远离边界的 states 上。

### 支持它的观察

- global mean 接近零，但 event probability 明显改变；
- 加入 boundary distance 和 discrepancy direction 后，对 event fork 的解释或排序明显优于 raw delta；
- 大 numerical delta 经常没有 semantic effect。

### 推翻或削弱它的观察

- raw numerical delta 已经与 semantic effect 完全一致；
- boundary information 没有带来额外解释力；
- 所谓 boundary-conditioned effect 只存在于事后挑选的 fork states。

## H3：部分 compiler configuration 会产生稳定的 boundary-aligned shift

### 直觉

某些实现选择不一定产生明显的全局 mean shift，但可能在 near-boundary states 上更常将结果推向同一个语义方向。

### 支持它的观察

- near-boundary states 上 directional semantic shift 跨重复、轨迹和样本稳定；
- effect 不能仅由两边独立运行噪声解释；
- effect 对某类 compiler intervention 有稳定响应。

### 推翻或削弱它的观察

- 两个方向的 fork 完全对称；
- directional shift 对 state sampling 极度敏感；
- 换一条合理轨迹后方向反转或消失。

## H4：GPU execution nondeterminism 主要增加 disagreement，但不一定产生 directional shift

### 直觉

若同状态噪声大致对称，它可能增加两个方向的 fork，从而降低路径复现性；只有当噪声分布、reference margin 分布或决策机制不对称时，才进一步产生有向语义效应。

### 支持它的观察

- repeated execution 中两个方向 disagreement 同时增加；
- directional shift 较小，但 disagreement 明显；
- 固定 nondeterministic source 后 disagreement 明显下降。

### 推翻或削弱它的观察

- runtime randomness 稳定产生单方向 event shift；
- 大部分 disagreement 实际来自未冻结的 algorithmic RNG 或 state，而不是 GPU execution。

## H5：一步 event drift 不必然转化为一步 update drift

### 直觉

语义事件改变后，后续计算可能放大、抵消或完全忽略该变化。因此 fork 不能自动代表训练影响。

### 支持它的观察

- 部分 event forks 的 gradient/update effect 很小；
- 另一些很小的 numerical discrepancies 通过事件边界产生明显 update effect；
- event effect 和 update effect 的排序不完全一致。

### 推翻或削弱它的观察

- event fork 与 update discrepancy 几乎存在一一对应；
- raw numerical delta 已能同样准确预测 update impact。

## H6：长期训练影响由局部 drift 与训练动力学共同决定

### 直觉

单步差异不是简单逐步相加。它可能被 optimizer、normalization 和后续 gradient 吸收，也可能通过状态反馈被放大。

### 支持它的观察

- 相似的 one-step effect 在不同训练阶段产生不同长期后果；
- 长期影响取决于 discrepancy 与 gradient、曲率、clipping 或 optimizer state 的关系；
- 局部 effect 的存在不保证最终质量变化，但在特定稳定性条件下能预测风险。

### 推翻或削弱它的观察

- 所有长期差异都能由简单累加的局部 mean shift 解释；
- one-step semantic/update drift 与长期结果没有任何稳定关系，此时长期预测 claim 应放弃。

## 当前最重要的反例意识

- deterministic 不等于 global bias；
- floating point 不等于 variance；
- reduction 既可能确定，也可能随机；
- zero-mean numerical noise 不保证 zero semantic shift；
- persistent numerical shift 不保证长期训练变差；
- repair 成功不等于找到唯一 root cause。

