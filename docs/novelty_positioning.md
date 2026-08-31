# 新颖性边界

## 不能声称什么

本项目不声称：

- 第一次研究 signed numerical bias；
- 第一次区分 bias 与 variance；
- 第一次做 training differential testing；
- 第一次研究 optimizer 与低精度误差的交互；
- 第一次使用 stochastic rounding 消除 bias。

FlashAttention failure、low-bit training、stochastic rounding 和 numerical
stability 工作已经分别研究过这些问题的一部分。

## 本项目要补的空白

现有 tolerance 方法主要在 output 或 intermediate tensor 上判断 candidate 与
reference 相差多大。机制论文通常从一个已知的低精度 failure 出发。

Kernel Analyzer 面向的是另一种测试任务：

> 对一个事前不知道是否有问题的 concrete implementation substitution，在完全
> matched 的 training state 上，测量它经过 actual backward 和 target optimizer
> 后留下的 update effect。

核心组合是：

1. concrete forward + actual backward + declared repair；
2. 先用精确 source/response 分解解释平均方向如何形成；
3. 在 local、gradient、update 三阶段使用同一 matched contrast；
4. 同时报告总误差能量、aligned scaling、剩余 residual mean、效应量和置信区间；
5. 用 protocol-relative equivalence range 定义训练数值等价；
6. 用 paired long-run experiment 分开 direct effect、feedback 和 loss consequence。

这不是“再发明一个 tolerance”，也不是“列一张坏算子名单”。它把训练实现的主要
检查位置从 tensor output 推进到真实 optimizer update，并保留对方向形成位置的
解释。

## Orbit mean 的位置

Orbit mean 只争取一个较窄贡献：对 reduction / summation / reassociation family，
它可能成为 source-stage candidate predictor。必须在冻结 schedules 和 confirmation
states 上证明其方向预测，并通过 BF16→FP32 accumulator 干预同步变化。它不能被
写成一般实现的静态 Oracle。

## 最安全的论文表述

> To the best of our review, we provide an auditable matched-state method for
> measuring implementation-induced effects at local output, parameter
> gradient, and actual optimizer update, and for testing whether residual
> structure that passes ordinary tolerance remains training-equivalent under a
> declared model, state, repair, and optimizer protocol.

更强的“first”表述只有在系统文献检索完成后才能使用。
