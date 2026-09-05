# Kernel Analyzer

Kernel Analyzer 分析 LLM training 中具体数值实现产生的差异：哪些形成系统性的方向
或缩放，哪些主要表现为波动，以及它们是否处在预先声明的 update 范围内。研究以
Triton 训练计算为重点，同时允许常规 PyTorch/ATen/CUDA 实现成为正式 candidate。
研究从 [FlashAttention 的偏差与训练失败分析](https://arxiv.org/abs/2510.04212) 出发：

```text
具体实现的数值运算
        ↓
数学推导：为什么误差不会公平抵消
        ↓
真实 backward 与 optimizer：偏差怎样进入参数更新
        ↓
统一分析：总差异、方向、缩放与适用范围
        ↓
针对成因的修改与配对训练：检验解释和实际价值
```

数学推导负责解释成因，测量负责检查方向、缩放和总体差异，训练负责验证修改是否
影响质量、效率或稳定性。三者不能互相替代。分析完成不要求结果为阳性，也不要求
出现 loss 分叉或崩溃；只观察到 loss 不同也不能倒推出某种 bias。

研究以手写和编译生成的 Triton 实现为重点，也保留 ATen/CUDA 和混合计算案例。
被测实现和参考实现的角色由实验定义，不由库名或实现语言决定。

## 当前入口

1. [科研主线](docs/current_mainline.md)：我们要证明什么。
2. [案例与证据地图](docs/case_evidence_map.md)：推导、更新和 loss 证据分别在哪里。
3. [实验方法](docs/method.md)：如何比较、如何避免跨协议拼接。
4. [主张边界](docs/claims.md)：已经支持什么，哪些仍不能声称。
5. [全部文档与版本入口](docs/README.md)：历史推导、测量和结果的归属。
6. [统一分析 v1 结果](docs/training_numerical_analysis_v1.md)：由机器记录生成的重采与复算状态。

最新 Liger 全参数小模型实验已经延续到 10000 步：参数相对距离由 19.69% 增至
23.50%，验证 loss 差由 +0.02778 变为 −0.02325。这支持实现引起的轨迹分叉，
不支持持续恶化或已经出现训练崩溃。
[实验设置和数据](docs/liger_single_boundary_collapse_experiment.md)

统一实际写入重采现已完成：Phi 的 AdamW 公式差异在 BF16 参数写入时归零；两个
DeepSeek Triton backward 位置的差异真实写入参数；Liger 写入差异总能量超出范围，
但本轮未确认强共同方向。另一个事前冻结的 Liger 4096 步数据流再次出现轨迹分叉，
其验证 loss 差与两条历史数据流方向相反，因此当前不支持稳定质量改善或恶化。
[机器生成汇总](docs/training_numerical_analysis_v1.md)

首轮覆盖的 1,562 个输出位置、历史长程审计记录数、统一测量的案例数属于不同集合，
不相加为“数学成因与 loss 后果都已闭合的案例总数”。

源码位于 `src/` 与 `scripts/`，实验数据位于 `results/`。
本轮整理保留全部结果、失败记录和数学推导；讲稿由用户单独维护。
