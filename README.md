# Kernel Analyzer

Kernel Analyzer 的研究目标是建立一个**机制与工具共同组成的训练数值偏差分析框架**：
以真实 Triton 实现为重点，也支持常规 PyTorch/ATen/CUDA；用有明确条件的数学推导
和统计检验解释实现差异如何形成系统性参数更新，再用针对性修改验证训练后果。
我们争取可复现的训练崩溃机制；没有崩溃时，仍要争取具有实际幅度的 loss/perplexity、
训练稳定性或质量相当条件下的成本变化，不以微小 loss 非同一性替代这一目标。
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
影响质量、效率或稳定性。三者不能互相替代。单项分析有效不要求阳性；但**工具能
报告阴性，不等于论文已完成强机制和训练结果的目标**。只观察到 loss 不同也不能
倒推出某种 bias。

## 四个共同目标

1. **框架、机制和工具都有**：复用现有接入、三阶段测量与报告；不同家族不改判断
   公式，代表案例有具体偏差条件、修改预测和新数据验证。
2. **真实包含 Triton**：主要机制和修改能定位到实际执行的 Triton 计算；常规实现
   同样可被测，混合路径不强行归为 Triton 内部。
3. **统计学理论完备**：针对实际声称的结论交代对象、假设、效应量、区间和错误控制，
   并用同一生产代码验证。固定集合可计算，不等于随机总体理论已经完成。
4. **训练结果有说服力**：优先验证机制相关的崩溃；否则验证预先声明、超出普通波动
   且有实际意义的训练变化。小差异仍保留，不包装成大后果。

这是共同完成目标，不是四项已经实现的声明。文献核查支持这一具体研究联系具有
创新空间，不支持无条件“首次”：[同行对比与创新边界](docs/novelty_positioning.md)。

研究以手写和编译生成的 Triton 实现为重点，也保留 ATen/CUDA 和混合计算案例。
被测实现和参考实现的角色由实验定义，不由库名或实现语言决定。

## 当前入口

1. [科研主线](docs/current_mainline.md)：我们要证明什么。
2. [案例与证据地图](docs/case_evidence_map.md)：推导、更新和 loss 证据分别在哪里。
3. [实验方法](docs/method.md)：如何比较、如何避免跨协议拼接。
4. [主张边界](docs/claims.md)：已经支持什么，哪些仍不能声称。
5. [全部文档与版本入口](docs/README.md)：历史推导、测量和结果的归属。
6. [实际写入分析 v2](docs/training_numerical_analysis_v2.md)：由机器记录核验的复采、新家族和训练结果；旧 v1 保留历史限制。
7. [清单驱动的自动采集](docs/numerical_coverage_execution.md)：完整端点分母、结构绑定、统一采集和未支持项记录。
8. [fused RoPE / position scaling 核验](docs/fused_rotary_position_scaling_audit.md)：真实 Triton
   相同输入比较及 optimizer-state 条件结果。
9. [随机状态超界比例检验](docs/population_exceedance_inference.md)：不依赖能量幅度上界的
   精确有限样本端点及其与平均 Q 的边界。
10. [随机状态总体推断合同](docs/population_inference_contract.md)：平均能量、超界比例和
    固定集合分别能证明什么，以及条件不足时为什么必须不作判断。
11. [算子族与重点问题组证据深度](results/property/numerical_coverage_v1/operator_problem_group_depth_v1.md)：
    自动区分覆盖、update 证据、数学来源、修改验证和训练后果。

整份 2026-09-05 新计划及上述四目标仍未完成。当前清单有 520 个有效位置与 17 个目录家族；
其中新增的 99 个计数来自把此前已经完成的 SiLU/normalization 结果重新接回清单，
不是新增运行或新增问题。该规模已经
提供工具广度，后续不再默认追求更多位置；主要缺口转为随机训练状态的统计含义和少数
重要 Triton 问题的机制—训练闭环。以下复采与语言训练是已完成的选定实验，
不能替代这些缺口。历史 `COMPLETE_BOUNDED_MAINLINE`
标签仅对应当时的有限验收清单；当前审计已明确区分选定实验完成与整份计划完成。

最新 Liger 全参数小模型实验已经延续到 10000 步：参数相对距离由 19.69% 增至
23.50%，验证 loss 差由 +0.02778 变为 −0.02325。这支持实现引起的轨迹分叉，
不支持持续恶化或已经出现训练崩溃。
[实验设置和数据](docs/liger_single_boundary_collapse_experiment.md)

原定六项实际写入重采均已完成，另有一个 Granite 新家族确认。审计发现 v1 的参数写入模拟额外舍入了 update，不能直接
代表目标 AdamW 的实际写入。v2 改为真正执行 AdamW 并读取参数前后变化；已完成的
Phi 复采在声明的 FP32 master 协议下有约 5.47% 的写入差异 RMS。这不是原 BF16
协议的直接复现，也不由能量大小推断 bias 或训练质量。旧结果原样保留并注明限制。
[修正状态与执行入口](docs/training_numerical_analysis_v2.md)

新的 WikiText / 真实 tokenizer 配对确认已完成：16 组新初始化各训练 1024 步，
15 组参考实现的共同验证 loss 较低，原实现减参考的平均差约 +0.000696。
这是声明小模型设置下可重复的小幅训练后果，不单独证明持久 bias 或显著工程收益。
[冻结确认结果](results/property/training_numerical_analysis_v2/language_training_confirmation_iid/summary.json)

同一组全部轨迹已续至 4096 步：32 条轨迹的后期窗口平均直接差异均与第一个窗口
同向，16 组最终均有 loss 分叉；但 14 组原实现 loss 更低，平均差 −0.00132849。
因此支持已测窗口的直接方向与 loss 分叉，不支持持续恶化或每个训练步骤同向。
[自动生成的本轮完整结果](results/property/training_numerical_analysis_v2/final_report.md)

新接入的编译生成 Triton optimizer 家族形成了当前最完整的一条机制与训练证据链。
在 32 个真实 Mamba gradient 上，AdamW8bit 的 block size 从 64 增至 256、1024 时，
moment 与参数写入 distortion 按事前预测严格增大。随后冻结的 8 条非重叠 WikiText
数据流各训练 1024 步；默认 256-block AdamW8bit 相对 FP32 AdamW 的最终评估
loss 差为 8/8 正，均值 `+0.02718`，95% 区间 `[+0.01217,+0.04220]`，
超过预声明的 `+0.01` 门槛。没有崩溃；64-block 的训练改善未确认。该结果只覆盖
一个 checkpoint 和声明设置，不替代跨模型验证或完整总体理论。
[机制、协议与边界](docs/optimizer_update_family_audit.md)

进一步的 component 分解显示，保留 FP32 一阶 moment 可把固定 gradient 上的参数
写入 RMS 从 5.45% 降到 3.21%。但在另外 8 条新训练数据流上，这个修改没有改善
最终评估 loss：`default−modified` 均值为 −0.01070，95% 区间
[−0.02708,+0.00567]。这是一项保留的阴性结果，也表明单步 update 更接近参考不能
取代实际训练验证。

首轮覆盖的 1,562 个输出位置、历史长程审计记录数、统一测量的案例数属于不同集合，
不相加为“数学成因与 loss 后果都已闭合的案例总数”。

最新自动接入还覆盖了 Ministral 的 fused RoPE / position scaling Triton 计算。
同一目标参数的固定集合参数写入差异在 cold、warm 和保持 warm 参数但清空 moments
时分别约为 8.67%、0.45% 和 10.02%。这验证了统一流程可以接入新的真实 Triton
家族，并表明 optimizer state 会显著改变差异能否进入参数；它没有 loss 结果，也不
证明 position scaling 是唯一根因或随机训练状态总体中具有相同幅度。
[核验与结论边界](docs/fused_rotary_position_scaling_audit.md)

当前去重后最值得深入的四组由机器证据表明确记录：AdamW8bit 已确认数学来源、实际
参数写入和非微小 loss 后果，但尚未得到有效修改；fused RoPE 是强固定集合现象，
尚缺唯一来源和训练结果；Liger 已有累加恒等式和轨迹分叉，但不支持持续恶化；
Flash-SDPA 是普通 CUDA 的强对照，尚不是 Triton 机制链。这个排序不由位置数决定。

源码位于 `src/` 与 `scripts/`，实验数据位于 `results/`。
本轮整理保留全部结果、失败记录和数学推导；讲稿由用户单独维护。
