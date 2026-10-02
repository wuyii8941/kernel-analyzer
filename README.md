# Kernel Analyzer

Kernel Analyzer 分析 LLM training 中具体实现相对声明参考的数值差异，重点支持真实
Triton，也保留 PyTorch/ATen/CUDA 与混合计算。研究问题是：**bias 从哪段计算开始，
为什么形成，来源干预能否改变它，以及这些变化是否影响训练。**

低精度和 optimizer state 是可能的实验条件，不是所有案例的预设成因；QK channel
也不是默认解释。后续优先分析同精度实现选择，既有低精度案例和阴性结果全部保留。
研究借鉴 [FlashAttention 的来源分析](https://arxiv.org/abs/2510.04212)，不要求每个
案例崩溃，也不把 loss 非同一性解释成质量恶化。

**当前口径以 [阶段性总结（2026-10-02）](docs/stage_summary_20261002.md) 为准**：研究问题、
估计目标、已纠正的说法与下一步（自动参照 K_R 的组合求值器）都写在那里；与之冲突的
旧结论以它为准。

## 系统已经能自动做什么？

**在参考语义和执行接入已审核之后，系统能自动测量、分析并汇总指定实现的误差。
目前不是把任意 kernel 文件放进目录就能自动完成训练测试。**

| 输入条件 | 当前能力 |
|---|---|
| 已有 candidate/reference 输出和容差 | 通用逐坐标误差、allclose、非有限值检查 |
| 已接入的家族、输入与训练状态、明确参数范围 | 已有采集器执行声明比较，复用 local / gradient / 实际参数写入分析 |
| 已保存的原坐标统计量和协议 | 统一复算总 RMS、加权缩放、诊断信息与声明范围内的判断 |
| 新家族或不同调用约定 | 仍需审核参考并接入调用、输出及必要的 backward/参数映射 |
| 只有任意算子源码，没有参考、输入或训练接入 | 尚不支持自动推导全部语义和测试环境；不能自动签发训练结论 |

操作说明与可执行检查见 **[算子接入与自动化边界](docs/system.md)**。
当前训练分析入口是 `scripts/run_training_numerical_analysis.py`；
安装后的 `kernel-analyzer analyze` 仍是旧 T1–T4 接口，不能混用两者的完成语义。

对已经有 candidate/reference 可调用包装的新 kernel，最小 bias 检查入口是：

```python
from kernel_analyzer import check_bias
report = check_bias(candidate, reference, make_inputs, samples=32)
print(report["status"])
```

实际状态值为 `SYSTEMATIC_BIAS_CONFIRMED`、`SYSTEMATIC_BIAS_NOT_CONFIRMED` 或
`UNRESOLVED_MEASUREMENT`；入口默认检查输出，可选 `check_backward=True` 检查梯度。
它不自动生成 reference、解释根因或推断训练 loss，详细字段和边界见上述说明。

没有完整 reference 时，现有 `diagnose_kernel` 可以执行调用者声明的性质检查与
合法变体比较，但目前不作总体 bias 判断。家族规格模板、真实案例回归和统计边界的
下一步工作集中在[小工具实施计划](docs/system.md#轻量诊断工具实施计划)，尚未完成的
部分不会被写成现有能力。

## 研究与证据入口

- [当前主线](docs/current_mainline.md)：研究目标与下一轮优先级。
- [实验方法](docs/method.md)及[统计与实验对齐](docs/statistics_experiment_alignment.md)：定义、假设和判断范围。
- [主张账本](docs/claims.md)与[全部案例结论](docs/root_cause_closure_current.md)：根因、平均 bias、训练后果、未解问题和原始记录。
- [全部文档](docs/README.md)：专题推导和历史复现入口。

数值测试、bias 检验、来源解释和训练后果是不同结果。总误差大不证明均值 bias；
相同输入的单计算比较与整段计算区域替换也不能混称为单 kernel 根因。

## 当前最强证据及限制

AdamW8bit 的保存残差能够解释所测 history 的 moment 差异；针对性残差读回得到
独立配对训练改善。8 对确认训练的 OFF−ON 平均验证 loss 差为 +0.0282858，
95% 配对 t 区间为 [+0.0157405,+0.0408311]。范围限于固定 Mamba checkpoint、
评估集、采样协议和 t 推断假设，不证明平均 bias 是唯一原因，也不是跨模型保证。
见[独立确认记录](results/property/result_analysis_v4/iid_training_confirmation/verification.json)。

其他案例提供实现来源、状态依赖或边界证据，不按模型位置重复计数：
Liger 的旧 FP32 顺序实验只有摘要和公式 update；新复采已测得原坐标真实 AdamW
写入，并在声明经验输入库中确认正的方向投影均值，但没有相应长期质量改善证据。
Softmax 保存状态的不一致已定位，尚未证明自然总体的更新均值和稳定质量损害。
Liger 的 10000 步轨迹出现 loss 差异反转，不支持持续恶化。
详见[全部案例结论](docs/root_cause_closure_current.md)和[训练后果说明](docs/liger_single_boundary_collapse_experiment.md)。

已有覆盖证明工具不止服务于一个案例，但有效位置数不是独立 bias 数量。
[自动采集记录](docs/numerical_coverage_execution.md)保留成功、超时、缺参考与路径不匹配；
目录清单和旧 COMPLETE 标签不代表任意 kernel 都已支持，也不代表根因研究完成。

## 目录与维护

`src/` 保存公共测量和统计代码，`scripts/` 保存执行与复算入口，`tests/` 保存验证，
`results/` 保存协议和实验结果。使用已有研究环境；系统默认 Python 不一定包含 PyTorch。
近期单步探针、独立训练确认、失败记录与工具演示分别从[结果索引](results/README.md)
进入；实验总数和逐案例状态只维护在案例总表，不在多份文档同步抄写。
新任务的输出、缓存和临时文件均放在本仓库内，不写入 `/home`。

重复的历史状态说明可以删除；原始结果、数学推导、阴性与失败记录，以及仍有调用者的
历史代码保留。研究者维护的讲稿不随自动整理改写。
