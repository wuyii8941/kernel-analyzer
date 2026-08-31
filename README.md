# Kernel Analyzer

Kernel Analyzer 检查一个具体 LLM training implementation 相对声明 repair 的数值
差异，怎样经过真实 backward 和目标 optimizer 进入参数更新。

当前科研主线是：

```text
matched candidate / repair
        ↓
方向形成的两项来源
        ↓
local output → parameter gradient → optimizer update
        ↓
效应量、置信区间与短程排序
        ↓
4096-step paired consequence
```

项目不再把一个 tensor tolerance、一个 16/32 步分数或最终参数距离当成完整的
训练正确性结论。Orbit mean 只作为 reduction / summation / reassociation 类实现
的 source-side candidate predictor，不是通用静态 Oracle。

请从以下文档开始：

1. [当前科研主线](docs/current_mainline.md)
2. [统一实验方法](docs/method.md)
3. [证据账本](docs/claims.md)
4. [三类实现的统一测量结果](docs/three_mechanism_profiles.md)
5. [当前文档入口](docs/README.md)
6. [长程机器审计](results/property/declared_persistent_4096/all_bias_case_audit.json)

`docs/all_bias_long_horizon_audit.md` 是便于阅读的逐行表；当它正在整理时，不覆盖
机器 JSON 的计数与标签。

当前覆盖范围为 Qwen3-1.7B、Mamba-130M、Phi-4 和 DeepSeek-R1-Qwen3-8B，
每个模型包含序列长度 64、128 和 256。1,562/1,562 个 concrete output positions
完成了 F+B 绑定与首轮数值处置；这不表示 1,562 个位置都完成了 32 或 4096 步
训练实验。

当前长程机器审计包含 23 个唯一主矩阵 case IDs、301 条逐行记录。机器标签中有
43 条同时具备 long-run bias 证据和 paired loss split：3 条 direct cases 已有后半程
窗口，8 条有整段 long-run direct evidence 但尚未单独导出后半程窗口，32 条为
feedback-sustained cases。只有 4 条目前具备显式后半程窗口确认。另有 105 条记录
属于“已有 bias 候选证据且训练结果受到影响”的更宽口径，其中包括尚未测量
4096-step persistence 的历史候选，不能全部叫 final persistent cases。不能安全重放
的记录保持 unresolved，不改成 negative。

源码位于 `src/` 与 `scripts/`，结果位于 `results/`。任何清理操作都不得删除
机器可读实验结果。
