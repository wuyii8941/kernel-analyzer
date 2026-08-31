# 模型和算子覆盖总表

这张表专门解决“有结果”与“完成全算子普查”混在一起的问题。机器可读版本是 [`coverage_table_v1.json`](../results/coverage/coverage_table_v1.json)。

## 已经有明确分母的部分

四个模型做了系统性普查，每个模型测 64、128、256 三种长度，共 12 个模型–长度单元：

- Qwen3-1.7B
- Phi-4-mini
- DeepSeek-R1-Qwen3-8B
- Mamba-130M

这 12 个单元合计有 466,419 个 eager 调用、70,171 个 candidate 调用和 186,807 个 F+B proof units。全坐标 T1 审计有 1,562 个 endpoint：1,390 通过、172 拒绝、0 个待处理。

这些数字证明了“核心四模型的覆盖分母”，但不代表每个 endpoint 都已经完成 32 步 bias 测量。

## 不同数字分别在数什么

| 数字 | 含义 | 能支持什么 |
|---:|---|---|
| 466,419 | 普通执行中记录到的前向调用 | 说明实际训练步的调用规模 |
| 70,171 | 待检查的编译实现中的算子调用 | 说明编译实现的执行规模 |
| 186,807 | 一段前向计算及其真实反向计算组成的数学单元 | 说明前向与反向绑定的规模 |
| 1,562 | 进入第一轮数值检查的具体张量位置 | 这是全量第一轮检查分母，不是深度轨迹分母 |
| 804 | 按生成代码区域和负责反向计算的单元得到的分组 | 用于减少重复实现的深测工作 |
| 791 | 后续搜索中的“算子区域 + 反向路径”组合 | 属于另一轮搜索，不与 804 一一对应 |
| 493 | 791 个组合去重后得到的不同实现模式 | 用于后续不重复地选择候选 |
| 6 | 历史 strict repair/carrier/trajectory records | 证据历史计数 |
| 43 | 当前 machine audit 中同时有 long-run bias evidence 与 paired loss split 的行 | 分为 3 条 late-window direct、8 条 aggregate direct 和 32 条 feedback-sustained；只有 4 条现有显式 late-window confirmation |
| 105 | 更宽的 outcome-relevant rows | 包含 persistence 尚未测量的历史候选，不能全算 final persistent cases |

因此，“全量覆盖”只表示 1,562 个 endpoint 都完成了 F+B 绑定和第一轮数值处置。只有继续通过单算子修复、参数可达和轨迹门槛的少量记录，才进入 32 步实验。

## 另外六个模型

Qwen3-VL、Gemma 3、OLMoE、Llama 3.2、Ministral 3、Gemma 4 都有真实实验结果，但它们是定向或 held-out 实验。当前没有足够的统一调用分母，不能把它们写成全算子普查。

## 对 sample completion 的影响

新的 sample completion 协议要求所有案例都使用同一条链：

```text
算子输出差异 -> 参数梯度差异 -> SGD/AdamW 更新差异 -> 32 步参数轨迹
```

这段 sample-completion 快照保留作历史审计，不能覆盖当前长程审计。当前机器口径见 `results/property/declared_persistent_4096/all_bias_case_audit.json`：23 个唯一主矩阵 ID、301 条逐行记录；43 条有 long-run bias evidence 与 paired loss split，但只有 4 条目前有显式 late-window confirmation。更宽的 105 条 outcome-relevant records 含 persistence 尚未测量的历史候选，不能全部叫 final persistent cases。不同协议仍不能把记录简单相加，未决记录不改判为阴性。
