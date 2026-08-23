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

## 另外六个模型

Qwen3-VL、Gemma 3、OLMoE、Llama 3.2、Ministral 3、Gemma 4 都有真实实验结果，但它们是定向或 held-out 实验。当前没有足够的统一调用分母，不能把它们写成全算子普查。

## 对 sample completion 的影响

新的 sample completion 协议要求所有案例都使用同一条链：

```text
算子输出差异 -> 参数梯度差异 -> SGD/AdamW 更新差异 -> 32 步参数轨迹
```

旧的 12 个覆盖单元和旧的 3 个 headline case 不能自动计入这个新分母；它们需要按 [`sample_completion_v1`](../results/property/sample_completion_v1/protocol.json) 重新统一导出。当前新协议的统一案例数和统一控制数都记录为 0，避免把不同协议的结果拼成一个虚假的“20 个案例”。
