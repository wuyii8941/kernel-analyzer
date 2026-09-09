# Fused causal attention 家族接入

## 比较对象

在 Qwen3-1.7B 第一层 attention 中，candidate 强制使用 PyTorch SDPA 的 Flash
Attention backend，reference 强制使用 math backend。其余 27 层在两边都强制使用
math backend。输入是 32 个无 padding 的长度 64 文本状态；双方都使用
`attn_mask=None, is_causal=True`，因此比较的是同一个 causal attention 数学目标。

如果 Flash backend 不可用，强制 backend 上下文直接报错，不允许静默退回。运行记录
显示每次模型 forward 有 28 次 attention 调用：candidate 的第 0 次为 Flash，后续
均为 math；reference 的全部调用均为 math。

这个家族不是把已有 softmax backward 换一个名字。被替换边界同时包括 causal
attention 的 score、softmax 和 value aggregation，结论覆盖该 fused boundary。

## 自动执行范围

研究者审核 causal 语义、两个 backend、目标层、参数和固定集合。冻结后，程序自动完成
完整模型 candidate/reference 重放、target backend 核对、重复性检查、layer-0 local
输出、q-projection gradient、实际 AdamW 参数写入、统一统计和报告复算。

第一次交互状态检查误以为进程消失，后来完整结果证明进程仍在另一个可见范围执行。
错误判断和更正均保留；随后启动的后台重试检测到 `raw.json` 后拒绝覆盖并记录非零退出。

## 固定集合结果

32 个 candidate 状态均通过两遍逐项重复检查。后 16 个确认状态：

| 测量位置 | 相对 reference 的总 RMS 差 |
|---|---:|
| layer-0 attention 输出 | 0.1719% |
| `q_proj.weight` gradient | 11.4707% |
| 实际 FP32-master AdamW 参数写入 | 36.9127% |

参数写入的整体 aligned 系数约为 −6.8124%。总 RMS 超过声明的 1% 固定集合范围，
统一输出为 `NON_EQUIVALENT`。这个 profile 展示：局部输出上的小差异可以经过真实
backward 和 cold AdamW 变成很大的参数写入差异。

同一步完整模型 loss 的 candidate−reference 差范围为 −0.01668 到 +0.01337，32 个
状态均值约 +0.000073。正负并不稳定，所以这些 loss 只描述单步 forward 差异，不是
训练质量恶化或改善证据。

## 结论范围

当前结果支持一个新的 fused attention 实现家族和明显的固定集合 training-write
effect。它不支持随机状态总体、warm optimizer、完整训练 loss 或所有 Flash Attention
实现的通用结论。candidate 是 PyTorch CUDA Flash-SDPA，不称为 Triton；它的作用是
验证同一框架也能分析常规高性能实现。

可复算来源：

- `results/property/numerical_coverage_v1/qwen_flash_sdpa_attention_v1/`
- `results/property/numerical_coverage_v1/qwen_flash_sdpa_attention_family_evidence_v1.json`
- `results/property/numerical_coverage_v1/operator_family_report_v9.json`
- `results/property/numerical_coverage_v1/operator_family_table_v3.md`
