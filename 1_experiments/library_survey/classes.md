# 需要定义的规格：按类别汇总

由 `build_tables.py` 生成，不要手改。每个分族的类别见 `families.md` 的「规格类别」列。

| | 规格类别 | 分族 | 入口 kernel | 涉及的库 | sm_86 能跑的分族 |
|---|---|---|---|---|---|
| A | 线性注意力与状态递推 | 103 | 359 | Lightning Attention、flash-linear-attention、lightllm、sglang | 91 |
| B | 分页 / 变长 KV 注意力 | 23 | 67 | LMDeploy、SageAttention、flash-linear-attention、kernl、lightllm、sglang | 21 |
| C | 稀疏注意力、索引器与 KV 压缩 | 53 | 146 | LMDeploy、flash-linear-attention、lightllm、sglang | 35 |
| E | KV cache 写入、读出与 KV 量化 | 13 | 18 | LMDeploy、lightllm、sglang | 5 |
| F | 量化格式与量化 / 反量化 | 41 | 84 | LMDeploy、OpenAI triton_kernels、SageAttention、applied-ai、gemlite、kernl、lightllm、sglang | 15 |
| G | 量化与低比特 GEMM | 13 | 32 | DeepSpeed、LMDeploy、applied-ai、gemlite、sglang | 8 |
| H | MoE：路由、排序对齐、分组 GEMM、EP 分发 | 39 | 61 | LMDeploy、OpenAI triton_kernels、applied-ai、lightllm、sglang | 32 |
| K | 块稀疏矩阵乘 | 3 | 3 | stk | 3 |
| I | 采样、投机解码、哈希与水印 | 12 | 32 | lightllm、sglang | 12 |
| J | 布局、元数据与调度 | 16 | 25 | OpenAI triton_kernels、lightllm、sglang | 16 |
| M | RoPE 与位置编码变体 | 6 | 6 | kernl、lightllm、sglang | 6 |
| L | LoRA | 2 | 14 | sglang | 2 |
| O | 其他 | 5 | 9 | OpenAI triton_kernels、lightllm、sglang | 3 |
| R | 随机 | 15 | 20 | LMDeploy、OpenAI triton_kernels、attorch、flash-attention、sglang、trident | 15 |
