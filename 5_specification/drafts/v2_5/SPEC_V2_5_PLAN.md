# 规格 v2.5 方案：按「参照身份 × kernel 种类 × 决定维度」组织（2026-10-16）

依据：15 个陌生库的清点（1852 个入口 kernel，797 个分族，library_survey 9bb8abe）、v2.4 的 31 个角色，
以及本轮另查的开源工作（Triton-distributed / TileLink / DITRON，Transformer Engine 的 FP8 与 MX 训练缩放策略，
Liger、Unsloth、FlagGems、TritonBench、ONNX 算子规范、StableHLO 参考解释器）。

## 1. 故事的主线

一个 kernel 的规格由三件事决定：

1. **参照身份**：定义从哪里来，决定结论能说到哪一步。
2. **kernel 种类**：它实现的是哪一类运算。
3. **决定维度**：要把哪些选择写定，规格才算有定义。

清点结果表明，种类是有限的（17 种），维度也是有限的（12 个），而且每一种 kernel 只在少数几个维度上需要定义。
所以规格不是按库、也不是按 kernel 一个个写，而是按「种类 × 维度」写一次，多个库共用。

### 1.1 参照身份

| 身份 | 来源 | 比较能得出的结论 | 清点中的占比 |
|---|---|---|---|
| torch 运算 | 公开的 torch 运算或模块，语义由 torch 文档定 | 符合文档定义；文档含糊处登记读法 | 103 个分族，13% |
| torch 组合 | 几个 torch 运算的组合，约定由 kernel 接口确定 | 同上；组合规则由接口定 | 350 个分族，44% |
| 需要定义 | 标准（OCP、IEEE）、论文、模型实现、引擎自带的 torch 参考 | 取决于来源：标准与论文给任务定义；模型实现与引擎参考只能说明「与参照程序一致」，共有的错误看不见 | 329 个分族，41% |
| 随机 | 依赖随机数 | 以 kernel 自身的随机数为条件比较；随机数流本身做统计检验 | 15 个分族，2% |

反向 kernel 的参照是前向参照的梯度；状态梯度是否传递另写（维度 D12）。

### 1.2 十二个决定维度

| 编号 | 维度 | 典型内容 |
|---|---|---|
| D1 | 公式与参数化 | 递推式与变体、打分函数、bias 加在哪里、门控用 ln 还是 log2 |
| D2 | 标量约定 | scale 默认值、eps、theta、温度下限、缩放 margin |
| D3 | 布局与索引 | block table、indptr、页大小、权重打包次序、CSR 元数据、状态池槽位、swizzle |
| D4 | 并列与次序 | top-k 并列、路由并列、原子加次序（参照只能比较集合）、同一专家内 token 次序 |
| D5 | 数值格式 | scale 粒度、scale 舍入方向、元素舍入模式、饱和与 NaN、两次舍入、零点形式、累加类型 |
| D6 | 掩码与对齐 | 有缓存前缀时的因果对齐、滑窗端点、softcap / sink / ALiBi 的形式、填充槽 |
| D7 | 状态语义 | 初始与最终状态的布局、缓存写入位置、环形缓冲、amax 历史窗口 |
| D8 | 近似的参照目标 | 量化注意力比反量化后的精确值还是原输入的 SDPA；分块递推与逐步递推的关系；近似 GELU；投机解码的接受规则 |
| D9 | 随机性 | 以 kernel 自己的掩码或均匀随机数为条件；分布检验 |
| D10 | 非语义部分的排除 | split 个数、调度计划、autotune 配置 |
| D11 | 合法域与尾部 | 形状整除要求、越界读写、空行、seq_len 是否含当前 token |
| D12 | 反向约定 | 参照为前向参照的梯度；最终状态是否参与梯度；dropout 反向以保留掩码为条件 |

### 1.3 十七种 kernel

| 种类 | 清点中的类别 / v2.4 角色 | 需要定义的维度 |
|---|---|---|
| K1 稠密与分组 GEMM（split-K、stream-K、持久化、块缩放 MX/NVFP4、低比特权重 GEMM、块稀疏） | G、K；MATMUL | D3 D4 D5 D11 |
| K2 逐元素与激活（SwiGLU/GeGLU 变体、GELU 近似） | O；ACT | D1 D8 D11 |
| K3 归一化（LN / RMS / GN 变体：eps 位置、偏移、casting） | NORM、BN | D1 D2 D11 |
| K4 稠密注意力（因果对齐、GQA 映射、softcap / sink / ALiBi、滑窗） | ATTN、GQA | D2 D6 D11 D12 |
| K5 分页 / 变长注意力与 KV 缓存 | B、E；KVCACHE、PACK | D3 D5 D6 D7 D11 |
| K6 稀疏注意力、索引器与 KV 压缩 | C；KVCOMP | D1 D3 D4 D5 D6 D7 |
| K7 线性注意力与状态递推 | A | D1 D2 D3 D7 D8 D11 D12 |
| K8 位置编码 | M；ROPE | D1 D2 D3 |
| K9 损失函数（CE 变体、蒸馏与对齐损失、L1 / NLL） | CE、LABELS | D1 D2 D11 D12 |
| K10 MoE（路由、对齐、分组 GEMM、EP 分发） | H；MOE | D1 D3 D4 D5 D7 D11 |
| K11 量化与格式转换，含训练用缩放策略 | F；CAST、SCALER | D2 D5 D7 D8 |
| K12 归约与累积（welford、cumsum、split-K 原子加） | REDUCE、ACCUM、INDEXREDUCE | D4 D5 D11 |
| K13 优化器与训练状态（fused Adam / AdamW、8-bit 优化器状态、EMA、梯度裁剪、loss scaling） | ADAM、SGD、RMSPROP、EMA、CLIP、SCHED | D1 D2 D5 D7 D12 |
| K14 分布式与通信（AG+GEMM、GEMM+RS、EP all-to-all、分布式 flash-decode 的部分结果合并、跨 rank 归一化） | RANKMEAN | D1 D3 D4 D7 D11 |
| K15 采样与投机解码 | I | D1 D2 D4 D8 D9 |
| K16 布局、元数据与调度 | J | D3 D10 D11 |
| K17 LoRA 与适配器 | L | D1 D2 D3 |

清点里没有、但训练一定会碰到的种类，来自 v2.4 的角色和本轮另查的工作：K13、K14、K11 里的训练缩放策略、K9 里的
z-loss 与蒸馏损失、K5 里的序列打包、以及 CKPT（重算）与 STATEIO（保存恢复）两条路径规格。

## 2. 需要定义的 13 类：来源、身份、保证标签

| 类 | 分族 / 入口 kernel | 定义来源与身份 | 主要维度 | 保证标签 | 一次定义可覆盖的库 |
|---|---|---|---|---|---|
| A 线性注意力与状态递推 | 103 / 359 | GLA、DeltaNet、Gated DeltaNet、RWKV-6/7、Mamba-2、Lightning Attention-2 论文（任务定义）；fla 各目录的 naive.py 与 HF 实现（参照程序，用于交叉核对） | D1 D7 D8 D12 | 递推：rigorous_enclosure；分块算法若在精确算术下与递推相等，则 G_π − f 应为零，否则按 declared_approximation 登记 | fla、SGLang、lightllm、Lightning Attention |
| B 分页 / 变长 KV 注意力 | 23 / 67 | 布局由各引擎格式说明定（D3）；定下后写成「按索引取出 KV，再做 SDPA」的 torch 组合 | D3 D6 D11 | rigorous_enclosure | SGLang、lightllm、LMDeploy、SageAttention、kernl、fla |
| C 稀疏注意力、索引器与 KV 压缩 | 53 / 146 | NSA 论文；DeepSeek V3.2 / V4 技术报告与官方代码、FlashMLA；MiniMax 实现（前者任务定义，后者参照程序） | D1 D4 D5 D6 | 打分与 top-k：exact；并列：allowed_set | SGLang、LMDeploy、lightllm、fla |
| E KV cache 写入、读出与 KV 量化 | 13 / 18 | 各引擎的 torch 参考与测试（参照程序） | D3 D5 D7 | 写入位置：exact；量化：按 F 类标签 | LMDeploy、lightllm、SGLang |
| F 量化格式与量化 / 反量化 | 41 / 84 | OCP MX v1.0、NVIDIA FP8 / NVFP4 文档（标准，任务定义）；各库的 torch 对照（参照程序） | D2 D5 D8 | exact（舍入是确定函数）；饱和与 NaN：allowed_set | 8 个库，且被 G、E、C、H 共用 |
| G 量化与低比特 GEMM | 13 / 32 | GPTQ、AWQ 原实现；compressed-tensors 格式说明 | D3 D5 | 反量化：exact；乘法：rigorous_enclosure | gemlite、LMDeploy、SGLang、applied-ai、DeepSpeed |
| H MoE | 39 / 61 | DeepSeek-V3、Mixtral、Qwen-MoE、GPT-OSS 的 HF 实现（参照程序）；各引擎 torch 参考 | D1 D4 D7 D11 | 路由：exact；并列与同专家内次序：allowed_set（集合语义） | SGLang、LMDeploy、lightllm、triton_kernels、applied-ai |
| I 采样、投机解码、哈希与水印 | 12 / 32 | 投机采样论文；各库测试；哈希与水印只能照抄算法（参照程序） | D1 D4 D8 D9 | 给定随机数后：exact | SGLang、lightllm |
| J 布局、元数据与调度 | 16 / 25 | 各库格式说明 | D3 D10 | exact；调度启发式排除 | triton_kernels、SGLang、lightllm |
| K 块稀疏矩阵乘 | 3 / 3 | stk 的 to_dense 后接 torch.mm（参照程序） | D3 | rigorous_enclosure | stk |
| L LoRA | 2 / 14 | LoRA 论文；SGLang 测试 | D1 D3 | rigorous_enclosure | SGLang |
| M RoPE 变体 | 6 / 6 | HF 实现（如 Qwen2-VL 的 mrope）；kernl 自带参考 | D1 D2 D3 | rigorous_enclosure | kernl、lightllm、SGLang |
| O 其他 | 5 / 9 | 对应论文与官方实现 | D1 D2 | 逐个登记 | SGLang、lightllm、triton_kernels |

A、C、F、H 四类合计 236 个分族，占需要定义的 72%。F 类是 G、E、C、H 的公共部件，先定。

## 3. 清点之外的开源工作带来的补充

1. **分布式与通信 kernel 已经在 Triton 里**。Triton-distributed 实现了 AllGather+GEMM、GEMM+ReduceScatter 以及专家并行的
   AllToAll 分发与合并；DITRON 进一步把注意力、MLP、all-reduce 融成持久化 megakernel。这些 kernel 的规格要定：参与者与权重、
   归约在精确算术下与次序无关（只对同一有限贡献集合的纯求和成立）、对称内存的布局、分发与合并的 token 映射、
   分布式 flash-decode 中部分 logsumexp 的合并方式。这正是论文主张里的分布式部分，v2.4 只有 RANKMEAN 一个角色。
2. **训练用的量化带状态**。Transformer Engine 的延迟缩放用上一轮的缩放因子并记录 amax 历史窗口，缩放因子为
   (FP8_MAX / amax) / 2^margin；MXFP8 按行或按列缩放，量化后的张量与其转置在数值上不等价。这些是 D7（状态）与 D5
   的组合，属于 K11，推理侧的 F 类没有。
3. **优化器状态的量化**（8-bit 优化器的分块量化）属于 K13 与 F 类的交叉，训练偏差的直接来源之一。
4. **从同一段程序编译出的 kernel**（Inductor、Helion 一类）：G_π − f 只能检查下降过程，检查不了任务意图，身份要标成参照程序。

## 4. 随机类的处理（清点里待定的问题）

接受「以 kernel 自身随机数为条件」的比较作为语义检查：dropout 以保留掩码为条件，比较保留元素是否等于 x / (1 − p)；
采样以固定的均匀随机数为条件，按逆 CDF 比较。随机数流本身另做统计检验：掩码的保留比例做二项比例检验并报置信区间，
采样结果对目标分布做拟合优度检验。两部分的结论分开报告。

## 5. 清点中的疑似问题说明了什么

清点列出的 15 条疑似问题，按维度归类：

| 维度 | 条数 | 例子 |
|---|---|---|
| D11 合法域与尾部 | 6 | attorch 交叉熵漏最后一块并越界写；flash-attention kernel_fwd 写输出不带 mask；Lightning Attention 要求 n 为 256 的倍数；gemlite MX GEMM 越界读；SGLang identity 漏尾部列；LMDeploy 量化 decode 忽略 seq_len |
| D1 公式 | 4 | applied-ai softmax 反向传反了位置；trident dropout 反向乘 (1 − p) 而非除；trident RMSNorm 的 eps 加在开方外；SGLang softcap 用 exp 算 tanh 溢出 |
| D5 数值格式 | 2 | LMDeploy per_token_quant_int8 远离零舍入而自带参考为就近偶数；triton_kernels MX scale 向上取 2 的幂而非 OCP 的向下 |
| D7 / D12 状态与反向 | 2 | fla HGRN 与 ABC 丢掉最终状态的梯度；RWKV-7 返回最后一个 batch 行而非最后一个 token |
| D6 掩码 | 1 | kernl 屏蔽填充 key 的偏移用错 |

合法域与尾部是最常见的一类，所以每份规格都必须写 D11，工具的测试输入必须包含不是块大小倍数的形状。

## 6. v2.5 要交付什么

1. **合同模板扩展**：在现有字段外加一张 12 维度检查单，每个维度写「已定 / 不适用 / 未定」，未定的就是「未建立」的理由。
2. **共用子合同**：F 类的舍入与缩放、B 类的 KV 布局、H 类的路由与对齐、D4 的集合语义各写一份，各库引用，不重复写。
3. **13 类的定义**，按顺序：F → B → A → H → C → E/G → I → M/O/K/L/J。每类定完，清点中对应的分族转为可审查。
4. **训练侧补充**：K14 通信 kernel 的子合同（参与者、权重、与次序无关的前提）、K11 的缩放状态、K13 的优化器状态量化。
5. **身份标注**进入工具输出：每条结论注明定义来自标准 / 论文 / 模型实现 / 引擎参考，后两者的结论只写「与参照程序一致」。

## 7. 怎样报告覆盖

- 按种类与维度报告：每种 kernel 的每个适用维度，有定义的比例。
- 按分族报告：797 个分族中可审查的数量，以及每批陌生库首次接触即有定义的比例（附二项分布置信区间；连续 59 个全部命中才能说缺失率低于 5%）。
- 不按库数、不按 kernel 数宣称覆盖。

## 8. 不变的边界

执行偏差（K − G_π）对任意 kernel 自动给出，不需要以上任何定义；语义判断需要定义，定义的身份决定结论能说到哪一步；
没有定义时报告「未建立」并写出缺的维度。我们不宣称所有陌生算子的规格都已解决。
